"""Teacher-vs-generated Answer segmentation evaluation."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import timedelta
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within
from reason3d_runtime import restore_task_state, load_segmentation_config, build_segmentation_dataset, OBJECTIVE, ObjAncSegmentationModel
from reason3d_no_fallback import evaluate_no_fallback, merge_no_fallback_payloads
from reason3d_eval_subsets import source_indices, merge_source_payloads
from ground3d_migration.eval_shards import publish_shard,wait_for_shards
from spatiallm_compressed_visual_objanc_full_attention.data import collate_samples
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from spatiallm_compressed_visual_objanc_full_attention.model.curriculum import normalize_phrase


def load_checkpoint(model, path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("objective") != OBJECTIVE or checkpoint.get("architecture_signature") != model.architecture_signature():
        raise ValueError("Evaluation requires a matching native-superpoint Reason3D checkpoint")
    if checkpoint.get("base_manifest") != model.answer_lm.source_manifests:
        raise ValueError("Evaluation frozen base manifest differs")
    if checkpoint.get("tokenizer_snapshot") != model.answer_lm.tokenizer_snapshot():
        raise ValueError("Evaluation tokenizer differs")
    state = checkpoint.get("trainable_state")
    if not isinstance(state, dict) or set(state) != set(model.trainable_state_dict()):
        raise ValueError("Checkpoint trainable-state keys differ")
    restore_task_state(model,state)
    return checkpoint


def mask_iou(logits, truth, threshold):
    prediction = logits.sigmoid() >= threshold
    truth = truth.to(prediction.device, torch.bool)
    return float(((prediction & truth).sum().float() / (prediction | truth).sum().clamp_min(1).float()).item())


class MetricBook:
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = Counter()

    def _add_value(self, key, value):
        self.sums[key] += float(value)
        self.counts[key] += 1

    def add(self, role, selected, matched, oracle, selected_q, matched_q, best_q, rank, topk):
        for prefix in ("all", role.lower()):
            self._add_value(f"{prefix}/selected_iou", selected)
            self._add_value(f"{prefix}/matched_iou", matched)
            self._add_value(f"{prefix}/oracle_iou", oracle)
            self._add_value(f"{prefix}/best_query_rank", rank)
            self._add_value(f"{prefix}/qmatch", float(selected_q == matched_q))
            self._add_value(f"{prefix}/best_iou_query_recall", float(selected_q == best_q))
            for k, value in topk.items(): self._add_value(f"{prefix}/top{k}_oracle_iou", value)
            for level in (.25, .5):
                self._add_value(f"{prefix}/selected_acc@{level:.2f}", float(selected >= level))
                self._add_value(f"{prefix}/oracle_acc@{level:.2f}", float(oracle >= level))

    def summary(self):
        return {
            key: self.sums[key] / self.counts[key]
            for key in sorted(self.sums)
            if self.counts[key]
        }

    def state_dict(self):
        return {"sums": dict(self.sums), "counts": dict(self.counts)}

    def merge(self, state):
        for key, value in state["sums"].items():
            self.sums[key] += float(value)
        self.counts.update({key: int(value) for key, value in state["counts"].items()})


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["evaluation_index"] = int(index)
        return sample


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation without padding or duplicating validation samples."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        timeout = int(os.environ.get("EVAL_COLLECTIVE_TIMEOUT_SECONDS", "7200"))
        if timeout <= 0:
            raise ValueError("EVAL_COLLECTIVE_TIMEOUT_SECONDS must be positive")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group("gloo", timeout=timedelta(seconds=timeout))
    return rank, world_size, local_rank, distributed


def evaluate_mode(model, loader, mode, threshold, examples_limit, *, use_amp):
    book, examples = MetricBook(), []
    answer_counts, fallback = Counter(), Counter()
    eval_rank = int(os.environ.get("RANK", "0"))
    total_batches = len(loader)
    print(f"[eval rank={eval_rank}] {mode} start: {total_batches} batches", flush=True)
    for batch_index, batch in enumerate(loader, 1):
        if batch_index == 1 or batch_index % 100 == 0:
            print(f"[eval rank={eval_rank}] {mode}: {batch_index}/{total_batches}", flush=True)
        device_type = next(model.parameters()).device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_amp and device_type == "cuda",
        ):
            result = model(
                batch,
                epoch=0,
                optimizer_step=0,
                steps_per_epoch=1,
                answer_mode=mode,
            )
        decoder = result["decoder_output"]
        for b, roles in enumerate(result["entity_roles"]):
            mapping = batch["point_to_superpoint"][b].to(decoder["pred_masks"].device, torch.long)
            target_rows = result["target_rows"][b]
            raw_truths = batch["gt_point_masks"][b][target_rows]
            query_masks = decoder["pred_masks"][b, :, :batch["gt_superpoint_masks"][b].shape[1]][:, mapping]
            scores = result["query_entity_scores"][b, :, :len(roles)]
            assignment_source, assignment_target = result["assignments"][b]
            matched_by_target = {int(t): int(q) for q,t in zip(assignment_source, assignment_target)}
            for entity, (role, truth) in enumerate(zip(roles, raw_truths)):
                ious = [mask_iou(query_masks[q], truth, threshold) for q in range(query_masks.shape[0])]
                order = torch.argsort(scores[:, entity], descending=True).tolist()
                selected_q, best_q, matched_q = order[0], max(range(len(ious)), key=ious.__getitem__), matched_by_target[entity]
                rank = order.index(best_q) + 1
                topk = {k: max(ious[q] for q in order[:k]) for k in (1, 3, 5, 10)}
                book.add(role, ious[selected_q], ious[matched_q], ious[best_q], selected_q, matched_q, best_q, rank, topk)
        for source in result["answer_sources"]: fallback[source] += 1
        measured_answers = [
            attempted if mode == "generated" and attempted is not None else chosen
            for attempted, chosen in zip(result["attempted_generated_answers"], result["chosen_answers"])
        ]
        for teacher, answer in zip(batch["teacher_answer"], measured_answers):
            predicted, target = parse_answer(answer), parse_answer(teacher)
            answer_counts["samples"] += 1
            answer_counts["format_valid"] += int(predicted.valid)
            p_obj = next((x for x in predicted.entities if x.role == "OBJ"), None)
            t_obj = next(x for x in target.entities if x.role == "OBJ")
            answer_counts["obj_correct"] += int(p_obj is not None and normalize_phrase(p_obj.phrase) == normalize_phrase(t_obj.phrase))
            p_anc = Counter(normalize_phrase(x.phrase) for x in predicted.entities if x.role == "ANC")
            t_anc = Counter(normalize_phrase(x.phrase) for x in target.entities if x.role == "ANC")
            answer_counts["anc_tp"] += sum((p_anc & t_anc).values()); answer_counts["anc_pred"] += sum(p_anc.values()); answer_counts["anc_true"] += sum(t_anc.values())
        if len(examples) < examples_limit:
            for index in range(min(len(batch["question"]), examples_limit-len(examples))):
                examples.append({"sample_id": batch["sample_id"][index], "question": batch["question"][index], "teacher_answer": batch["teacher_answer"][index],
                                 "evaluation_index": int(batch["evaluation_index"][index]),
                                 "conditioned_answer": result["chosen_answers"][index], "answer_source": result["answer_sources"][index],
                                 "attempted_generated_answer": result["attempted_generated_answers"][index],
                                 "selected_query": int(result["selected_query"][index]), "matched_query": int(result["matched_query"][index])})
    return {
        "metrics": book.state_dict(),
        "answer_counts": dict(answer_counts),
        "fallback": dict(fallback),
        "examples": examples,
    }


def merge_mode_payloads(payloads, examples_limit):
    book, answer_counts, fallback, examples = MetricBook(), Counter(), Counter(), []
    for payload in payloads:
        book.merge(payload["metrics"])
        answer_counts.update(payload["answer_counts"])
        fallback.update(payload["fallback"])
        examples.extend(dict(row) for row in payload["examples"])
    examples.sort(key=lambda row: row["evaluation_index"])
    examples = examples[:examples_limit]
    for row in examples:
        row.pop("evaluation_index", None)
    summary = book.summary(); samples = max(answer_counts["samples"], 1)
    precision = answer_counts["anc_tp"]/max(answer_counts["anc_pred"],1); recall = answer_counts["anc_tp"]/max(answer_counts["anc_true"],1)
    summary.update({"answer/format_valid_rate": answer_counts["format_valid"]/samples, "answer/obj_phrase_accuracy": answer_counts["obj_correct"]/samples,
                    "answer/anc_precision": precision, "answer/anc_recall": recall, "answer/anc_f1": 2*precision*recall/max(precision+recall,1e-12),
                    "answer/fallback_rate": sum(value for key,value in fallback.items() if key not in {"generated","teacher"})/samples,
                    "answer/fallback_reasons": dict(fallback), "samples": answer_counts["samples"]})
    return summary, examples


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path); parser.add_argument("checkpoint", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=["teacher","generated","generated_no_fallback"], default=["teacher","generated","generated_no_fallback"])
    parser.add_argument("--max-samples", type=int); parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--eval-cache-root",
        type=Path,
        help="Dedicated normal/superpoint cache prepared for cross-dataset validation",
    )
    args = parser.parse_args(); config = load_segmentation_config(args.config)
    rank, world_size, local_rank, distributed = distributed_context()
    # Object summaries are CPU data; independent shards finish at different times.
    control = dist.group.WORLD if distributed else None
    output = ensure_within(args.output_dir, config["project"]["artifact_root"], must_exist=False)
    checkpoint_path = ensure_within(args.checkpoint, config["project"]["artifact_root"], must_exist=True)
    if args.val_annotations is not None:
        val_annotations = ensure_within(
            args.val_annotations, config["project"]["artifact_root"], must_exist=True
        )
        config["data"]["val_annotations"] = str(val_annotations)
    if args.dataset_name is not None:
        config["data"]["dataset_name"] = args.dataset_name
    if args.eval_cache_root is not None:
        cache_root = ensure_within(
            args.eval_cache_root, config["project"]["artifact_root"], must_exist=True
        )
        config["data"]["normal"]["cache_dir"] = str(cache_root / "normals")
        config["data"]["superpoint"]["cache_dir"] = str(cache_root / "superpoints")
    dataset = build_segmentation_dataset(config, "val")
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    partitions = source_indices(dataset)
    source_counts = {source:len(indices) for source,indices in partitions.items()}
    dataset = IndexedDataset(dataset)
    source_loaders = {}
    for source,indices in partitions.items():
        subset = Subset(dataset, indices)
        sampler = ExactDistributedEvalSampler(subset, rank, world_size)
        source_loaders[source] = DataLoader(subset, batch_size=1, sampler=sampler,
            num_workers=int(config["data"]["num_workers"]), collate_fn=collate_samples)
    if rank == 0:
        print(f"Validation samples by dataset: {source_counts}",flush=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    run_token=[uuid.uuid4().hex if rank==0 else None]
    if distributed:
        dist.broadcast_object_list(run_token,src=0,group=control)
        dist.barrier(group=control)
        dist.destroy_process_group()
    shard_dir=output/".rank_results"/run_token[0]
    model = ObjAncSegmentationModel(config).to(device); checkpoint = load_checkpoint(model, checkpoint_path); model.eval()
    use_amp = bool(config["training"].get("amp", True))
    if len(set(args.modes))!=len(args.modes):
        raise ValueError("Evaluation modes must not repeat")
    local_payload={}
    for mode in args.modes:
        local_payload[mode]={}
        for source,loader in source_loaders.items():
            print(f"[eval rank={rank}] source={source}, mode={mode}, local_batches={len(loader)}",flush=True)
            if mode=="generated_no_fallback":
                local_payload[mode][source]=evaluate_no_fallback(model,loader,float(config["validation"]["mask_threshold"]),args.examples,use_amp=use_amp)
            else:
                local_payload[mode][source]=evaluate_mode(model,loader,mode,float(config["validation"]["mask_threshold"]),args.examples,use_amp=use_amp)
    print(f"[eval rank={rank}] {args.modes} complete; collecting atomic rank files",flush=True)
    if distributed:
        path=publish_shard(shard_dir,run_token[0],rank,world_size,local_payload)
        print(f"Rank {rank} published {path}",flush=True)
        if rank!=0:return
        gathered=wait_for_shards(shard_dir,run_token[0],world_size,timeout=float(os.environ.get("EVAL_RANK_WAIT_SECONDS","86400")),report=lambda message:print(message,flush=True))
    else:
        gathered = [local_payload]
    if rank == 0:
        mode_summaries,mode_examples,by_dataset=merge_source_payloads(
            gathered,args.modes,source_counts,merge_mode_payloads,merge_no_fallback_payloads,args.examples)
        summary={"objective":OBJECTIVE,"checkpoint":str(checkpoint_path),"step":checkpoint["step"],
            "completed_epochs":checkpoint["completed_epochs"],"distributed_world_size":world_size,
            "validation_samples":len(dataset),"validation_dataset":config["data"].get("dataset_name"),
            "validation_annotations":config["data"]["val_annotations"],"evaluation_modes":args.modes,
            "validation_samples_by_dataset":source_counts}
        metadata=dict(summary)
        for mode,metrics in mode_summaries.items():
            key={"teacher":"teacher_answer","generated":"generated_answer","generated_no_fallback":"generated_no_fallback"}[mode]
            summary[key]=metrics
        if "teacher" in mode_summaries and "generated" in mode_summaries:
            summary["selected_generated_gap"]=mode_summaries["teacher"].get("obj/selected_iou",0)-mode_summaries["generated"].get("obj/selected_iou",0)
        summary["by_dataset"]=by_dataset
        (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        for source,metrics in by_dataset.items():
            source_summary={**metadata,**metrics,"validation_dataset":source,
                "parent_validation_dataset":config["data"].get("dataset_name"),
                "validation_samples_by_dataset":{source:source_counts[source]}}
            (output/f"summary_{source}.json").write_text(json.dumps(source_summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        with (output/"examples.jsonl").open("w", encoding="utf-8") as handle:
            for mode, rows in mode_examples.items():
                for row in rows: handle.write(json.dumps({"mode": mode, **row}, ensure_ascii=False)+"\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
