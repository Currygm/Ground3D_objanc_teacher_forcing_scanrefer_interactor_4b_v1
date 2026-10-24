"""No-Interactor Reason3D evaluation with raw greedy decoding and zero-OBJ failures."""
from __future__ import annotations

import argparse
import json
import os
import uuid
import hashlib
import random
import numpy as np
from datetime import timedelta
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within
from reason3d_runtime import restore_task_state, load_segmentation_config, build_segmentation_dataset, OBJECTIVE
from reason3d_unconstrained_model import ObjAncSegmentationModel
from reason3d_no_fallback import evaluate_no_fallback, merge_no_fallback_payloads
from reason3d_eval_subsets import source_indices, merge_source_payloads
from ground3d_migration.diagnostics import tie_aware_rank
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

    def add(self, role, selected, matched, oracle, selected_q, matched_q, best_q, rank, topk, *, segmentation=None):
        for prefix in ("all", role.lower()):
            self._add_value(f"{prefix}/selected_iou", selected)
            self._add_value(f"{prefix}/matched_iou", matched)
            self._add_value(f"{prefix}/oracle_iou", oracle)
            self._add_value(f"{prefix}/best_query_rank", rank)
            self._add_value(f"{prefix}/qmatch", float(selected_q == matched_q))
            self._add_value(f"{prefix}/best_iou_query_recall", float(selected_q == best_q))
            for k, value in topk.items():
                self._add_value(f"{prefix}/top{k}_oracle_iou", value)
                for threshold in (.25,.5):self._add_value(f'{prefix}/recall{int(threshold*100)}_top{k}',value>=threshold)
            for key,value in (segmentation or {}).items():self._add_value(f'{prefix}/{key}',value)
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


def measure_result(book, batch, result, threshold):
    decoder=result['decoder_output']
    for b,roles in enumerate(result['entity_roles']):
        mapping=batch['point_to_superpoint'][b].to(decoder['pred_masks'].device,torch.long)
        truths=batch['gt_point_masks'][b][result['target_rows'][b]]
        masks=decoder['pred_masks'][b,:,:batch['gt_superpoint_masks'][b].shape[1]][:,mapping]
        scores=result['query_entity_scores'][b,:,:len(roles)]
        source,target=result['assignments'][b]
        matched_by_target={int(t):int(q) for q,t in zip(source,target)}
        for e,(role,truth) in enumerate(zip(roles,truths)):
            ious=[mask_iou(m,truth,threshold) for m in masks]
            order=torch.argsort(scores[:,e],descending=True,stable=True).tolist()
            selected,best,matched=order[0],max(range(len(ious)),key=ious.__getitem__),matched_by_target[e]
            prediction=masks[selected].sigmoid()>=threshold; gt=truth.to(prediction.device,torch.bool)
            intersection=(prediction & gt).sum().float()
            segmentation={'precision':float(intersection/prediction.sum().clamp_min(1)),
                'recall':float(intersection/gt.sum().clamp_min(1)),
                'dice':float(2*intersection/(prediction.sum()+gt.sum()).clamp_min(1))}
            for key,value in tie_aware_rank(scores[:,e],ious).items():
                for prefix in ['all',role.lower()]:book._add_value(prefix+'/tie_aware_'+key,value)
            topk={k:max(ious[q] for q in order[:k]) for k in (1,3,5,10,50)}
            book.add(role,ious[selected],ious[matched],ious[best],selected,matched,best,order.index(best)+1,topk,segmentation=segmentation)

def add_failed_raw_obj(book):
    # Undefined Query rank is represented by a sentinel after all 50 candidates.
    book.add('OBJ',0.,0.,0.,-1,-2,-3,51,{k:0. for k in (1,3,5,10,50)},
        segmentation={'precision':0.,'recall':0.,'dice':0.})
    for prefix in ('obj','all'):
        for key in ('best_query_top1_rate','best_query_top3_rate','best_query_top5_rate','best_query_top10_rate'):
            book._add_value(prefix+'/tie_aware_'+key,0.)
        book._add_value(prefix+'/tie_aware_best_query_rank',51.)
        book._add_value(prefix+'/tie_aware_best_query_reciprocal_rank',0.)

def evaluate_mode(model, loader, mode, threshold, examples_limit, *, use_amp):
    book, examples = MetricBook(), []
    visual_max={"llm_tokens":0,"superpoints":0}
    raw_book, raw_counts = MetricBook(), Counter()
    normal_book,zero_book=MetricBook(),MetricBook()
    answer_counts, fallback = Counter(), Counter()
    for batch_index,batch in enumerate(loader,1):
        model.answer_lm._raw_sample={"sample_id":str(batch["sample_id"][0]),"evaluation_index":int(batch["evaluation_index"][0]),"mode":mode}
        if batch_index==1 or batch_index%25==0:print(f"[rank {os.environ.get('RANK','0')}] {mode}: {batch_index}/{len(loader)}",flush=True)
        seed=int(model.config['training']['seed'])+int.from_bytes(hashlib.sha256(str(batch['sample_id'][0]).encode()).digest()[:4],'little')
        random.seed(seed); np.random.seed(seed%2**32); torch.manual_seed(seed)
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
        for key,metric in [('llm_tokens','llm_visual_tokens'),('superpoints','full_superpoints')]:
            count=int(result['metrics'][metric]);book._add_value('visual/'+key+'_mean',count);visual_max[key]=max(visual_max[key],count)
        for key in ('causal_forward','generation_prefill','generation_incremental','visual_encode_calls','interactor_calls','interactor_text_tokens'):
            book._add_value('calls/'+key,float(result['metrics'].get(key,0)))
        for key in ('visual_seconds','interactor_seconds'):
            book._add_value('timing/'+key,float(getattr(model.answer_lm,key,0)))
        measure_result(book,batch,result,threshold)
        if mode=='generated':
            if len(batch['question'])!=1:raise ValueError('Evaluation requires batch size 1')
            raw_counts['attempted']+=1
            if result['synthetic_obj'][0]:
                raw_counts['format_unevaluable']+=1
                add_failed_raw_obj(raw_book)
                measure_result(zero_book,batch,result,threshold)
            else:
                raw_counts['evaluable']+=1
                measure_result(raw_book,batch,result,threshold)
                measure_result(normal_book,batch,result,threshold)
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
        "metrics": book.state_dict(), "visual_max":visual_max,
        "answer_counts": dict(answer_counts),
        "fallback": dict(fallback),
        "examples": examples,
        'raw_metrics':raw_book.state_dict(),
        'raw_counts':dict(raw_counts),
        'normal_metrics':normal_book.state_dict(),'zero_obj_metrics':zero_book.state_dict(),
    }


def merge_mode_payloads(payloads, examples_limit):
    book, answer_counts, fallback, examples = MetricBook(), Counter(), Counter(), []
    raw_book,raw_counts=MetricBook(),Counter()
    normal_book,zero_book=MetricBook(),MetricBook()
    for payload in payloads:
        book.merge(payload["metrics"])
        answer_counts.update(payload["answer_counts"])
        fallback.update(payload["fallback"])
        examples.extend(dict(row) for row in payload["examples"])
        raw_book.merge(payload.get('raw_metrics',{'sums':{},'counts':{}}))
        raw_counts.update(payload.get('raw_counts',{}))
        normal_book.merge(payload.get('normal_metrics',{'sums':{},'counts':{}}))
        zero_book.merge(payload.get('zero_obj_metrics',{'sums':{},'counts':{}}))
    examples.sort(key=lambda row: row["evaluation_index"])
    examples = examples[:examples_limit]
    for row in examples:
        row.pop("evaluation_index", None)
    summary = book.summary(); samples = max(answer_counts["samples"], 1)
    summary['visual_usage']={key+'_mean':summary.get('visual/'+key+'_mean') for key in ['llm_tokens','superpoints']}
    summary['visual_usage'].update({key+'_max':max(p.get('visual_max',{}).get(key,0) for p in payloads) for key in ['llm_tokens','superpoints']})
    precision = answer_counts["anc_tp"]/max(answer_counts["anc_pred"],1); recall = answer_counts["anc_tp"]/max(answer_counts["anc_true"],1)
    summary.update({"answer/format_valid_rate": answer_counts["format_valid"]/samples, "answer/obj_phrase_accuracy": answer_counts["obj_correct"]/samples,
                    "answer/anc_precision": precision, "answer/anc_recall": recall, "answer/anc_f1": 2*precision*recall/max(precision+recall,1e-12),
                    "answer/fallback_rate": sum(value for key,value in fallback.items() if key not in {"generated","teacher"})/samples,
                    "answer/fallback_reasons": dict(fallback), "samples": answer_counts["samples"]})
    if raw_counts['attempted']:
        summary['raw_generated_no_fallback']={**raw_book.summary(),'samples':raw_counts['attempted'],'evaluable':raw_counts['evaluable'],
            'attempted':raw_counts['attempted'],'coverage':raw_counts['evaluable']/raw_counts['attempted'],
            'format_unevaluable':raw_counts['format_unevaluable'],'teacher_fallbacks':0,
            'failed_query_rank_sentinel':51,'denominators':dict(raw_book.counts)}
        zero_count=raw_counts['format_unevaluable']
        summary['teacher_fallbacks']=0
        summary['zero_obj_rate']=zero_count/samples
        summary['generated_with_zero_obj']={**book.summary(),'samples':answer_counts['samples'],'teacher_fallbacks':0,'denominators':dict(book.counts)}
        summary['normal_generated_subset']={**normal_book.summary(),'samples':raw_counts['evaluable'],'denominators':dict(normal_book.counts)}
        summary['zero_obj_subset']={**zero_book.summary(),'samples':zero_count,'denominators':dict(zero_book.counts)}
        summary['raw_coverage']=raw_counts['evaluable']/raw_counts['attempted']
    return summary, examples


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path); parser.add_argument("checkpoint", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=["teacher","generated"], default=["teacher","generated"])
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
    if rank == 0 and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh empty evaluation output directory")
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
    seed=int(config["training"]["seed"])+rank
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    model = ObjAncSegmentationModel(config).to(device); checkpoint = load_checkpoint(model, checkpoint_path); model.eval()
    from reason3d_unconstrained import install_unconstrained
    trace_path=output/f"raw_generations.rank_{rank:05d}.jsonl"
    if (output/"summary.json").exists():raise FileExistsError("Use a fresh evaluation output")
    trace=trace_path.open("x",encoding="utf-8")
    install_unconstrained(model,lambda row:(trace.write(json.dumps({**getattr(model.answer_lm,"_raw_sample",{}),**row},ensure_ascii=False)+"\n"),trace.flush()))
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
    trace.close()
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
        from reason3d_unconstrained import GENERATION_PROTOCOL
        summary["generation_protocol"]=dict(GENERATION_PROTOCOL)
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
