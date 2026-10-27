"""Teacher-vs-generated Answer segmentation evaluation."""
from __future__ import annotations

import argparse
import json
import os
import hashlib
import random
import uuid
from datetime import timedelta
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within, load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.data import collate_samples
from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import build_segmentation_dataset
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from spatiallm_compressed_visual_objanc_full_attention.model.curriculum import normalize_phrase
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import OBJECTIVE, ObjAncSegmentationModel
from ground3d_migration.diagnostics import tie_aware_rank
from ground3d_migration.runtime import memory_report
from ground3d_migration.eval_shards import publish_shard, wait_for_shards
from runtime_helpers import checkpoint_manifest, load_trainable_state, validate_checkpoint


def load_checkpoint(model, path: Path, manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, OBJECTIVE, model.architecture_signature(), manifest)
    from train_segmentation import restore_tokenizer_checkpoint
    restore_tokenizer_checkpoint(model,checkpoint)
    load_trainable_state(model, checkpoint.get('trainable_state'))
    from anchor_runtime import restore_buffers
    if not checkpoint.get('buffer_states'):raise ValueError('Missing checkpoint buffers')
    restore_buffers(model,checkpoint['buffer_states'][0])
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
        if torch.cuda.is_available():torch.cuda.set_device(local_rank)
        dist.init_process_group("gloo",timeout=timedelta(hours=2))
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
    sample_indices=[]; sample_ids=[];subset_books={k:MetricBook() for k in ['unique','multiple']}
    from anchor_eval import load_subset_lookup
    subset_lookup=load_subset_lookup(model.config)
    trace_path=Path(os.environ['GROUND3D_UNCONSTRAINED_OUTPUT'])/f'predictions.{mode}.rank_{os.environ.get("RANK","0")}.jsonl'
    trace=trace_path.open('x',encoding='utf-8')
    visual_max={"llm_tokens":0,"superpoints":0}
    raw_book, raw_counts = MetricBook(), Counter()
    normal_book,zero_book=MetricBook(),MetricBook()
    answer_counts, fallback = Counter(), Counter()
    for batch in loader:
        sample_indices.extend(int(i) for i in batch['evaluation_index']);sample_ids.extend(str(i) for i in batch['sample_id'])
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
        for b in range(len(batch['sample_id'])):
            subset=subset_lookup[(str(batch['scene_id'][b]),str(batch['object_id'][b]))]
            failed=mode=='generated' and result['synthetic_obj'][b]
            selected_iou=0. if failed else mask_iou(result['raw_logits'][b],batch['gt_point_masks'][b][0],threshold)
            sub=subset_books[subset]
            sub._add_value('obj/selected_iou',selected_iou)
            for level in (.25,.5):sub._add_value(f'obj/selected_acc@{level:.2f}',float(selected_iou>=level))
            trace.write(json.dumps(dict(sample_id=str(batch['sample_id'][b]),evaluation_index=int(batch['evaluation_index'][b]),subset=subset,selected_query=int(result['selected_query'][b]),strict_raw_iou=selected_iou,format_failure=bool(failed)))+'\n')
            trace.flush()

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
    trace.close()
    return {
        "subset_metrics":{k:v.state_dict() for k,v in subset_books.items()},
        "metrics": book.state_dict(), "visual_max":visual_max,
        "sample_indices":sample_indices,"sample_ids":sample_ids,
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
        examples.extend(payload["examples"])
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
    by_subset={}
    for subset in ['unique','multiple']:
        subset_book=MetricBook()
        for payload in payloads:subset_book.merge(payload.get('subset_metrics',{}).get(subset,{'sums':{},'counts':{}}))
        by_subset[subset]={**subset_book.summary(),'samples':subset_book.counts['obj/selected_iou']}
    summary['by_subset']=by_subset
    return summary, examples


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path); parser.add_argument("checkpoint", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument('--anchor-diagnostic',choices=['off','disable','shuffle_geometry'],default='off')
    parser.add_argument("--max-samples", type=int); parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--rank-wait-timeout", type=float, default=86400,
                        help="Rank 0 filesystem result wait limit in seconds (default: 24 hours)")
    parser.add_argument("--val-annotations", type=Path)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--eval-cache-root",
        type=Path,
        help="Dedicated normal/superpoint cache prepared for cross-dataset validation",
    )
    args = parser.parse_args()
    if any(v is not None for v in (args.max_samples,args.val_annotations,args.dataset_name,args.eval_cache_root)):
        raise ValueError('Formal evaluation requires the full ScanRefer dataset and fixed source')
    if args.rank_wait_timeout <= 0: parser.error("--rank-wait-timeout must be positive")
    config = load_segmentation_config(args.config)
    rank, world_size, local_rank, distributed = distributed_context()
    control=dist.group.WORLD if distributed else None
    manifest=[checkpoint_manifest(config) if rank==0 else None]
    if distributed:dist.broadcast_object_list(manifest,src=0,group=control)
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
    if len(dataset)!=9483:raise ValueError("Formal evaluation requires full ScanRefer 9483")
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    dataset = IndexedDataset(dataset)
    sampler = ExactDistributedEvalSampler(dataset, rank, world_size)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=int(config["data"]["num_workers"]), collate_fn=collate_samples)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    run_token = [uuid.uuid4().hex if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(run_token, src=0, group=control)
    shard_dir = output / ".rank_results" / run_token[0]
    if distributed:
        shard_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier(group=control)
        # No DDP is used in evaluation. Tear down communication before unequal
        # model load/evaluation durations; results are atomically persisted below.
        dist.destroy_process_group()
    model = ObjAncSegmentationModel(config).to(device); checkpoint = load_checkpoint(model, checkpoint_path,manifest[0]); model.eval(); model.decoder.diagnostic=args.anchor_diagnostic
    use_amp = bool(config["training"].get("amp", True))
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    local_payload = {
        "teacher": evaluate_mode(model, loader, "teacher", float(config["validation"]["mask_threshold"]), args.examples, use_amp=use_amp),
        "generated": evaluate_mode(model, loader, "generated", float(config["validation"]["mask_threshold"]), args.examples, use_amp=use_amp),
    }
    local_payload["gpu_memory"]=memory_report()
    if distributed:
        saved = publish_shard(shard_dir, run_token[0], rank, world_size, local_payload)
        print(f"Rank {rank} evaluation complete; saved {saved}", flush=True)
        if rank != 0:
            return  # finished workers never enter a collective or wait for rank 0
        gathered = wait_for_shards(shard_dir, run_token[0], world_size,
                                   timeout=args.rank_wait_timeout,
                                   report=lambda message: print(message, flush=True))
    else:
        gathered = [local_payload]
    for shard_rank, payload in enumerate(gathered):
        expected = len(ExactDistributedEvalSampler(dataset, shard_rank, world_size))
        for mode in ("teacher", "generated"):
            actual = payload[mode]["answer_counts"].get("samples", 0)
            if actual != expected:
                raise ValueError(f"Incomplete {mode} results for rank {shard_rank}: {actual}/{expected}")
    for mode in ('teacher','generated'):
        ids=[i for item in gathered for i in item[mode]['sample_indices']]
        keys=[i for item in gathered for i in item[mode]['sample_ids']]
        if sorted(ids)!=list(range(len(dataset))) or len(set(keys))!=len(dataset):raise ValueError('Missing or duplicate evaluation IDs')
    if rank == 0:
        teacher, teacher_examples = merge_mode_payloads([item["teacher"] for item in gathered], args.examples)
        generated, generated_examples = merge_mode_payloads([item["generated"] for item in gathered], args.examples)
        summary = {"anchor_diagnostic":args.anchor_diagnostic,"run":config["run"],"objective": OBJECTIVE, "checkpoint": str(checkpoint_path), "step": checkpoint["step"],
                   'completed_epochs':checkpoint['completed_epochs'],
                   "distributed_world_size": world_size, "validation_samples": len(dataset),
                   "aggregation": {"method": "atomic_rank_files" if distributed else "single_process",
                                   "run_id": run_token[0], "rank_wait_timeout_seconds": args.rank_wait_timeout},
                   "validation_dataset": str(config["data"].get("dataset_name", "unknown")),
                   "validation_annotations": str(config["data"]["val_annotations"]),
                   "teacher_answer": teacher, "generated_answer": generated,
                   "gpu_memory":{"phase":"evaluation","aggregation":"max_across_ranks",**{key:max((p['gpu_memory'][key] or 0) for p in gathered) or None for key in ['peak_allocated_bytes','peak_reserved_bytes']}},
                   "ranking_definition":"tie-aware: earliest score rank among queries tied for maximum GT IoU; legacy argmax metrics retained",
                   "oracle_generated_gap": teacher.get("obj/oracle_iou",0)-generated.get("obj/oracle_iou",0),
                   "selected_generated_gap": teacher.get("obj/selected_iou",0)-generated.get("obj/selected_iou",0)}
        (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        with (output/"examples.jsonl").open("w", encoding="utf-8") as handle:
            for mode, rows in (("teacher", teacher_examples), ("generated", generated_examples)):
                for row in rows: handle.write(json.dumps({"mode": mode, **row}, ensure_ascii=False)+"\n")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
