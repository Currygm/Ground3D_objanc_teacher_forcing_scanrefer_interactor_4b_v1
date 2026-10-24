"""Evaluate released official Ground3D 3D weights on our exact 8329 functional val QAs."""
from pathlib import Path
import argparse
import json
import os
import uuid
import hashlib

PROJECT=Path(__file__).resolve().parent

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path("/home/yibo/ground3d-lmm"))
    p.add_argument("--checkpoint",type=Path,default=Path("/home/yibo/ground3d-lmm/Ground3D-LMM-4B-3D/pytorch_model.pth"))
    p.add_argument("--qwen",type=Path,default=Path("/home/yibo/ground3d-lmm/Qwen3-VL-4B-Instruct"))
    p.add_argument("--annotations",type=Path,default=PROJECT/"data/ground3d_functional_val/annotations.jsonl")
    p.add_argument("--output",type=Path,default=PROJECT/"evaluations/ground3d_official_3d_functional")
    p.add_argument("--max-samples",type=int,help="Partial smoke only; never reported as full validation")
    p.add_argument("--check-data",action="store_true",help="CPU only, verify all raw point labels and manifest")
    p.add_argument("--check-model",action="store_true",help="CPU only, build official model and strictly validate all task weights")
    return p.parse_args()

def save_results(output,results,rows,metadata,run_id):
    from ground3d_functional import summarize,atomic_json
    summary=summarize(results,expected_indices=range(len(rows)),modes=["generated_no_fallback"])
    summary.update(metadata)
    atomic_json(output/"summary.json",summary)
    for source,metrics in summary["by_dataset"].items():
        atomic_json(output/f"summary_{source}.json",dict(metrics=metrics,by_level=summary["by_dataset_and_level"][source]))
    for level,metrics in summary["by_level"].items():
        atomic_json(output/f"summary_{level}.json",dict(metrics=metrics))
    temp=output/("predictions."+run_id+".tmp")
    with temp.open("x",encoding="utf-8") as f:
        for r in results:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    os.replace(temp,output/"predictions.jsonl")
    return summary

def main():
    args=arguments()
    from ground3d_official_eval import (read_annotations,build_official,OfficialDataset,
        OfficialInference,evaluate_both,file_sha256,OFFICIAL_SHA256)
    from ground3d_functional import project_path,validate_frame,atomic_json
    output=project_path(args.output)
    if not args.check_data and not args.check_model and (output/"summary.json").exists():
        raise FileExistsError("Completed output already exists; select a new --output directory")
    rows,manifest=read_annotations(args.annotations,max_samples=args.max_samples)
    if args.check_data:
        cache={}
        for i,row in enumerate(rows):
            validate_frame(row,manifest["data_root"],cache)
            if i%1000==0:print(f"CPU data check {i}/{len(rows)}",flush=True)
        print(json.dumps(dict(status="passed",samples=len(rows),frames_and_label_spaces=len(cache),cuda_used=False)),flush=True)
        return
    if args.check_model:
        output.mkdir(parents=True,exist_ok=True)
        model,report=build_official(args.repo,args.checkpoint,args.qwen,output/"cpu_preflight",device="cpu")
        import torch
        report.update(cuda_initialized=torch.cuda.is_initialized(),device=str(next(model.parameters()).device))
        if report["cuda_initialized"]:raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
        atomic_json(output/"cpu_loading_report.json",report)
        print(json.dumps(dict(status="passed",loaded_tensors=report["loaded_tensors"],
            frozen_base_missing=len(report["allowed_base_missing"]),cuda_initialized=False)),flush=True)
        return
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from ground3d_migration.eval_shards import publish_shard,wait_for_shards
    rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0))
    if not torch.cuda.is_available():raise RuntimeError("GPU evaluation requires CUDA; CPU checks use --check-data or --check-model")
    run=[uuid.uuid4().hex if rank==0 else None]
    if world>1:dist.init_process_group("gloo",timeout=timedelta(hours=2))
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        digest=file_sha256(args.checkpoint)
        if digest!=OFFICIAL_SHA256:raise ValueError("Official checkpoint hash differs")
    if world>1:
        dist.broadcast_object_list(run,src=0)
        dist.barrier()
        dist.destroy_process_group()
    device=torch.device("cuda",local);torch.cuda.set_device(local)
    torch.manual_seed(42)
    model,loading=build_official(args.repo,args.checkpoint,args.qwen,
        output/".native_predictions"/run[0]/f"rank_{rank}",device=device,verify_hash=False)
    atomic_json(output/f"loading_rank_{rank}.json",loading)
    dataset=OfficialDataset(rows,manifest["data_root"])
    infer=OfficialInference(model,device,threshold=.5)
    results=[];official04=[]
    indices=list(range(rank,len(rows),world))
    for offset,index in enumerate(indices):
        if offset%25==0:print(f"[rank {rank}] {offset}/{len(indices)} {rows[index]['sample_id']}",flush=True)
        # OOM, corrupted data or nonfinite logits abort the run; no silent model-zero substitution.
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            matched,native=evaluate_both(dataset,index,infer)
        results.append(matched);official04.append(native)
    payload=dict(matched=results,official04=official04)
    shards=output/".rank_results"/run[0]
    if world>1:
        publish_shard(shards,run[0],rank,world,payload)
        if rank:return
        gathered=wait_for_shards(shards,run[0],world,timeout=float(os.environ.get("EVAL_RANK_WAIT_SECONDS",86400)))
    else:gathered=[payload]
    metadata=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=OFFICIAL_SHA256,
        model="official Ground3D-LMM-4B-3D; no ScanRefer/Reason3D task state",
        validation_samples=len(rows),full_adapted_validation_samples=8329,
        annotations=str(args.annotations.resolve()),annotation_sha256=manifest["annotation_sha256"],
        partial_evaluation=len(rows)!=8329,world_size=world,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),metrics_scale="0 to 1",
        input_modality="XYZ+RGB point cloud only; no RGB image",
        query_policy="Native first generated phrase/query; no GT matching; missing phrase scores zero",
        gt_policy="Original raw point object_label / part_label; exact same QAs and denominator as our model",
        generation="Native official greedy generation, max_new_tokens=1024, repetition_penalty=1.2; causal replay",
        scope="Single-turn functional_object_grounding and functional_part_grounding only",
        official_source_sha256={str(p.relative_to(args.repo)):file_sha256(p) for p in
            [args.repo/"uniseg3d/query_decoder.py",args.repo/"uniseg3d/uniseg3d.py",
             args.repo/"uniseg3d/prompt_template.py",args.repo/"uniseg3d/point_qwen3_vl.py"]})
    main_rows=sorted([r for p in gathered for r in p["matched"]],key=lambda r:r["evaluation_index"])
    native_rows=sorted([r for p in gathered for r in p["official04"]],key=lambda r:r["evaluation_index"])
    native_dir=output/"threshold_official_04";native_dir.mkdir(parents=True,exist_ok=True)
    save_results(native_dir,native_rows,rows,dict(metadata,mask_threshold=.4,threshold_operator=">"),run[0])
    summary=save_results(output,main_rows,rows,dict(metadata,mask_threshold=.5,threshold_operator=">="),run[0])
    print(json.dumps({k:summary[k] for k in ["metrics","by_dataset_and_level","validation_samples"]},indent=2),flush=True)

if __name__=="__main__":main()
