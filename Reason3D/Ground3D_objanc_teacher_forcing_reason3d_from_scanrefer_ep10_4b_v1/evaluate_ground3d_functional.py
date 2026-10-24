"""Evaluate native Reason3D epoch13 weights on Ground3D functional grounding."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import uuid

PROJECT=Path(__file__).resolve().parent

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--annotations",type=Path,default=PROJECT/"data/ground3d_functional_val/annotations.jsonl")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,default=PROJECT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml")
    p.add_argument("--modes",nargs="+",choices=["generated_no_fallback","teacher"],default=["generated_no_fallback","teacher"])
    p.add_argument("--sources",nargs="+",choices=["scannet","scannetpp"],default=["scannet","scannetpp"])
    p.add_argument("--levels",nargs="+",choices=["object_qa","part_qa"],default=["object_qa","part_qa"])
    p.add_argument("--max-samples",type=int)
    p.add_argument("--check-data",action="store_true",help="CPU only: verify manifest/eligibility; do not construct model or touch CUDA")
    return p.parse_args()

def main():
    args=arguments()
    from ground3d_functional import (SCHEMA,PROJECT,FunctionalDataset,aggregate,atomic_json,
        evaluate_sample,project_path,summarize,validate_frame)
    annotations=project_path(args.annotations)
    raw=annotations.read_bytes()
    manifest=json.loads(annotations.with_name("manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"]!=SCHEMA or manifest["split"]!="val":
        raise ValueError("Expected Ground3D functional validation manifest")
    if hashlib.sha256(raw).hexdigest()!=manifest["annotation_sha256"]:
        raise ValueError("Adapted annotation hash differs from preparation manifest")
    all_rows=[json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(all_rows)!=manifest["samples"]:raise ValueError("Manifest sample count differs")
    if len(set(args.modes))!=len(args.modes):raise ValueError("Modes must not repeat")
    rows=[r for r in all_rows if r["source"] in args.sources and r["level"] in args.levels]
    if args.max_samples is not None:
        if args.max_samples<1:raise ValueError("--max-samples must be positive")
        rows=rows[:args.max_samples]
    if not rows:raise ValueError("No evaluation samples")
    if len({r["sample_id"] for r in rows})!=len(rows):raise ValueError("Duplicate sample IDs")
    checkpoint_path=project_path(args.checkpoint)
    if not checkpoint_path.is_file():raise FileNotFoundError(checkpoint_path)
    if args.check_data:
        cache={}
        for i,row in enumerate(rows):
            validate_frame(row,manifest["data_root"],cache)
            if i%1000==0:print(f"CPU data check {i}/{len(rows)}",flush=True)
        print(json.dumps({"status":"passed","samples":len(rows),"frames_and_label_spaces":len(cache),
            "cuda_used":False,"checkpoint":str(checkpoint_path)},indent=2))
        return
    import torch
    import torch.distributed as dist
    from evaluate_reason3d import distributed_context,load_checkpoint
    from reason3d_runtime import load_segmentation_config,ObjAncSegmentationModel
    from ground3d_migration.eval_shards import publish_shard,wait_for_shards
    config=load_segmentation_config(args.config)
    rank,world,local,distributed=distributed_context()
    output=project_path(args.output)
    if rank==0:output.mkdir(parents=True,exist_ok=True)
    run=[uuid.uuid4().hex if rank==0 else None]
    if distributed:
        dist.broadcast_object_list(run,src=0);dist.barrier();dist.destroy_process_group()
    device=torch.device("cuda",local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type!="cuda":raise RuntimeError("GPU evaluation requires CUDA; use --check-data for CPU verification")
    torch.cuda.set_device(local)
    torch.manual_seed(42)
    model=ObjAncSegmentationModel(config)
    # Reuse the exact native objective, architecture, base-manifest, tokenizer and task-state checks.
    checkpoint=load_checkpoint(model,checkpoint_path)
    model=model.to(device);model.eval()
    dataset=FunctionalDataset(rows,data_root=manifest["data_root"],cache_dir=PROJECT/"cache/ground3d_functional_geometry")
    indices=list(range(rank,len(rows),world));local_rows=[]
    for mode in args.modes:
        for offset,index in enumerate(indices):
            if offset%50==0:print(f"[rank {rank}] {mode} {offset}/{len(indices)} sample={rows[index]['sample_id']}",flush=True)
            try:
                with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=bool(config["training"].get("amp",True))):
                    prediction=evaluate_sample(model,dataset,index,mode,threshold=float(config["validation"]["mask_threshold"]))
            except torch.OutOfMemoryError:
                print(f"CUDA OOM at {rows[index]['sample_id']}; infrastructure failures are not counted as model zero scores",flush=True)
                raise
            local_rows.append(prediction)
    payload={"rows":local_rows}
    shards=output/".rank_results"/run[0]
    if distributed:
        publish_shard(shards,run[0],rank,world,payload)
        if rank!=0:return
        gathered=wait_for_shards(shards,run[0],world,timeout=float(os.environ.get("EVAL_RANK_WAIT_SECONDS","86400")),report=lambda s:print(s,flush=True))
    else:gathered=[payload]
    results=sorted([r for p in gathered for r in p["rows"]],key=lambda r:(r["mode"],r["evaluation_index"]))
    summary=summarize(results,expected_indices=range(len(rows)),modes=args.modes)
    summary.update(checkpoint=str(checkpoint_path),completed_epochs=checkpoint["completed_epochs"],step=checkpoint["step"],
        schema=SCHEMA,validation_samples=len(rows),full_adapted_validation_samples=manifest["samples"],
        annotations=str(annotations),annotation_sha256=manifest["annotation_sha256"],
        protocol=manifest["protocol"],preparation_exclusions=manifest["exclusions"],
        task_scope=sorted(set(r["task"] for r in rows)),world_size=world,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),metrics_scale="0 to 1",
        source_split="official Ground3D val; per-frame point masks; not a complete Ground3D QA benchmark",
        input_modality="XYZ+RGB point cloud only; no RGB image or text answer supplied in generated mode")
    atomic_json(output/"summary.json",summary)
    for source,metrics in summary["by_dataset"].items():
        atomic_json(output/f"summary_{source}.json",{"checkpoint":str(checkpoint_path),"metrics":metrics,
            "by_level":summary["by_dataset_and_level"][source]})
    for level,metrics in summary["by_level"].items():
        atomic_json(output/f"summary_{level}.json",{"checkpoint":str(checkpoint_path),"metrics":metrics})
    temp=output/("predictions."+run[0]+".tmp")
    with temp.open("w",encoding="utf-8") as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    os.replace(temp,output/"predictions.jsonl")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("by_dataset_and_level","preparation_exclusions")},indent=2),flush=True)

if __name__=="__main__":main()
