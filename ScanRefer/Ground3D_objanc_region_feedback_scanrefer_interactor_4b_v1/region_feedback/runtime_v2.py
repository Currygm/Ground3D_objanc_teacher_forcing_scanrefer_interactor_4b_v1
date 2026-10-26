"""Cache-only Stage A train/evaluation: fixed first-pass outputs, one trainable LM."""
import argparse
import contextlib
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler
from ground3d_migration.runtime import capture_rng,memory_report
from ground3d_migration.eval_shards import publish_shard,wait_for_shards
from runtime_helpers import AsyncJsonLog
from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within
from .config import load_feedback_config
from .factory import provenance,cache_identity,build_model
from .cache import CachedDataset
from .checkpoint import save_checkpoint,load_checkpoint,import_stage_a
from .metrics import candidate_metrics,zero_metrics,summarize

def collate_cached(rows):return rows

def cached_loader(dataset,config,split,rank,world,epoch=0,max_samples=None):
    if max_samples is not None:
        if max_samples<1:raise ValueError("max_samples must be positive")
        dataset=Subset(dataset,range(min(max_samples,len(dataset))))
    cfg=config["training"]
    if split=="train":
        sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,
                                   seed=int(cfg["seed"]),drop_last=False)
        sampler.set_epoch(epoch)
    else:
        dataset=Subset(dataset,range(rank,len(dataset),world));sampler=None
    generator=torch.Generator().manual_seed(int(cfg["seed"])+rank+1000003*epoch)
    loader=DataLoader(dataset,batch_size=int(cfg["batch_size"]),sampler=sampler,
        num_workers=int(cfg["num_workers"]),persistent_workers=False,
        collate_fn=collate_cached,generator=generator)
    return dataset,loader

def load_cache(config,identity,split,*,max_samples=None):
    root=Path(config["cache_dir"])/split
    count=36569 if split=="train" else 9483
    # A partial val cache is usable only for explicitly limited acceptance.
    if max_samples is not None and split=="val":
        manifest=root/"manifest.json"
        if manifest.is_file():
            actual=json.loads(manifest.read_text()).get("count")
            if actual!=count:
                if actual!=max_samples:raise ValueError("Partial cache requires matching --max-samples")
                count=actual
    return CachedDataset(root,cache_identity(config,identity),expected_count=count,split=split,
                         full_scene=config.get("cache_mode")=="full_scene_v4")

def training_identity(config,identity,datasets):
    return {**identity,"cache_manifests":{split:ds.manifest_sha256 for split,ds in datasets.items()}}

def validate_initialization(stage,*,resume=None,stage_a_checkpoint=None,from_first_stage=False):
    if stage=="a":
        if stage_a_checkpoint or from_first_stage:raise ValueError("Stage A cannot use Stage B initialization flags")
    elif stage=="b":
        if sum(bool(x) for x in [resume,stage_a_checkpoint,from_first_stage])!=1:
            raise ValueError("Stage B requires exactly one of --stage-a-checkpoint, --from-first-stage or --resume")
    else:raise ValueError("Unknown training stage")

def validate_candidate_contract(stage,masks,first_masks,main,first):
    if masks.shape!=first_masks.shape:raise RuntimeError("Candidate masks no longer align with the cached scene")
    if stage=="a" and (not torch.equal(masks,first_masks) or main["oracle_iou"]!=first["oracle_iou"]):
        raise RuntimeError("Stage A changed fixed candidate masks/oracle")
    if stage not in {"a","b"}:raise RuntimeError("Unknown evaluation stage")

def train_main():
    from .runtime import explicit_device
    parser=argparse.ArgumentParser(description="Stage A/B: cached object evidence and shared Qwen language LoRA")
    parser.add_argument("config",type=Path)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--stage-a-checkpoint",type=Path)
    parser.add_argument("--from-first-stage",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args();config=load_feedback_config(args.config)
    try:validate_initialization(config["feedback"]["stage"],resume=args.resume,
        stage_a_checkpoint=args.stage_a_checkpoint,from_first_stage=args.from_first_stage)
    except ValueError as error:parser.error(str(error))
    if args.output_dir:
        config["output_dir"]=str(ensure_within(args.output_dir,config["project_root"],must_exist=False))
    identity=provenance(config)
    cached={s:load_cache(config,identity,s) for s in ["train","val"]}
    identity=training_identity(config,identity,cached)
    output=Path(config["output_dir"])
    if output.exists() and any(output.glob("*.pt")) and not args.resume:
        raise FileExistsError("Output has checkpoints; use --resume or a new output directory")
    device=explicit_device()
    rank=int(os.getenv("RANK","0"));world=int(os.getenv("WORLD_SIZE","1"))
    if world>1:dist.init_process_group("nccl",timeout=timedelta(hours=2))
    control=dist.new_group(backend="gloo",timeout=timedelta(hours=2)) if world>1 else None
    seed=int(config["training"]["seed"])
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    model=build_model(config,identity).to(device).train()
    if args.stage_a_checkpoint:
        parent=ensure_within(args.stage_a_checkpoint,config["project_root"],must_exist=True)
        import_stage_a(model,parent,identity)
    _,initial=cached_loader(cached["train"],config,"train",rank,world)
    accumulation=int(config["training"]["gradient_accumulation_steps"])
    total=math.ceil(len(initial)/accumulation)*int(config["training"]["epochs"]);del initial
    optimizer=torch.optim.AdamW(model.optimizer_parameter_groups())
    minimum=float(config["training"]["cosine_min_lr_ratio"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(step/max(total,1),1))))
    step=0;start_epoch=0
    if args.resume:
        path=ensure_within(args.resume,config["project_root"],must_exist=True)
        step,start_epoch=load_checkpoint(model,path,config,identity,optimizer=optimizer,
                                        scheduler=scheduler,restore_rng_state=True)
    wrapped=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False) if world>1 else model
    output.mkdir(parents=True,exist_ok=True)
    logger=AsyncJsonLog(output/"metrics.jsonl") if rank==0 else None
    if rank==0:
        report={"total_parameters":sum(p.numel() for p in model.parameters()),
                "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
                "optimizer_groups":{g["name"]:sum(p.numel() for p in g["params"]) for g in optimizer.param_groups},
                "cache_manifests":identity["cache_manifests"]}
        (output/"parameter_report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report),flush=True)
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch,int(config["training"]["epochs"])):
            _,loader=cached_loader(cached["train"],config,"train",rank,world,epoch)
            torch.cuda.reset_peak_memory_stats(device)
            for micro,records in enumerate(loader):
                update=(micro+1)%accumulation==0 or micro+1==len(loader)
                window=min(accumulation,len(loader)-(micro//accumulation)*accumulation)
                sync=wrapped.no_sync() if world>1 and not update else contextlib.nullcontext()
                with sync,torch.autocast("cuda",dtype=torch.bfloat16,enabled=bool(config["training"]["amp"])):
                    result=wrapped(records)
                    if not torch.isfinite(result["loss"]):raise FloatingPointError("Nonfinite Stage A loss")
                    (result["loss"]/window).backward()
                if not update:continue
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                    float(config["training"]["max_grad_norm"]),error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step();step+=1
                if rank==0 and step%int(config["training"]["log_every"])==0:
                    record={"epoch":epoch+1,"step":step,"method":model.method,"evidence_mode":model.mode,
                        **{k:float(v) for k,v in result["losses"].items()},
                        "first_failure_rate":sum(result["first_failed"])/len(records),
                        "prefix_lengths":result["prefix_lengths"],"training_memory":memory_report()}
                    logger.submit(record);print(json.dumps(record),flush=True)
            states=[None]*world
            if world>1:dist.all_gather_object(states,capture_rng(),group=control)
            else:states=[capture_rng()]
            if rank==0:
                path=output/f"epoch_{epoch+1:03d}_step_{step:08d}.pt"
                save_checkpoint(model,optimizer,scheduler,path,step=step,epoch=epoch+1,
                    config=config,provenance=identity,rng_states=states)
                temporary=output/"latest.pt.tmp"
                temporary.unlink(missing_ok=True);os.link(path,temporary);os.replace(temporary,output/"latest.pt")
                print(json.dumps({"saved":str(path),"training_memory":memory_report()}),flush=True)
            if world>1:dist.barrier(group=control)
    finally:
        if logger:logger.close()
        if world>1:dist.destroy_process_group()

def baseline_result(records,device):
    return {"masks":[r["masks"].to(device) for r in records],
        "scores":[r["first_scores"].to(device) for r in records],
        "first_masks":[r["masks"].to(device) for r in records],
        "first_scores":[r["first_scores"].to(device) for r in records],
        "answers":[r["answer"] for r in records],"first_answers":[r["answer"] for r in records],
        "failed":[r["failed"] for r in records],"first_failed":[r["failed"] for r in records],
        "prefix_lengths":[len(r["scene"]) for r in records]}

def evaluate_main():
    from .runtime import explicit_device
    parser=argparse.ArgumentParser(description="Evaluate object-feedback Stage A/B with generated second-pass predictions")
    parser.add_argument("config",type=Path);parser.add_argument("checkpoint",type=Path,nargs="?")
    parser.add_argument("--baseline",action="store_true")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--run-id",default=os.getenv("FEEDBACK_RUN_ID"))
    args=parser.parse_args()
    if args.baseline==bool(args.checkpoint):parser.error("Choose --baseline or a feedback checkpoint")
    config=load_feedback_config(args.config)
    identity=provenance(config);val=load_cache(config,identity,"val",max_samples=args.max_samples)
    # A checkpoint binds both split manifests, so full training cache is required
    # for feedback evaluation as well. Baseline-only acceptance needs only val.
    datasets={"val":val}
    if args.checkpoint:datasets["train"]=load_cache(config,identity,"train")
    identity=training_identity(config,identity,datasets)
    output=ensure_within(args.output_dir,config["project_root"],must_exist=False)
    rank=int(os.getenv("RANK","0"));world=int(os.getenv("WORLD_SIZE","1"))
    if world>1 and not args.run_id:parser.error("Distributed evaluation requires a shared run ID")
    run_id=args.run_id or __import__("uuid").uuid4().hex
    device=explicit_device()
    if rank==0:
        output.mkdir(parents=True,exist_ok=False)
        (output/"run.json").write_text(json.dumps({"run_id":run_id,"world_size":world}))
    else:
        deadline=time.monotonic()+120
        while not (output/"run.json").exists():
            if time.monotonic()>deadline:raise TimeoutError("Rank zero did not initialize evaluation")
            time.sleep(.2)
        if json.loads((output/"run.json").read_text())!={"run_id":run_id,"world_size":world}:
            raise ValueError("Evaluation directory belongs to another run")
    model=None
    if args.checkpoint:
        model=build_model(config,identity).to(device).eval()
        checkpoint=ensure_within(args.checkpoint,config["project_root"],must_exist=True)
        load_checkpoint(model,checkpoint,config,identity,evaluation=True)
    _,loader=cached_loader(val,config,"val",rank,world,max_samples=args.max_samples)
    records=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for items in loader:
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=bool(config["training"]["amp"])):
                result=baseline_result(items,device) if args.baseline else model.predict(items)
            for i,item in enumerate(items):
                main=candidate_metrics(result["scores"][i],result["masks"][i],item["mapping"],item["truth"])
                first=candidate_metrics(result["first_scores"][i],result["first_masks"][i],item["mapping"],item["truth"])
                validate_candidate_contract(config["feedback"]["stage"],result["masks"][i],
                                            result["first_masks"][i],main,first)
                e=item["evidence"]
                records.append({"sample_id":item["sample_id"],"index":item["index"],"main":main,"first":first,
                    "strict":zero_metrics(main) if result["failed"][i] else main,"failed":result["failed"][i],
                    "first_failed":item["failed"],"answer":result["answers"][i],"first_answer":item["answer"],
                    "prefix_length":result["prefix_lengths"][i],
                    "objects":[{"role":role,"phrase":phrase,"source_query":int(q),"valid":bool(v)}
                        for role,phrase,q,v in zip(e["roles"],e["phrases"],e["source_query_ids"],e["valid_mask"])]})
    torch.cuda.synchronize(device)
    publish_shard(output/"shards",run_id,rank,world,{"records":records,
        "seconds":time.perf_counter()-started,"memory":memory_report()})
    if rank==0:
        shards=wait_for_shards(output/"shards",run_id,world,timeout=3600)
        records=[r for shard in shards for r in shard["records"]]
        expected=min(len(val),args.max_samples) if args.max_samples else len(val)
        if (len(records)!=expected or len({r["sample_id"] for r in records})!=expected
            or sorted(r["index"] for r in records)!=list(range(expected))):
            raise ValueError("Evaluation sample coverage/uniqueness mismatch")
        summary={**summarize(records),"config":config,"provenance":identity,
            "checkpoint":str(args.checkpoint) if args.checkpoint else None,"baseline":args.baseline,
            "stage":config["feedback"]["stage"],"stage_a_parent":getattr(model,"_stage_a_parent",None),
            "oracle_iou_delta":sum(r["main"]["oracle_iou"]-r["first"]["oracle_iou"] for r in records)/len(records),
            "newly_covered50":sum(r["first"]["oracle_iou"]<.5<=r["main"]["oracle_iou"] for r in records)/len(records),
            "lost_coverage50":sum(r["main"]["oracle_iou"]<.5<=r["first"]["oracle_iou"] for r in records)/len(records),
            "teacher_fallbacks":0,"gt_used_in_prediction":False,
            "first_failure_rate":sum(r["first_failed"] for r in records)/len(records),
            "rank_runtime":[{k:v for k,v in s.items() if k!="records"} for s in shards]}
        (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        with (output/"predictions.jsonl").open("w") as handle:
            for row in sorted(records,key=lambda r:r["index"]):handle.write(json.dumps(row,allow_nan=False)+"\n")
        print(json.dumps(summary),flush=True)
