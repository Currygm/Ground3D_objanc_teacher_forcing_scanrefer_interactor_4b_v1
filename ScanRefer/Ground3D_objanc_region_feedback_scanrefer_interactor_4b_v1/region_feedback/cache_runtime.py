"""Generate original-checkpoint caches before any Stage A LoRA update."""
import argparse
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import torch.distributed as dist
from runtime_helpers import sha256
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import build_segmentation_dataset
from spatiallm_compressed_visual_objanc_full_attention.data import collate_samples
from .config import load_feedback_config
from .factory import provenance,cache_identity,build_first_stage
from .cache import CacheWriter,make_record,validate_record

def check_disk_budget(sample_bytes,count,budget_gib,free_bytes):
    estimate=int(sample_bytes)*int(count)
    if estimate>float(budget_gib)*1024**3:raise ValueError("Estimated cache exceeds --max-cache-gib budget")
    if estimate>int(free_bytes):raise ValueError("Insufficient free disk space for estimated cache")
    return estimate

def cache_main():
    from .runtime import explicit_device
    parser=argparse.ArgumentParser(description="Cache frozen original-checkpoint OBJ/ANC object evidence")
    parser.add_argument("config",type=Path)
    parser.add_argument("--split",choices=["train","val"],required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-samples",type=int)
    parser.add_argument("--max-cache-gib",type=float,default=200.)
    args=parser.parse_args()
    if args.max_samples is not None and args.max_samples<1:parser.error("max-samples must be positive")
    if args.max_cache_gib<=0:parser.error("max-cache-gib must be positive")
    config=load_feedback_config(args.config);identity=provenance(config)
    source_identity=cache_identity(config,identity)
    full_scene=config.get("cache_mode")=="full_scene_v4"
    base=load_segmentation_config(config["base_config"])
    dataset=build_segmentation_dataset(base,args.split)
    expected=36569 if args.split=="train" else 9483
    if len(dataset)!=expected:raise ValueError("Source dataset count changed")
    count=min(expected,args.max_samples) if args.max_samples else expected
    root=Path(config["cache_dir"])/args.split
    rank=int(os.getenv("RANK","0"));world=int(os.getenv("WORLD_SIZE","1"))
    device=explicit_device()
    if world>1:dist.init_process_group("gloo",timeout=timedelta(hours=2))
    try:
        if rank==0:writer=CacheWriter(root,args.split,source_identity,count,resume=args.resume,full_scene=full_scene)
        if world>1:dist.barrier()
        if rank:writer=CacheWriter(root,args.split,source_identity,count,resume=True,full_scene=full_scene)
        # Never load feedback weights here. Original checkpoint remains pinned.
        first=build_first_stage(config,identity).to(device).eval()
        seed=int(config["training"]["seed"]);budget_checked=False
        for index in range(rank,count,world):
            path=root/"records"/f"{index:08d}.pt"
            random.seed(seed+index);np.random.seed(seed+index);torch.manual_seed(seed+index)
            item=dataset[index]
            if path.exists():
                if not args.resume:raise FileExistsError("Cache exists; use --resume")
                value=writer.validate_existing(index)
                if value["sample_id"]!=str(item["sample_id"]):raise ValueError("Cached sample binding changed")
                # A finalized cache also protects resumed records by original hash.
                manifest=root/"manifest.json"
                if manifest.is_file():
                    entry=json.loads(manifest.read_text())["entries"][index]
                    if sha256(path)!=entry["sha256"]:raise ValueError("Existing finalized cache checksum changed")
                continue
            batch=collate_samples([item])
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=bool(config["training"]["amp"])):
                proposal=first.predict(batch)[0]
                value=make_record(proposal,batch,0,first.model.answer_lm,index=index,include_full_scene=full_scene)
            if not budget_checked:
                buffer=io.BytesIO();torch.save(value,buffer);sample_bytes=buffer.tell()
                remaining=count-len(list((root/"records").glob("*.pt")))
                total_estimate=check_disk_budget(sample_bytes,count,args.max_cache_gib,2**63-1)
                check_disk_budget(sample_bytes,remaining,args.max_cache_gib,shutil.disk_usage(root).free)
                print(json.dumps({"rank":rank,"pilot_sample_bytes":sample_bytes,
                    "estimated_total_bytes":total_estimate,"remaining_records":remaining,
                    "note":"Pilot estimate; per-sample sizes vary. Budget also checked cumulatively."}),flush=True)
                budget_checked=True
            writer.write(index,value)
            # Rank-local bytes plus global current disk count prevent silent unlimited writes.
            if index%100==rank or index+world>=count:
                used=sum(p.stat().st_size for p in (root/"records").glob("*.pt"))
                if used>args.max_cache_gib*1024**3:raise ValueError("Actual cache exceeded disk budget")
            if index%100==rank:print(json.dumps({"rank":rank,"cached_index":index,"count":count}),flush=True)
        if world>1:dist.barrier()
        if rank==0:
            result=writer.finalize()
            print(json.dumps({"cache_complete":True,"split":args.split,"count":result["count"],
                              "manifest":str(root/"manifest.json")}),flush=True)
    finally:
        if world>1:dist.destroy_process_group()
