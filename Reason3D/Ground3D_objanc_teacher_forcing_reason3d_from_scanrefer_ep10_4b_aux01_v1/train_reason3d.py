"""Ground3D: fresh Reason3D Answer3 + joint10 from ScanRefer joint epoch10."""
from __future__ import annotations
import argparse,contextlib,json,math,os,random,time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ground3d_migration.runtime import capture_rng,restore_rng,memory_report
from runtime_helpers import atomic_save,sha256,AsyncJsonLog
from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within
from spatiallm_compressed_visual_objanc_full_attention.data import collate_samples
from reason3d_runtime import OBJECTIVE,ObjAncSegmentationModel,load_segmentation_config,build_segmentation_dataset,load_source_initialization,restore_task_state
from reason3d_training_schedule import set_training_epoch
from reason3d_stage_checkpoints import should_save_epoch,stage_progress


def scheduler_for(optimizer,total,minimum):
    return LambdaLR(optimizer,lambda step:minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(step/max(total,1),1))))


def data_manifest(config):
    return {"train":sha256(config["data"]["train_sources"][0]["annotations"]),"val":sha256(config["data"]["val_annotations"])}


def save_checkpoint(model,optimizer,scheduler,output,step,completed_epochs,config,rng_states):
    stage=model.module if isinstance(model,DDP) else model
    phase,phase_epoch=stage_progress(completed_epochs,int(config["training"]["answer_warmup_epochs"]))
    if stage._training_phase!=phase:raise ValueError("Checkpoint phase differs")
    payload={"format_version":1,"objective":OBJECTIVE,"training_stage":"reason3d_teacher_forcing_segmentation",
        "answer_training_phase":phase,"stage_epoch":phase_epoch,"completed_epochs":completed_epochs,"step":step,
        "architecture_signature":stage.architecture_signature(),"trainable_state":stage.trainable_state_dict(),
        "tokenizer_snapshot":stage.answer_lm.tokenizer_snapshot(),"base_manifest":stage.answer_lm.source_manifests,
        "optimizer":optimizer.state_dict(),"lr_scheduler":scheduler.state_dict(),"config":config,
        "data_sha256":data_manifest(config),"rng_states":rng_states,"initialized_from":config["finetune_initialization"]}
    path=Path(output)/f"epoch_{completed_epochs:03d}_step_{step:08d}.pt"
    atomic_save(payload,path)
    return path


def load_resume(model,optimizer,scheduler,path,world):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("objective")!=OBJECTIVE or checkpoint.get("architecture_signature")!=model.architecture_signature():raise ValueError("Resume objective or architecture differs")
    if checkpoint.get("config")!=model.config:raise ValueError("Resume config differs")
    if checkpoint.get("base_manifest")!=model.answer_lm.source_manifests or checkpoint.get("tokenizer_snapshot")!=model.answer_lm.tokenizer_snapshot():raise ValueError("Resume frozen base/tokenizer differs")
    if checkpoint.get("data_sha256")!=data_manifest(model.config):raise ValueError("Resume dataset hash differs")
    completed=int(checkpoint["completed_epochs"]);step=int(checkpoint["step"])
    phase,stage_epoch=stage_progress(completed,int(model.config["training"]["answer_warmup_epochs"]))
    if not 0<completed<=int(model.config["training"]["epochs"]) or checkpoint.get("answer_training_phase")!=phase or checkpoint.get("stage_epoch")!=stage_epoch:raise ValueError("Resume stage metadata differs")
    if checkpoint["lr_scheduler"]["last_epoch"]!=step:raise ValueError("Resume scheduler step differs")
    states=checkpoint.get("rng_states")
    if not isinstance(states,list) or len(states)!=world:raise ValueError("Resume RNG/world size differs")
    restore_task_state(model,checkpoint["trainable_state"])
    optimizer.load_state_dict(checkpoint["optimizer"]);scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return step,completed,states


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("config",type=Path);parser.add_argument("--output-dir",type=Path);parser.add_argument("--resume",type=Path)
    parser.add_argument("--max-steps",type=int,help="Debug early stop; incomplete epochs are not saved or auto-evaluated")
    args=parser.parse_args();config=load_segmentation_config(args.config)
    output=ensure_within(args.output_dir or config["project"]["output_dir"],config["project"]["artifact_root"],must_exist=False)
    if not args.resume and output.exists() and (any(output.glob("*.pt")) or (output/"metrics.jsonl").exists()):raise FileExistsError("Use --resume or a fresh OUTPUT_DIR")
    world,rank,local=[int(os.environ.get(k,default)) for k,default in [("WORLD_SIZE","1"),("RANK","0"),("LOCAL_RANK","0")]]
    distributed=world>1;control=None
    if distributed:
        torch.cuda.set_device(local);dist.init_process_group("nccl",timeout=timedelta(hours=2));control=dist.new_group(backend="gloo",timeout=timedelta(hours=2))
    device=torch.device("cuda",local) if torch.cuda.is_available() else torch.device("cpu")
    print(json.dumps({"rank":rank,"logical_device":local,"visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"device":str(device)}),flush=True)
    t=config["training"];seed=int(t["seed"]);random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    dataset=build_segmentation_dataset(config,"train")
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    generator=torch.Generator().manual_seed(seed+rank)
    workers=int(config["data"]["num_workers"])
    loader=DataLoader(dataset,batch_size=int(config["data"]["batch_size"]),sampler=sampler,num_workers=workers,
        persistent_workers=workers>0,pin_memory=device.type=="cuda",collate_fn=collate_samples,generator=generator)
    accumulation=int(t["gradient_accumulation_steps"]);steps_per_epoch=math.ceil(len(loader)/accumulation);total_steps=int(t["epochs"])*steps_per_epoch
    stage=ObjAncSegmentationModel(config)
    if not args.resume:load_source_initialization(stage,Path(config["finetune_initialization"]["checkpoint"]),config)
    stage.to(device)
    optimizer=AdamW(stage.optimizer_parameter_groups());scheduler=scheduler_for(optimizer,total_steps,float(t["cosine_min_lr_ratio"]))
    step,start_epoch=0,0;rng_states=None
    if args.resume:
        resume=ensure_within(args.resume,config["project"]["artifact_root"],must_exist=True)
        step,start_epoch,rng_states=load_resume(stage,optimizer,scheduler,resume,world)
        if step!=start_epoch*steps_per_epoch:raise ValueError("Resume world/batch/accumulation changed")
    output.mkdir(parents=True,exist_ok=True);logger=AsyncJsonLog(output/"metrics.jsonl") if rank==0 else None
    if rank==0:
        (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
        print(json.dumps({"train_samples":len(dataset),"steps_per_epoch":steps_per_epoch,"total_steps":total_steps,"answer_epochs":3,"joint_epochs":10}),flush=True)
    optimizer.zero_grad(set_to_none=True);started=time.time();model=None;active=None;stop=False
    try:
        for epoch in range(start_epoch,int(t["epochs"])):
            phase="answer_only" if epoch<int(t["answer_warmup_epochs"]) else "joint"
            if phase!=active:
                if distributed:dist.barrier(group=control)
                model=None;set_training_epoch(stage,epoch)
                model=DDP(stage,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if distributed else stage
                active=phase
                if rng_states is not None:restore_rng(rng_states[rank]);rng_states=None
                print(json.dumps({"rank":rank,"event":"training_phase","phase":phase,"global_epoch":epoch+1,"trainable_tensors":sum(p.requires_grad for p in stage.parameters())}),flush=True)
            sampler.set_epoch(epoch);generator.manual_seed(seed+rank+epoch*100003)
            if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
            for microstep,batch in enumerate(loader):
                should_step=(microstep+1)%accumulation==0 or microstep+1==len(loader)
                window=min(accumulation,len(loader)-(microstep//accumulation)*accumulation)
                sync=model.no_sync() if distributed and not should_step else contextlib.nullcontext()
                try:
                    with sync,torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda" and bool(t["amp"])):
                        result=model(batch,epoch=epoch,optimizer_step=step,steps_per_epoch=steps_per_epoch,answer_mode="teacher")
                        if not torch.isfinite(result["loss"]):raise FloatingPointError("Nonfinite training loss")
                        (result["loss"]/window).backward()
                except (FloatingPointError,torch.OutOfMemoryError) as exc:
                    print(json.dumps({"event":type(exc).__name__,"rank":rank,"phase":active,"epoch":epoch+1,"step":step,"scene_ids":batch.get("scene_id"),"sample_ids":batch.get("sample_id"),"raw_points":[len(x) for x in batch["point_cloud"]]},ensure_ascii=False),flush=True);raise
                if not should_step:continue
                norm=torch.nn.utils.clip_grad_norm_([p for p in stage.parameters() if p.requires_grad],float(t["max_grad_norm"]),error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step();step+=1
                if rank==0 and step%int(t["log_every"])==0:
                    record={"epoch":epoch+1,"stage_epoch":stage_progress(epoch+1,3)[1],"training_phase":active,"step":step,"grad_norm":float(norm),
                        "dataset_samples":len(dataset),"samples_per_rank":len(sampler),"ddp_padding_samples":len(sampler)*world-len(dataset),
                        **{k:float(v) for k,v in result["losses"].items()},**{k:float(v) for k,v in result["metrics"].items()},
                        "elapsed_sec":time.time()-started,"lr":{g["name"]:g["lr"] for g in optimizer.param_groups},**memory_report()}
                    print(json.dumps(record),flush=True);logger.submit(record)
                if args.max_steps is not None and step>=args.max_steps:stop=True;break
            completed=epoch+1
            if microstep+1==len(loader) and should_save_epoch(completed,3,13,int(t["save_every_epochs"])):
                if distributed:
                    states=[None]*world;dist.all_gather_object(states,capture_rng(),group=control)
                else:states=[capture_rng()]
                if rank==0:print(f"Saved {save_checkpoint(model,optimizer,scheduler,output,step,completed,config,states)}",flush=True)
                if distributed:dist.barrier(group=control)
            if stop:break
    finally:
        if logger:logger.close()
        if distributed:dist.destroy_process_group()


if __name__=="__main__":main()
