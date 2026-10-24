"""ScanRefer Answer3 -> joint10 training, with separate optimizer schedules."""
from __future__ import annotations
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ground3d_migration.runtime import capture_rng,restore_rng,memory_report
from runtime_helpers import (AsyncJsonLog,atomic_save,checkpoint_manifest,frozen_digest,
                            load_trainable_state,validate_checkpoint,sha256)
from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within,load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.data import collate_samples
from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import build_segmentation_dataset
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import OBJECTIVE,ObjAncSegmentationModel
from spatiallm_compressed_visual_objanc_full_attention.model.answer_stage_3b import (
    ANSWER_OBJECTIVE,AnswerPretrainingStage,answer_signature,is_answer_parameter)

STAGE_NAMES={'answer':'answer_pretraining','joint':'scanrefer_teacher_forcing_segmentation'}

def stage_settings(config,stage_name):
    if stage_name not in STAGE_NAMES:raise ValueError(f'Unknown stage: {stage_name}')
    return {'epochs':int(config['training']['answer_pretrain_epochs'] if stage_name=='answer' else config['training']['epochs']),
        'output_dir':str(Path(config['project']['artifact_root'])/'outputs/open_answer') if stage_name=='answer' else config['project']['output_dir']}

def load_answer_checkpoint(model,path,manifest):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(checkpoint,ANSWER_OBJECTIVE,answer_signature(model.answer_lm,model.config),manifest)
    if checkpoint.get('training_stage')!='answer_pretraining' or checkpoint.get('completed_epochs')!=3:
        raise ValueError('Joint training requires the completed three-epoch Answer checkpoint')
    if checkpoint.get('data_sha256')!=data_manifest(model.config):raise ValueError('Answer training data SHA256 differs')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad and is_answer_parameter(n)}
    state=checkpoint.get('trainable_state')
    if not isinstance(state,dict) or set(state)!=set(parameters):raise ValueError('Answer checkpoint task parameter keys differ')
    if any(not isinstance(state[n],torch.Tensor) or state[n].shape!=p.shape or not torch.isfinite(state[n]).all()
           for n,p in parameters.items()):raise ValueError('Answer checkpoint task weights have invalid shapes/values')
    restore_tokenizer_checkpoint(model,checkpoint)
    with torch.no_grad():
        for name,parameter in parameters.items():parameter.copy_(state[name])
    return {'path':str(path),'sha256':sha256(path),'completed_epochs':3}

def tokenizer_checkpoint(model):
    language=getattr(model,'answer_lm',None)
    return language.tokenizer_snapshot() if hasattr(language,'tokenizer_snapshot') else None

def restore_tokenizer_checkpoint(model,checkpoint):
    expected=tokenizer_checkpoint(model)
    if expected!=checkpoint.get('tokenizer_snapshot'):raise ValueError('Checkpoint tokenizer snapshot differs')
    if expected is not None:model.answer_lm.restore_tokenizer_snapshot(checkpoint['tokenizer_snapshot'])

def scheduler_for(optimizer,total,minimum):
    return LambdaLR(optimizer,lambda step:minimum+(1-minimum)*.5*(1+math.cos(math.pi*min(step/max(total,1),1))))

def unwrap(model):
    return model.module if isinstance(model,DistributedDataParallel) else model

def data_manifest(config):
    data=config['data']
    result={s['name']:sha256(s['annotations']) for s in data['train_sources']}
    if data.get('val_annotations'):
        result['validation']=sha256(data['val_annotations'])
    return result

def check_output_directory(output,resume):
    if not resume and Path(output).exists() and any(Path(output).glob('*.pt')):
        raise FileExistsError('Checkpoints already exist; use --resume or a fresh --output-dir')

def save_checkpoint(model,optimizer,scheduler,output,step,completed_epochs,config,*,manifest,frozen_hash,stage_name='joint',parent=None,rng_states=None):
    stage=unwrap(model); path=Path(output)/f'epoch_{completed_epochs:03d}_step_{step:08d}.pt'
    payload={'format_version':3,'objective':getattr(stage,'objective',OBJECTIVE),'training_stage':STAGE_NAMES[stage_name],
        'step':step,'completed_epochs':completed_epochs,'architecture_signature':stage.architecture_signature(),
        'trainable_state':stage.trainable_state_dict(),'optimizer':optimizer.state_dict(),
        'lr_scheduler':scheduler.state_dict(),'config':config,'base_checkpoint':config['model']['model_name_or_path'],
        'base_manifest':manifest,'frozen_hash':frozen_hash,
        'data_sha256':data_manifest(config),'answer_parent':parent,'tokenizer_snapshot':tokenizer_checkpoint(stage),'rng_states':rng_states or [capture_rng()]}
    atomic_save(payload,path)
    latest=Path(output)/'latest.pt'; temporary=latest.with_name('latest.pt.tmp')
    if temporary.exists():temporary.unlink()
    os.link(path,temporary); os.replace(temporary,latest)
    return path

def load_resume(model,optimizer,scheduler,path,manifest,stage_name='joint'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(checkpoint,getattr(model,'objective',OBJECTIVE),model.architecture_signature(),manifest)
    if checkpoint.get('training_stage')!=STAGE_NAMES[stage_name]:raise ValueError('Resume checkpoint belongs to another stage')
    if checkpoint['config']!=model.config:
        raise ValueError('Resume training config differs; schedule/data changes require a new experiment')
    if checkpoint['data_sha256']!=data_manifest(model.config):
        raise ValueError('Resume training data SHA256 differs')
    restore_tokenizer_checkpoint(model,checkpoint)
    load_trainable_state(model,checkpoint.get('trainable_state'))
    optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['lr_scheduler'])
    states=checkpoint.get('rng_states')
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    if not isinstance(states,list) or len(states)!=world:raise ValueError('Resume RNG/world size mismatch')
    restore_rng(states[rank])
    return int(checkpoint['step']),int(checkpoint['completed_epochs'])

def main(stage_name='joint'):
    parser=argparse.ArgumentParser()
    parser.add_argument('config',type=Path); parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--answer-checkpoint',type=Path)
    args=parser.parse_args(); config=load_segmentation_config(args.config)
    settings=stage_settings(config,stage_name)
    if stage_name=='joint' and bool(args.resume)==bool(args.answer_checkpoint):
        parser.error('Joint stage requires exactly one of --answer-checkpoint or --resume')
    if stage_name=='answer' and args.answer_checkpoint:parser.error('Answer stage starts from Stage1; use --resume only for Answer-stage continuation')
    output=ensure_within(args.output_dir or settings['output_dir'],config['project']['artifact_root'],must_exist=False)
    check_output_directory(output,args.resume)
    world,rank,local=(int(os.getenv(k,d)) for k,d in [('WORLD_SIZE','1'),('RANK','0'),('LOCAL_RANK','0')])
    distributed=world>1; control=None
    if distributed:
        torch.cuda.set_device(local); dist.init_process_group('nccl',timeout=timedelta(hours=2))
        control=dist.new_group(backend='gloo',timeout=timedelta(hours=2))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    manifest=[checkpoint_manifest(config) if rank==0 else None]
    if distributed:dist.broadcast_object_list(manifest,src=0,group=control)
    manifest=manifest[0]
    seed=int(config['training']['seed']); random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
    dataset=build_segmentation_dataset(config,'train')
    sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    workers=int(config['data']['num_workers'])
    loader=DataLoader(dataset,batch_size=int(config['data']['batch_size']),sampler=sampler,num_workers=workers,
        persistent_workers=workers>0,pin_memory=device.type=='cuda',collate_fn=collate_samples)
    accumulation=int(config['training']['gradient_accumulation_steps'])
    total=settings['epochs']*math.ceil(len(loader)/accumulation)
    stage=(AnswerPretrainingStage(config) if stage_name=='answer' else ObjAncSegmentationModel(config)).to(device)
    parent=None
    if args.answer_checkpoint:
        answer_path=ensure_within(args.answer_checkpoint,config['project']['artifact_root'],must_exist=True)
        parent=load_answer_checkpoint(stage,answer_path,manifest)
    optimizer=AdamW(stage.optimizer_parameter_groups())
    scheduler=scheduler_for(optimizer,total,float(config['training']['cosine_min_lr_ratio']))
    step,start_epoch=0,0
    if args.resume:
        resume=ensure_within(args.resume,config['project']['artifact_root'],must_exist=True)
        step,start_epoch=load_resume(stage,optimizer,scheduler,resume,manifest,stage_name)
        parent=torch.load(resume,map_location='cpu',weights_only=False).get('answer_parent')
    stage.train()
    model=DistributedDataParallel(stage,device_ids=[local],broadcast_buffers=False) if distributed else stage
    frozen_before=frozen_digest(stage)  # after DDP initialization broadcasts
    if args.resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['frozen_hash']!=frozen_before:raise RuntimeError('Frozen resume weights differ')
    output.mkdir(parents=True,exist_ok=True)
    logger=AsyncJsonLog(output/'metrics.jsonl') if rank==0 else None
    optimizer.zero_grad(set_to_none=True); started=time.time()
    try:
        for epoch in range(start_epoch,settings['epochs']):
            sampler.set_epoch(epoch)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            for microstep,batch in enumerate(loader):
                should_step=(microstep+1)%accumulation==0 or microstep+1==len(loader)
                window=min(accumulation,len(loader)-(microstep//accumulation)*accumulation)
                sync=model.no_sync() if distributed and not should_step else contextlib.nullcontext()
                with sync,torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and bool(config['training']['amp'])):
                    result=model(batch,epoch=epoch,optimizer_step=step,steps_per_epoch=math.ceil(len(loader)/accumulation),answer_mode='teacher')
                    if not torch.isfinite(result['loss']):raise FloatingPointError(f'Nonfinite training loss at epoch {epoch+1}, step {step}')
                    (result['loss']/window).backward()
                if not should_step:continue
                torch.nn.utils.clip_grad_norm_([p for p in stage.parameters() if p.requires_grad],float(config['training']['max_grad_norm']),error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); step+=1
                if rank==0 and step%int(config['training']['log_every'])==0:
                    record={'stage':stage_name,'epoch':epoch+1,'step':step,'dataset_samples':len(dataset),'samples_per_rank':len(sampler),
                      'ddp_padding_samples':len(sampler)*world-len(dataset),
                      **{k:float(v) for k,v in result['losses'].items()},**{k:float(v) for k,v in result['metrics'].items()},
                      'elapsed_sec':time.time()-started,'lr':{g['name']:g['lr'] for g in optimizer.param_groups}}
                    print(json.dumps(record,ensure_ascii=False),flush=True); logger.submit(record)
            frozen_after=frozen_digest(stage)
            if frozen_after!=frozen_before:raise RuntimeError('Frozen parameter hash changed')
            if distributed:dist.barrier(group=control)
            rng_states=[None]*world
            if distributed:dist.all_gather_object(rng_states,capture_rng(),group=control)
            else:rng_states=[capture_rng()]
            if rank==0:
                path=save_checkpoint(model,optimizer,scheduler,output,step,epoch+1,config,manifest=manifest,frozen_hash=frozen_before,stage_name=stage_name,parent=parent,rng_states=rng_states)
                if frozen_digest(stage)!=frozen_before:raise RuntimeError('Frozen parameter hash changed after save')
                print(json.dumps({'saved':str(path),'validation_performed':False,'training_memory':memory_report()}),flush=True)
            if distributed:dist.barrier(group=control)
    finally:
        if logger:logger.close()
        if distributed:dist.destroy_process_group()

if __name__=='__main__':main()
