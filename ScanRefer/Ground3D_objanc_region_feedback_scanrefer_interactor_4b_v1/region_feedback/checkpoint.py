"""Exact feedback checkpoints; first-stage weights are pinned external inputs."""
from pathlib import Path
import os
import torch
from runtime_helpers import atomic_save,load_trainable_state,sha256
from ground3d_migration.runtime import restore_rng
from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within

def _snapshot(model):
    lm=getattr(model,"language",None)
    return lm.tokenizer_snapshot() if hasattr(lm,"tokenizer_snapshot") else None

def save_checkpoint(model,optimizer,scheduler,path,*,step,epoch,config,provenance,rng_states,parent=None):
    atomic_save({"format_version":3,"objective":model.objective,
        "architecture_signature":model.architecture_signature(),"config":config,
        "provenance":provenance,"trainable_state":model.trainable_state_dict(),
        "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
        "step":int(step),"completed_epochs":int(epoch),"rng_states":rng_states,
        "tokenizer_snapshot":_snapshot(model),"stage_a_parent":parent},path)
    return Path(path)

def load_checkpoint(model,path,config,provenance,*,optimizer=None,scheduler=None,restore_rng_state=False,evaluation=False):
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("format_version")!=3:
        raise ValueError("Legacy feedback checkpoint format is not supported")
    expected_config=config
    if evaluation:
        if optimizer is not None or scheduler is not None or restore_rng_state:
            raise ValueError("Evaluation path relaxation cannot be used for training resume")
        saved_config=checkpoint.get("config",{})
        if "output_dir" not in saved_config or "output_dir" not in config:
            raise ValueError("Missing original training output path")
        ensure_within(saved_config["output_dir"],config["project_root"],must_exist=False)
        expected_config={**config,"output_dir":saved_config["output_dir"]}
    if (checkpoint.get("objective")!=model.objective
        or checkpoint.get("architecture_signature")!=model.architecture_signature()
        or checkpoint.get("config")!=expected_config or checkpoint.get("provenance")!=provenance):
        raise ValueError("Feedback checkpoint source, data, config or architecture changed")
    if checkpoint.get("tokenizer_snapshot")!=_snapshot(model):
        raise ValueError("Feedback tokenizer protocol differs")
    rank=int(os.getenv("RANK","0"));world=int(os.getenv("WORLD_SIZE","1"))
    states=checkpoint.get("rng_states")
    if restore_rng_state and (not isinstance(states,list) or len(states)!=world):
        raise ValueError("Feedback resume RNG/world size differs")
    load_trainable_state(model,checkpoint.get("trainable_state"))
    if optimizer is not None:optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng_state:restore_rng(states[rank])
    model._stage_a_parent=checkpoint.get("stage_a_parent")
    model._loaded_training_config=checkpoint["config"]
    return int(checkpoint["step"]),int(checkpoint["completed_epochs"])

def import_stage_a(*args,**kwargs):
    raise ValueError("Stage B handoff is disabled for object-feedback v2")
