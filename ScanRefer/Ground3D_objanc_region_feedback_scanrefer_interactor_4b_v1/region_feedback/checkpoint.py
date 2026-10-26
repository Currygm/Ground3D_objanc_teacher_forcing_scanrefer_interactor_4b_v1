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
        "tokenizer_snapshot":_snapshot(model),"stage_a_parent":parent if parent is not None else getattr(model,"_stage_a_parent",None)},path)
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

def import_stage_a(model,path,provenance):
    from .model import OBJECTIVE,PROTOCOL
    from .object_prompt import PROMPT_PROTOCOL
    if model.config["feedback"]["stage"]!="b":raise ValueError("Only Stage B can import a Stage A reader")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    signature=checkpoint.get("architecture_signature",{});old=signature.get("feedback",{})
    if (checkpoint.get("format_version")!=3 or checkpoint.get("objective")!=OBJECTIVE
        or old.get("stage")!="a" or old.get("protocol")!=PROTOCOL or old.get("method")!="qwen_feedback"
        or signature.get("prompt_protocol")!=PROMPT_PROTOCOL):
        raise ValueError("Not a compatible Stage A text-score checkpoint")
    new=model.config["feedback"]
    if any(old.get(k)!=new.get(k) for k in ["evidence_mode","input_dim","language_dim","hidden_dim","num_queries"]):
        raise ValueError("Stage A/B reader dimensions or evidence control differ")
    saved=checkpoint.get("provenance",{})
    for key in ["first_sha","data","base_config","generation_protocol"]:
        if key not in saved or key not in provenance or saved[key]!=provenance[key]:
            raise ValueError("Stage A/B original source, data or preprocessing differs")
    def source_code(identity):
        implementation=identity.get("implementation")
        if not isinstance(implementation,dict):raise ValueError("Missing Stage A/B source code provenance")
        return {k:v for k,v in implementation.items() if k.startswith(("ground3d_migration/",
            "spatiallm_compressed_visual_objanc_full_attention/")) or k in {
                "runtime_helpers.py","region_feedback/proposals.py","region_feedback/objects.py","region_feedback/regions.py"}}
    if source_code(saved)!=source_code(provenance):raise ValueError("Stage A/B first-pass implementation differs")
    if checkpoint.get("config",{}).get("training",{}).get("seed",42)!=model.config.get("training",{}).get("seed",42):
        raise ValueError("Stage A/B cached preprocessing seed differs")
    if checkpoint.get("tokenizer_snapshot")!=_snapshot(model):raise ValueError("Stage A/B tokenizer protocol differs")
    prefixes=("language.","object_projection.")
    expected={n:v for n,v in model.trainable_state_dict().items() if n.startswith(prefixes)}
    incoming=checkpoint.get("trainable_state")
    if (not isinstance(incoming,dict) or set(incoming)!=set(expected)
        or any(not isinstance(v,torch.Tensor) or v.shape!=expected[n].shape or not torch.isfinite(v).all()
               for n,v in incoming.items())):
        raise ValueError("Stage A reader state is incomplete or contains unexpected tensors")
    with torch.no_grad():
        destination=model.state_dict()
        for name,value in incoming.items():destination[name].copy_(value)
    parent={"path":str(Path(path).resolve()),"sha256":sha256(path)}
    model._stage_a_parent=parent
    return parent
