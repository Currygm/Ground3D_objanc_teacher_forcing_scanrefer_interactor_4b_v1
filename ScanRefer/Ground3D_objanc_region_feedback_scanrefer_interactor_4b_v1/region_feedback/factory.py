"""Pinned import of the evaluated Interactor checkpoint."""
from pathlib import Path
import torch
from runtime_helpers import sha256,load_trainable_state,validate_checkpoint
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
from .proposals import FrozenProposalStage
from .model import FeedbackModel

def provenance(config):
    base=load_segmentation_config(config["base_config"])
    pin=config["first_stage"]
    actual=sha256(pin["checkpoint"])
    if actual!=pin["sha256"]:
        raise ValueError("Frozen first-stage checkpoint SHA256 changed")
    data={s["name"]:sha256(s["annotations"]) for s in base["data"]["train_sources"]}
    data["validation"]=sha256(base["data"]["val_annotations"])
    root=Path(__file__).resolve().parents[1]
    paths=sorted((root/"region_feedback").glob("*.py"))
    paths+=sorted((root/"ground3d_migration").glob("*.py"))
    paths+=sorted((root/"spatiallm_compressed_visual_objanc_full_attention").rglob("*.py"))
    paths+=[root/name for name in ["runtime_helpers.py","train_feedback.py","evaluate_feedback.py"]]
    return {"first_sha":actual,"first_path":str(Path(pin["checkpoint"]).resolve()),
        "data":data,"base_config":base,"generation_protocol":pin["generation_protocol"],
        "implementation":{str(p.relative_to(root)):sha256(p) for p in paths}}

def build_first_stage(config,identity=None):
    identity=provenance(config) if identity is None else identity
    base=load_segmentation_config(config["base_config"])
    model=ObjAncSegmentationModel(base)
    checkpoint=torch.load(config["first_stage"]["checkpoint"],map_location="cpu",weights_only=False)
    validate_checkpoint(checkpoint,model.objective,model.architecture_signature(),model.answer_lm.source_manifests)
    if checkpoint.get("training_stage")!="scanrefer_teacher_forcing_segmentation":
        raise ValueError("First-stage import requires a completed joint checkpoint")
    if checkpoint.get("data_sha256")!=identity["data"]:
        raise ValueError("First-stage annotation hashes differ")
    if checkpoint.get("tokenizer_snapshot")!=model.answer_lm.tokenizer_snapshot():
        raise ValueError("First-stage tokenizer snapshot differs")
    load_trainable_state(model,checkpoint.get("trainable_state"))
    return FrozenProposalStage(model,config["first_stage"]["generation_protocol"])

def build_model(config,identity=None):
    first=build_first_stage(config,identity)
    return FeedbackModel(first.model.answer_lm,config,text_adapter=first.model.text_adapter,decoder=first.model.decoder)


def cache_identity(config,identity=None):
    identity=provenance(config) if identity is None else identity
    relevant={k:v for k,v in identity["implementation"].items() if not k.startswith("region_feedback/")
        or k in {"region_feedback/objects.py","region_feedback/regions.py","region_feedback/proposals.py",
                  "region_feedback/cache.py","region_feedback/cache_runtime.py",
                  "region_feedback/factory.py","region_feedback/config.py"}}
    # Only code on the first-pass path belongs to immutable cache identity.
    relevant={k:v for k,v in relevant.items() if k not in {"train_feedback.py","evaluate_feedback.py"}}
    return {"first_sha":identity["first_sha"],"data":identity["data"],"base_config":identity["base_config"],
            "implementation":relevant,"protocol":"fixed_objects_textscore_v3","coordinate_policy":"seed_per_sample_v2",
            "generation_protocol":identity.get("generation_protocol"),
            "cache_seed":int(config.get("training",{}).get("seed",42))}
