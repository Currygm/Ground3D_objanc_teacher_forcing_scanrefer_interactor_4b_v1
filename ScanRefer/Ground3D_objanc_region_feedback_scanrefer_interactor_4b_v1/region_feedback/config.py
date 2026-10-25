"""Stage A object-feedback v3 configuration; legacy experiments are explicit errors."""
from pathlib import Path
import re
import yaml
from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within,load_segmentation_config
from .model import PROTOCOL
PROJECT=Path(__file__).resolve().parents[1]

def load_feedback_config(path):
    config=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config,dict):raise ValueError("Unsupported feedback configuration")
    if config.get("feedback",{}).get("stage")=="b":
        raise ValueError("Stage B is disabled; object-feedback v3 currently supports Stage A only")
    if config.get("schema_version")!=3 or config.get("feedback",{}).get("protocol")!=PROTOCOL:
        raise ValueError("Legacy all-query configuration is disabled; choose feedback_a_visual.yaml")
    root=Path(config["project_root"]).resolve(strict=True)
    if root!=PROJECT or not root.is_relative_to(Path("/home/huangxiang/codes/guiming").resolve(strict=True)):
        raise ValueError("Wrong feedback artifact root")
    base=load_segmentation_config(config["base_config"])
    if Path(base["project"]["artifact_root"]).resolve()!=root:raise ValueError("Wrong base artifact root")
    first=config["first_stage"]
    ensure_within(first["checkpoint"],"/home/huangxiang/codes/guiming",must_exist=True)
    if not re.fullmatch("[0-9a-f]{64}",first["sha256"]) or first["generation_protocol"] not in {"constrained","unconstrained"}:
        raise ValueError("First-stage SHA256/protocol must be pinned")
    cfg=config["feedback"]
    if (cfg["input_dim"],cfg["language_dim"],cfg["num_queries"],cfg["hidden_dim"])!=(256,2560,50,256):
        raise ValueError("Wrong Ground3D object dimensions")
    if cfg["stage"]!="a" or cfg["evidence_mode"] not in {"empty","text","visual"} or cfg["method"] not in {"qwen_feedback","selector"}:
        raise ValueError("Wrong Stage A control")
    if cfg["method"]=="selector" and cfg["evidence_mode"]!="visual":raise ValueError("Selector uses visual evidence")
    if cfg["ranking_temperature"]<=0 or cfg["ranking_weight"]<=0:raise ValueError("Invalid ranking loss")
    for key in ["output_dir","evaluation_dir","cache_dir"]:
        config[key]=str(ensure_within(config[key],root,must_exist=False))
    t=config["training"]
    if any(int(t[k])<1 for k in ["epochs","batch_size","gradient_accumulation_steps","log_every"]):
        raise ValueError("Invalid training schedule")
    if t["num_workers"]<0 or not 0<=t["cosine_min_lr_ratio"]<=1:raise ValueError("Invalid training workers/scheduler")
    if any(float(config["optimizer"][k])<=0 for k in ["new_lr","qwen_lora_lr"]):
        raise ValueError("Learning rates must be positive")
    return config
