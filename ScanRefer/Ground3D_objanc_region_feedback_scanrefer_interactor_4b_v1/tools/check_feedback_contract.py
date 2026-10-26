"""Metadata-only preflight; full CUDA/model construction is deliberately excluded."""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from region_feedback.config import load_feedback_config
from region_feedback.factory import provenance
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config

def main():
    root=Path(__file__).resolve().parents[1]
    configs=[load_feedback_config(root/"configs"/("feedback_a_"+mode+".yaml"))
             for mode in ["empty","text","visual","selector"]]
    configs.append(load_feedback_config(root/"configs/feedback_b_visual.yaml"))
    for name in ["feedback_a_visual_geometry.yaml","feedback_b_visual_geometry.yaml"]:
        try:load_feedback_config(root/"configs"/name)
        except ValueError:pass
        else:raise ValueError("Legacy configuration must be disabled")
    identity=provenance(configs[0]);base=load_segmentation_config(configs[0]["base_config"])
    checkpoint=torch.load(configs[0]["first_stage"]["checkpoint"],map_location="cpu",weights_only=False,mmap=True)
    if checkpoint["objective"]!="ground3d_objanc_teacher_forcing_scanrefer_interactor_4b_v1":
        raise ValueError("Wrong original objective")
    if checkpoint["training_stage"]!="scanrefer_teacher_forcing_segmentation":
        raise ValueError("Wrong original training stage")
    if checkpoint["data_sha256"]!=identity["data"]:raise ValueError("Annotation provenance changed")
    signature=checkpoint["architecture_signature"]
    for key,expected in [("decoder",base["decoder"]),("loss",base["loss"]),
                         ("point_encoder",base["point_encoder"]),("zero_obj_protocol",base["answer_policy"])]:
        if signature[key]!=expected:raise ValueError("Original signature differs: "+key)
    counts={}
    for name,path in [("train",base["data"]["train_sources"][0]["annotations"]),("val",base["data"]["val_annotations"])]:
        value=json.loads(Path(path).read_text())
        counts[name]=len(value["annotations"] if isinstance(value,dict) else value)
    if counts!={"train":36569,"val":9483}:raise ValueError("Unexpected annotation counts")
    print(json.dumps({"metadata_preflight_only":True,"full_4b_model_constructed":False,"gpu_executed":False,
        "protocol":"object_evidence_shared_qwen_textscore_v3","active_configs":len(configs),"legacy_configs_rejected":2,
        "first_stage_sha256":identity["first_sha"],"first_stage_completed_epochs":checkpoint["completed_epochs"],
        "annotation_samples":counts,
        "cache_manifests_present":{s:(Path(configs[0]["cache_dir"])/s/"manifest.json").is_file()
                                   for s in ["train","val"]},
        "stage_b_cache_manifests_present":{s:(Path(configs[-1]["cache_dir"])/s/"manifest.json").is_file()
                                   for s in ["train","val"]}},indent=2))
if __name__=="__main__":main()
