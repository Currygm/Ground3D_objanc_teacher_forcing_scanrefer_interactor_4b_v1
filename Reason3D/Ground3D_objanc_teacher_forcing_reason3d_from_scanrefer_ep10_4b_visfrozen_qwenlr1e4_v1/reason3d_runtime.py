"""Ground3D Reason3D adaptation initialized from all ScanRefer epoch10 task state."""
from pathlib import Path
import json,yaml,torch
from torch.utils.data import ConcatDataset
from spatiallm_compressed_visual_objanc_full_attention.config import ensure_within
from spatiallm_compressed_visual_objanc_full_attention.data.dataset import ReasonSegDataset
from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import filter_answer_entities
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel as SourceModel
from runtime_helpers import sha256,load_trainable_state,frozen_digest
PROJECT=Path(__file__).resolve().parent
SOURCE=PROJECT.parent/"Ground3D_objanc_teacher_forcing_scanrefer_4b_v1"
OBJECTIVE="ground3d_objanc_teacher_forcing_reason3d_from_scanrefer_ep10_4b_visfrozen_qwenlr1e4_v1"


class ObjAncSegmentationModel(SourceModel):
    objective=OBJECTIVE
    def train(self,mode=True):
        super().train(mode)
        self.answer_lm.visual.eval()
        if mode and getattr(self,"_training_phase","joint")=="answer_only":
            self.answer_lm.visual.eval();self.decoder.eval();self.text_adapter.eval();self.visual_adapter.eval()
        return self
    def architecture_signature(self):
        signature=super().architecture_signature()
        signature.update(objective=OBJECTIVE,superpoint_source="reason3d_scene_native_v1",
            answer_lm_weight=self.answer_lm_weight,answer_obj_ce={"weight":self.obj_ce_weight,"version":"obj_content_token_mean_v1"},
            answer_warmup={"epochs":3,"trainable":"language_lora_and_structured_tokens","lr_schedule":"continuous_13_epochs_joint_qwen10_visual_frozen"})
        signature["visual_training"]={"frozen":True,"buffers":"eval"}
        signature["joint_qwen_lora_lr"]=float(self.config["training"]["joint_qwen_lora_lr"])
        return signature
    def forward(self,batch,**kwargs):
        if self.training and getattr(self,"_training_phase","joint")=="answer_only":
            if kwargs.get("answer_mode","teacher")!="teacher":raise ValueError("Answer warmup requires teacher answers")
            from reason3d_training_schedule import answer_only_forward
            return answer_only_forward(self,batch)
        return super().forward(batch,**kwargs)


def load_segmentation_config(path):
    config=yaml.safe_load(Path(path).read_text())
    project=config["project"];root=Path(project["authorized_code_root"]).resolve(strict=True)
    if Path(project["artifact_root"]).resolve(strict=True)!=PROJECT:raise ValueError("Wrong new project root")
    ensure_within(PROJECT,root,must_exist=True)
    for key in ["output_dir","evaluation_dir","cache_dir"]:project[key]=str(ensure_within(project[key],PROJECT,must_exist=False))
    if config["initialization"]!={"mode":"official_ground3d_3d"}:raise ValueError("Original frozen base bootstrap required")
    expected=json.loads((PROJECT/"docs/source_checkpoint.json").read_text());init=config["finetune_initialization"]
    if (init["checkpoint"],init["sha256"],init["source_completed_epochs"],init["source_step"],init["reset_optimizer_scheduler_step_epoch"])!=(expected["source_path"],expected["sha256"],10,45720,True):raise ValueError("Expected pinned ScanRefer epoch10 initialization")
    ensure_within(init["checkpoint"],SOURCE,must_exist=True)
    for key in ["model_name_or_path","ground3d_weights_dir"]:ensure_within(config["model"][key],SOURCE,must_exist=True)
    t=config["training"]
    if (t["objective"],t["epochs"],t["answer_warmup_epochs"],t["joint_epochs"],t["answer_lm_weight"],t["obj_ce_weight"],t["save_every_epochs"],t["lr_schedule"])!=(OBJECTIVE,13,3,10,3.,3.,5,"continuous_13_epochs_joint_qwen10_visual_frozen"):raise ValueError("Expected Answer3 + joint10, CE3 + OBJ CE3, joint save5")
    rates=("qwen_lora_lr","structured_token_lr","visual_lr","adapter_lr","decoder_lr","joint_qwen_lora_lr")
    if not t.get("freeze_visual_encoder") or tuple(float(t[k]) for k in rates)!=(1e-5,1e-5,0.,1e-4,1e-4,1e-4):
        raise ValueError("Expected frozen vision, original rates and Joint Qwen LoRA 1e-4")
    if t["answer_mode"]!="teacher" or t["allow_generated_answers"]:raise ValueError("Teacher-only training required")
    for key in ["loss","matcher","ranking","decoder","point_encoder"]:
        if config[key]!=expected["architecture_signature"][key]:raise ValueError("Source architecture or loss differs: "+key)
    data=config["data"]
    if (data["expected_train_samples"],data["expected_val_samples"])!=(1339,1145):raise ValueError("Expected full Reason3D dataset")
    if len(data["train_sources"])!=1 or data["train_sources"][0]["name"]!="reason3d" or data["sampling_mode"]!="full_dataset_no_replacement":raise ValueError("Expected full Reason3D sampling")
    for value in [data["train_sources"][0]["annotations"],data["val_annotations"],data["scene_root"]]:ensure_within(value,root,must_exist=True)
    if data["superpoint"].get("source")!="scene" or data["superpoint"].get("enabled",True) or data["normal"].get("enabled",True):raise ValueError("Use native scene superpoints without normals")
    return config


def restore_task_state(model,state):
    destination=model.trainable_state_dict()
    if not isinstance(state,dict) or set(state)!=set(destination):raise ValueError("Full task state keys differ")
    for name,value in state.items():
        if value.shape!=destination[name].shape or value.dtype!=destination[name].dtype or not torch.isfinite(value).all():raise ValueError("Invalid source task tensor: "+name)
    load_trainable_state(model,state)


def load_source_initialization(model,path,config):
    expected=json.loads((PROJECT/"docs/source_checkpoint.json").read_text())
    if sha256(path)!=expected["sha256"]:raise ValueError("ScanRefer parent hash differs")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    for key in ["objective","format_version","training_stage","completed_epochs","step","architecture_signature","base_manifest","frozen_hash"]:
        if checkpoint.get(key)!=expected[key]:raise ValueError("Source metadata differs: "+key)
    if SourceModel.architecture_signature(model)!=checkpoint["architecture_signature"]:raise ValueError("Source structural signature differs")
    if model.answer_lm.source_manifests!=checkpoint["base_manifest"]:raise ValueError("Source base manifests differ")
    if model.answer_lm.tokenizer_snapshot()!=checkpoint["tokenizer_snapshot"]:raise ValueError("Source tokenizer differs")
    if frozen_digest(model)!=checkpoint["frozen_hash"]:raise ValueError("Source frozen weights differ")
    restore_task_state(model,checkpoint["trainable_state"])
    report={"source":str(path),"sha256":expected["sha256"],"state_tensors_loaded":len(checkpoint["trainable_state"]),"optimizer_scheduler_step_epoch_reset":True}
    print(json.dumps(report),flush=True)
    return report


def read_annotations(config, split):
    if split not in ("train", "val"):
        raise ValueError("Expected train or val split")
    data = config["data"]
    path = data["train_sources"][0]["annotations"] if split == "train" else data["val_annotations"]
    path = ensure_within(path, config["project"]["authorized_code_root"], must_exist=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["annotations"] if isinstance(payload, dict) else payload
    if len(rows) != int(data[f"expected_{split}_samples"]):
        raise ValueError(f"Unexpected {split} sample count: {len(rows)}")
    return path, rows


def build_segmentation_dataset(config, split):
    data, project = config["data"], config["project"]
    path, rows = read_annotations(config, split)
    datasets = []
    if any(row.get("type") not in ("scannet", "mp3d") for row in rows):
        raise ValueError("Unexpected Reason3D source type")
    for kind, field, fields in (
        ("scannet", "semantic_id", ["xyz", "rgb", "superpoint_id", "semantic_id", "instance_id"]),
        ("mp3d", "instance_id", ["xyz", "rgb", "superpoint_id", "instance_id"]),
    ):
        subset = [row for row in rows if row["type"] == kind]
        if not subset:
            raise ValueError(f"Missing Reason3D source: {kind}")
        for row in subset:
            if row["target_label_field"] != field or row["scene_tuple_fields"] != fields:
                raise ValueError(f"Wrong scene/target schema: {row['scene_id']}")
            if row.get("grounding_span_source") != "teacher_answer":
                raise ValueError("Reason3D spans must index the teacher answer")
        datasets.append(ReasonSegDataset(
            path, scene_root=data["scene_root"], authorized_root=project["authorized_code_root"],
            split=split, tuple_fields=fields, target_label_field=field,
            annotation_text_field="description", scene_file_suffix="reason", dataset_name="reason3d_"+kind,
            num_bins=int(data["num_bins"]), world_extent_m=float(data["world_extent_m"]),
            loc_margin_m=float(data.get("loc_margin_m", .3)),
            random_voxel_sample=bool(data.get("train_random_voxel_sample", False) and split == "train"),
            seed=int(data.get("seed", 42)), segmentation_labels=True,
            superpoint_config=data["superpoint"], normal_config=data["normal"], require_teacher_answer=True,
            annotation_payload=[filter_answer_entities(row) for row in subset],
            annotation_authorized_roots=[project["authorized_code_root"],project["source_data_root"]],
        ))
    return ConcatDataset(datasets)



