"""CPU-only audit of real data, pinned source weights and phase boundaries."""
import os,sys,json,math,hashlib
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1"
PROJECT=Path(__file__).resolve().parents[1];os.chdir(PROJECT);sys.path.insert(0,str(PROJECT))
import torch
from reason3d_runtime import ObjAncSegmentationModel,load_segmentation_config,build_segmentation_dataset,load_source_initialization
from reason3d_training_schedule import set_training_epoch
from reason3d_eval_subsets import source_indices
from reason3d_stage_checkpoints import should_save_epoch

config=load_segmentation_config(PROJECT/"configs/reason3d_answer_lm3_objce3_warmup3_joint15.yaml")
report={"cuda_initialized":False,"datasets":{}}
for split in ["train","val"]:
    dataset=build_segmentation_dataset(config,split);groups=source_indices(dataset)
    report["datasets"][split]={"samples":len(dataset),"sources":{k:len(v) for k,v in groups.items()}}
    for source,indices in groups.items():
        leaf=dataset.datasets[0 if source=="scannet" else 1]
        for row in leaf.annotations:assert leaf._resolve_scene_path(row).is_file()
        if split=="val":
            sample=dataset[indices[0]]
            assert sample["gt_point_masks"][0].any()
            assert len(sample["point_cloud"])==sample["point_to_superpoint"].numel()
            assert sample["gt_superpoint_masks"].shape[1]==len(sample["superpoint_coord"])
            report["datasets"][split][source+"_probe"]={"scene":sample["scene_id"],"raw_points":len(sample["point_cloud"]),"superpoints":len(sample["superpoint_coord"]),"target_points":int(sample["gt_point_masks"][0].sum())}
print(json.dumps({"event":"data_audit","report":report["datasets"]}),flush=True)
model=ObjAncSegmentationModel(config)
source=load_source_initialization(model,Path(config["finetune_initialization"]["checkpoint"]),config)
checkpoint=torch.load(config["finetune_initialization"]["checkpoint"],map_location="cpu",weights_only=False)
for name,value in model.trainable_state_dict().items():
    torch.testing.assert_close(value,checkpoint["trainable_state"][name],rtol=0,atol=0)
report["source_handoff"]=source;report["all_task_tensors_equal"]=True
before=set(model.trainable_state_dict());set_training_epoch(model,0)
report["answer_trainable_tensors"]=sum(p.requires_grad for p in model.parameters())
assert not model.answer_lm.visual.training
for name,parameter in model.named_parameters():
    if parameter.requires_grad:assert name.startswith("answer_lm.structured.") or "lora_" in name
assert set(model.trainable_state_dict())==before
set_training_epoch(model,3)
report["joint_trainable_tensors"]=sum(p.requires_grad for p in model.parameters())
assert {n for n,p in model.named_parameters() if p.requires_grad}==model._checkpoint_names
assert set(model.trainable_state_dict())==before
for name,value in model.trainable_state_dict().items():torch.testing.assert_close(value,checkpoint["trainable_state"][name],rtol=0,atol=0)
report["optimizer_groups"]=[{"name":g["name"],"tensors":len(g["params"]),"lr":g["lr"]} for g in model.optimizer_parameter_groups()]
report["save_global_epochs"]=[e for e in range(1,14) if should_save_epoch(e,3,13,5)]
report["steps_per_epoch_four_gpus"]=math.ceil(math.ceil(1339/4)/2);report["final_steps"]=13*report["steps_per_epoch_four_gpus"]
assert not torch.cuda.is_initialized()
report["cuda_initialized"]=torch.cuda.is_initialized()
(PROJECT/"docs/cpu_handoff.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
