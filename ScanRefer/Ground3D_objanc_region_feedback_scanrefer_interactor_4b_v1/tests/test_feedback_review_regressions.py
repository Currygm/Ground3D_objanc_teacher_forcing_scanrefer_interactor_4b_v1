import copy
import hashlib
import inspect
import json
from pathlib import Path
import pytest
import torch
from test_feedback_runtime import Toy
from region_feedback import factory
from region_feedback.checkpoint import save_checkpoint,load_checkpoint
from ground3d_migration.runtime import capture_rng

def identity_config(tmp_path,monkeypatch):
    train=tmp_path/"train.json";train.write_text("[]")
    val=tmp_path/"val.json";val.write_text("[]")
    first=tmp_path/"first.pt";first.write_bytes(b"fixed source")
    base=tmp_path/"base.json"
    content={"data":{"train_sources":[{"name":"scanrefer","annotations":str(train)}],
             "val_annotations":str(val),"annotation_text_field":"description","scene_root":"scene_a"}}
    base.write_text(json.dumps(content))
    monkeypatch.setattr(factory,"load_segmentation_config",lambda path:json.loads(Path(path).read_text()))
    config={"base_config":str(base),"first_stage":{"checkpoint":str(first),
        "sha256":hashlib.sha256(first.read_bytes()).hexdigest(),"generation_protocol":"constrained"}}
    return config,base,content

def test_provenance_changes_when_resolved_preprocessing_changes(tmp_path,monkeypatch):
    config,base,content=identity_config(tmp_path,monkeypatch)
    before=factory.provenance(config)
    content["data"]["annotation_text_field"]="original_description"
    content["data"]["scene_root"]="scene_b"
    base.write_text(json.dumps(content))
    after=factory.provenance(config)
    assert before!=after,"changed preprocessing must invalidate resume/handoff provenance"

def test_provenance_covers_inherited_data_and_runtime_code(tmp_path,monkeypatch):
    config,_,_=identity_config(tmp_path,monkeypatch)
    paths=factory.provenance(config)["implementation"]
    assert "spatiallm_compressed_visual_objanc_full_attention/data/dataset.py" in paths
    assert "spatiallm_compressed_visual_objanc_full_attention/config.py" in paths
    assert "runtime_helpers.py" in paths
    assert "train_feedback.py" in paths

def test_evaluation_can_read_custom_training_output_but_not_changed_science(tmp_path):
    assert "evaluation" in inspect.signature(load_checkpoint).parameters,"evaluation artifact-path handling missing"
    model=Toy();optimizer=torch.optim.AdamW(model.parameters())
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
    trained={"project_root":str(tmp_path),"output_dir":str(tmp_path/"custom"),"epochs":5}
    original={**trained,"output_dir":str(tmp_path/"default")}
    path=tmp_path/"saved.pt"
    save_checkpoint(model,optimizer,scheduler,path,step=3,epoch=1,config=trained,
                    provenance={},rng_states=[capture_rng()])
    with pytest.raises(ValueError):load_checkpoint(model,path,original,{})
    assert load_checkpoint(model,path,original,{},evaluation=True)==(3,1)
    with pytest.raises(ValueError):load_checkpoint(model,path,{**original,"epochs":6},{},evaluation=True)
