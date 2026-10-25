import copy
import inspect
from pathlib import Path
import pytest
import torch
from region_feedback.checkpoint import save_checkpoint,load_checkpoint
from ground3d_migration.runtime import capture_rng
from test_feedback_runtime import Toy

def test_checkpoint_snapshot_reads_shared_lm_and_rejects_legacy(tmp_path):
    from test_feedback_v2_model import setup
    from region_feedback.model import FeedbackModel
    s,r,cfg=setup();m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder)
    opt=torch.optim.AdamW(m.optimizer_parameter_groups());sch=torch.optim.lr_scheduler.StepLR(opt,1)
    path=tmp_path/"v2.pt"
    save_checkpoint(m,opt,sch,path,step=1,epoch=1,config=cfg,provenance={},rng_states=[capture_rng()])
    state=torch.load(path,weights_only=False)
    assert state["format_version"]==3
    assert state["tokenizer_snapshot"]==m.language.tokenizer_snapshot()
    before={k:v.clone() for k,v in m.trainable_state_dict().items()}
    load_checkpoint(m,path,cfg,{})
    assert all(torch.equal(before[k],v) for k,v in m.trainable_state_dict().items())
    state["format_version"]=1;torch.save(state,path)
    with pytest.raises(ValueError,match="Legacy"):load_checkpoint(m,path,cfg,{})

def test_v2_checked_configs_and_legacy_stage_b_rejected():
    from region_feedback.config import load_feedback_config
    root=Path(__file__).resolve().parents[1]
    values=[load_feedback_config(root/"configs"/("feedback_a_"+x+".yaml")) for x in ["empty","text","visual"]]
    assert all(c["schema_version"]==3 and "cache_dir" in c for c in values)
    assert all(c["feedback"]["protocol"]=="object_evidence_shared_qwen_textscore_v3" for c in values)
    assert len({c["output_dir"] for c in values})==3
    with pytest.raises(ValueError,match="Stage B"):load_feedback_config(root/"configs/feedback_b_visual_geometry.yaml")
    with pytest.raises(ValueError):load_feedback_config(root/"configs/feedback_a_visual_geometry.yaml")

def test_cache_identity_binds_original_source_not_feedback_training(monkeypatch,tmp_path):
    from region_feedback.factory import cache_identity
    identity={"first_sha":"a","data":{"train":"b"},"base_config":{"data":{"seed":42}},
       "implementation":{"region_feedback/objects.py":"c","region_feedback/model.py":"d",
                         "spatiallm_compressed_visual_objanc_full_attention/data/dataset.py":"e"}}
    a=cache_identity({},identity)
    identity["implementation"]["region_feedback/model.py"]="changed"
    assert a==cache_identity({},identity)
    identity["implementation"]["region_feedback/objects.py"]="changed"
    assert a!=cache_identity({},identity)
