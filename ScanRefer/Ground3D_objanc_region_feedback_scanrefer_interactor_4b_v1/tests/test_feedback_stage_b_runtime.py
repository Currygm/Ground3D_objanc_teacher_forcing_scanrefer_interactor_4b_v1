import copy
from pathlib import Path
import pytest
import torch
from ground3d_migration.runtime import capture_rng

def identity():
    return {"first_sha":"fixed","data":{"train":"same"},"base_config":{"source":"same"},
        "generation_protocol":"unconstrained","implementation":{"ground3d_migration/language.py":"same"}}

def test_a_to_b_warmstart_transfers_only_reader_and_preserves_decoder(tmp_path):
    from region_feedback.checkpoint import save_checkpoint,import_stage_a
    from region_feedback.model import FeedbackModel
    from test_feedback_v2_model import setup
    from test_feedback_stage_b_model import build_b
    sa,ra,ca=setup();ca["training"]={"seed":42}
    a=FeedbackModel(sa.answer_lm,ca,text_adapter=sa.text_adapter,decoder=sa.decoder)
    opt=torch.optim.AdamW(a.optimizer_parameter_groups());sch=torch.optim.lr_scheduler.StepLR(opt,1)
    with torch.no_grad():a.object_projection[-1].weight.add_(.5)
    path=tmp_path/"a.pt"
    save_checkpoint(a,opt,sch,path,step=1,epoch=1,config=ca,provenance=identity(),rng_states=[capture_rng()])
    sb,rb,cb,b=build_b()
    before={k:v.clone() for k,v in b.decoder.state_dict().items()}
    parent=import_stage_a(b,path,identity())
    assert parent["path"]==str(path.resolve())
    assert torch.equal(a.object_projection[-1].weight,b.object_projection[-1].weight)
    assert all(torch.equal(before[k],v) for k,v in b.decoder.state_dict().items())
    state=torch.load(path,weights_only=False);state["trainable_state"].pop(next(iter(state["trainable_state"])))
    torch.save(state,tmp_path/"bad.pt")
    snapshot=b.object_projection[-1].weight.clone()
    with pytest.raises(ValueError):import_stage_a(b,tmp_path/"bad.pt",identity())
    assert torch.equal(snapshot,b.object_projection[-1].weight)
    with pytest.raises(ValueError):import_stage_a(b,path,{**identity(),"first_sha":"wrong"})

def test_stage_b_checkpoint_resume_preserves_parent_and_all_decoder_parameters(tmp_path):
    from test_feedback_stage_b_model import build_b
    from region_feedback.checkpoint import save_checkpoint,load_checkpoint
    s,r,cfg,m=build_b();opt=torch.optim.AdamW(m.optimizer_parameter_groups())
    sch=torch.optim.lr_scheduler.StepLR(opt,1)
    m([r])["loss"].backward();opt.step();sch.step()
    parent={"path":"a.pt","sha256":"parent"}
    m._stage_a_parent=parent
    path=tmp_path/"b.pt";save_checkpoint(m,opt,sch,path,step=2,epoch=1,config=cfg,
                                       provenance=identity(),rng_states=[capture_rng()])
    expected={n:v.clone() for n,v in m.trainable_state_dict().items()}
    with torch.no_grad():m.decoder.query_content.weight.add_(2)
    assert load_checkpoint(m,path,cfg,identity(),optimizer=opt,scheduler=sch,restore_rng_state=True)==(2,1)
    assert m._stage_a_parent==parent
    assert all(torch.equal(expected[n],v) for n,v in m.trainable_state_dict().items())

def test_stage_b_config_and_candidate_change_contract():
    from region_feedback.config import load_feedback_config
    from region_feedback.runtime_v2 import validate_candidate_contract
    root=Path(__file__).resolve().parents[1]
    cfg=load_feedback_config(root/"configs/feedback_b_visual.yaml")
    assert cfg["feedback"]["stage"]=="b" and cfg["cache_mode"]=="full_scene_v4"
    assert cfg["optimizer"]["decoder_lr"]==1e-4 and cfg["feedback"]["answer_lm_weight"]==.1
    old=torch.zeros(50,4);new=torch.ones(50,4)
    main={"oracle_iou":.8};first={"oracle_iou":.2}
    validate_candidate_contract("b",new,old,main,first)
    with pytest.raises(RuntimeError):validate_candidate_contract("a",new,old,main,first)
