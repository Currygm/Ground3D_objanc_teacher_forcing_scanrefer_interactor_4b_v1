import copy
import torch
from feedback_fixtures import source,batch
from region_feedback.cache import make_record
from region_feedback.proposals import FrozenProposalStage

def setup_b():
    from test_feedback_v2_model import setup
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    r=make_record(p,b,0,s.answer_lm,index=0,include_full_scene=True)
    _,_,cfg=setup()
    cfg["feedback"].update(stage="b",answer_lm_weight=.1,max_new_tokens=32)
    cfg["optimizer"].update(decoder_lr=1e-3,adapter_lr=1e-3)
    cfg["training"]={"seed":42}
    return s,r,cfg

def build_b():
    from region_feedback.stage_b import StageBFeedbackModel
    s,r,cfg=setup_b()
    m=StageBFeedbackModel(s.answer_lm,cfg,decoder=s.decoder,text_adapter=s.text_adapter,
                         visual_adapter=s.visual_adapter,objective_config=s.config)
    return s,r,cfg,m

def test_stage_b_full_decoder_and_all_losses_train_one_lm_without_classifier():
    s,r,cfg,m=build_b();m.train()
    assert m.language is s.answer_lm and m.decoder is s.decoder
    assert not hasattr(m,"student_lm") and not hasattr(m,"classifier")
    before=r["masks"].clone()
    result=m([r]);result["loss"].backward()
    assert result["masks"][0].requires_grad and not torch.equal(result["masks"][0],before)
    assert torch.equal(result["first_masks"][0],before) and torch.equal(r["masks"],before)
    for module in [m.decoder,m.text_adapter,m.visual_adapter,m.object_projection]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert sum(p.grad.abs().sum() for n,p in m.language.named_parameters() if "lora_" in n and p.grad is not None)>0
    assert all(p.grad is None for n,p in m.language.named_parameters() if not p.requires_grad)
    assert {"span","mask_bce","dice","auxiliary","ranking","answer_lm","total"}<=set(result["losses"])
    assert all(torch.isfinite(v) for v in result["losses"].values())
    groups=m.optimizer_parameter_groups()
    assert {g["name"] for g in groups}=={"qwen_lora","evidence","adapters","decoder"}
    assert len({id(p) for g in groups for p in g["params"]})==sum(len(g["params"]) for g in groups)

def test_stage_b_generated_prediction_ignores_all_supervision(monkeypatch):
    import region_feedback.stage_b as module
    s,r,cfg,m=build_b();m.eval()
    monkeypatch.setattr(module,"generate_object_answer",lambda *a,**k:{"answer":"<obj>chair</obj>","format_failure":False})
    observed={k:v for k,v in r.items() if k not in {"truth","supervision"}}
    with torch.no_grad():a=m.predict([observed])
    changed=copy.deepcopy(r);changed["truth"]=~changed["truth"]
    changed["supervision"]["teacher_answer"]="LEAK";changed["supervision"]["gt_superpoint_masks"].zero_()
    with torch.no_grad():b=m.predict([changed])
    assert torch.equal(a["masks"][0],b["masks"][0]) and torch.equal(a["scores"][0],b["scores"][0])
    assert a["answers"]==["<obj>chair</obj>"] and not a["failed"][0]
    assert a["masks"][0].shape==(50,4)

def test_stage_b_second_failure_uses_zero_obj_and_first_failure_can_recover(monkeypatch):
    import region_feedback.stage_b as module
    s,r,cfg,m=build_b();m.eval();r["failed"]=True
    monkeypatch.setattr(module,"generate_object_answer",lambda *a,**k:{"answer":"<obj>chair</obj>","format_failure":False})
    with torch.no_grad():good=m.predict([r])
    assert good["first_failed"]==[True] and good["failed"]==[False]
    monkeypatch.setattr(module,"generate_object_answer",lambda *a,**k:{"answer":"bad <|point|>","format_failure":True})
    with torch.no_grad():bad=m.predict([r])
    assert bad["failed"]==[True] and bad["answers"]==["bad <|point|>"]
    assert torch.isfinite(bad["scores"][0]).all() and bad["masks"][0].shape==(50,4)
