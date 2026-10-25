import copy
import torch
import pytest
from feedback_fixtures import source,batch
from region_feedback.proposals import FrozenProposalStage
from region_feedback.cache import make_record
from region_feedback.model import FeedbackModel

def setup(mode="visual",method="qwen_feedback"):
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    r=make_record(p,b,0,s.answer_lm,index=0)
    r["masks"][0]=torch.tensor([8.,8.,-8.,-8.])
    cfg={"feedback":{"protocol":"object_evidence_shared_qwen_textscore_v3","stage":"a","method":method,
        "evidence_mode":mode,"input_dim":8,"language_dim":64,"num_queries":50,"hidden_dim":16,
        "ranking_temperature":.1,"ranking_weight":.5},
        "optimizer":{"new_lr":1e-3,"qwen_lora_lr":1e-3,"weight_decay":0.}}
    return s,r,cfg

def test_single_lm_only_lora_and_new_modules_train_and_cache_unchanged():
    s,r,cfg=setup();lm=s.answer_lm
    m=FeedbackModel(lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder).train()
    assert m.language is lm and not hasattr(m,"student_lm")
    before={n:p.detach().clone() for n,p in lm.named_parameters() if not p.requires_grad}
    old=r["masks"].clone()
    result=m([r]);result["loss"].backward()
    assert m.object_projection[-1].weight.grad.abs().sum()>0
    assert not hasattr(m,"classifier")
    assert sum(p.grad.abs().sum() for n,p in lm.named_parameters() if "lora_" in n and p.grad is not None)>0
    assert all(p.grad is None for n,p in lm.named_parameters() if not p.requires_grad)
    opt=torch.optim.AdamW(m.optimizer_parameter_groups());opt.step()
    assert torch.equal(old,r["masks"])
    assert all(torch.equal(before[n],p) for n,p in lm.named_parameters() if n in before)
    assert all("lora_" in n and "visual" not in n for n,p in lm.named_parameters() if p.requires_grad)

def test_english_prompt_entity_slots_and_same_control_lengths():
    from region_feedback.object_prompt import render_object_prompt,SYSTEM_PROMPT
    s,r,cfg=setup()
    assert "first-pass" in SYSTEM_PROMPT and "may be incorrect" in SYSTEM_PROMPT
    results=[]
    for mode in ["empty","text","visual"]:
        s,r,cfg=setup(mode)
        m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder).eval()
        embeds,info=render_object_prompt(m.language,r,m.object_projection,mode)
        assert info["visual_slots"]==3
        assert "Predicted target: chair" in info["user_text"]
        assert "Predicted anchor 1: table" in info["user_text"]
        results.append(len(embeds))
        observed={k:v for k,v in r.items() if k!="truth"}
        x=m.predict([observed])
        assert x["scores"][0].shape==(50,)
        assert torch.equal(x["masks"][0],r["masks"])
        changed=copy.deepcopy(r);changed["truth"]=~r["truth"]
        assert torch.equal(x["scores"][0],m.predict([changed])["scores"][0])
    assert len(set(results))==1

def test_no_objects_and_empty_masks_are_finite_and_stage_b_rejected():
    from region_feedback.object_prompt import render_object_prompt
    s,r,cfg=setup();r["failed"]=True
    e=r["evidence"];e["features"]=e["features"][:0];e["roles"]=();e["phrases"]=()
    e["valid_mask"]=e["valid_mask"][:0];e["source_query_ids"]=e["source_query_ids"][:0]
    m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder).eval()
    embeds,info=render_object_prompt(m.language,r,m.object_projection,"visual")
    assert info["visual_slots"]==0 and "could not be parsed" in info["user_text"]
    assert torch.isfinite(m.predict([r])["scores"][0]).all()
    cfg["feedback"]["stage"]="b"
    with pytest.raises(ValueError,match="Stage B"):FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder)

def test_selector_does_not_run_qwen_second_pass():
    s,r,cfg=setup(method="selector")
    m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder)
    def forbidden(*a,**k):raise AssertionError("Selector must not execute Qwen")
    m.language._text_forward=forbidden
    out=m([r]);out["loss"].backward()
    assert all(not p.requires_grad for p in m.language.parameters())
    assert any(p.grad is not None for p in m.selector.parameters())
