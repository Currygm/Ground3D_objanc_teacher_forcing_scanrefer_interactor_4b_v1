import copy
import torch
from torch.nn import functional as F
from feedback_fixtures import source,batch
from region_feedback.proposals import FrozenProposalStage

def test_cached_scoring_keys_reproduce_original_obj_token_scores():
    s=source();p=FrozenProposalStage(s,"constrained").predict(batch())[0]
    assert p.scoring_keys.shape==(50,16)
    logits=p.scoring_keys@F.normalize(p.updated_text_features,dim=-1).T/s.decoder.text_temperature
    scores=logits.float().sigmoid()@p.phrase_spans.float().T/p.phrase_spans.sum(-1)[None]
    assert torch.allclose(scores,p.entity_scores,atol=1e-6)

def test_stage_a_uses_frozen_original_text_modules_without_classifier():
    from region_feedback.model import FeedbackModel
    from region_feedback.cache import make_record
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    r=make_record(p,b,0,s.answer_lm,index=0);r["masks"][0]=torch.tensor([8.,8.,-8.,-8.])
    cfg={"feedback":{"protocol":"object_evidence_shared_qwen_textscore_v3","stage":"a","method":"qwen_feedback",
        "evidence_mode":"visual","input_dim":8,"language_dim":64,"num_queries":50,"hidden_dim":16,
        "ranking_temperature":.1,"ranking_weight":.5},
        "optimizer":{"new_lr":1e-3,"qwen_lora_lr":1e-3,"weight_decay":0.}}
    m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder).train()
    assert not hasattr(m,"classifier")
    assert m.text_adapter is s.text_adapter and m.text_connector is s.decoder.text_connector
    frozen={n:v.clone() for n,v in m.named_parameters() if not v.requires_grad}
    result=m([r]);result["loss"].backward()
    assert m.object_projection[-1].weight.grad.abs().sum()>0
    assert sum(p.grad.abs().sum() for n,p in m.language.named_parameters() if p.grad is not None and "lora_" in n)>0
    opt=torch.optim.AdamW(m.optimizer_parameter_groups());opt.step()
    assert all(torch.equal(frozen[n],p) for n,p in m.named_parameters() if n in frozen)
    assert all("classifier" not in n and "text_adapter" not in n and "text_connector" not in n
               for n,p in m.named_parameters() if p.requires_grad)
    assert torch.equal(result["masks"][0],r["masks"])

def test_obj_score_is_mean_sigmoid_not_sigmoid_of_mean():
    from region_feedback.text_scores import fixed_candidate_text_scores
    query=torch.tensor([[1.,0.],[0.,1.]])
    tokens=torch.tensor([[1.,0.],[-1.,1.],[0.,1.]])
    spans=torch.tensor([[True,True,False]])
    actual=fixed_candidate_text_scores(query,tokens,spans,temperature=.7)
    logits=query@F.normalize(tokens,dim=-1).T/.7
    expected=logits.sigmoid()[:,:2].mean(-1)
    assert torch.allclose(actual[:,0],expected)
    assert not torch.allclose(actual[:,0],logits[:,:2].mean(-1).sigmoid())
