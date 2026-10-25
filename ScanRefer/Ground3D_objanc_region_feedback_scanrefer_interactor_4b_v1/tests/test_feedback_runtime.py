import copy
import importlib
import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn
from feedback_fixtures import source,batch,config

def api(name):
    assert importlib.util.find_spec("region_feedback."+name) is not None,name+" implementation missing"
    return importlib.import_module("region_feedback."+name)

def test_metrics_accept_equal_quality_query_and_preserve_full_denominator():
    m=api("metrics")
    masks=torch.tensor([[9.,9.,-9.],[9.,9.,-9.],[-9.,-9.,9.]])
    result=m.candidate_metrics(torch.tensor([0.,2.,1.]),masks,torch.arange(3),torch.tensor([1,1,0]))
    assert result["oracle_iou"]==1 and result["selected_iou"]==1
    assert result["best_id_hit"]==0 and result["best_tie_hit"]==1
    records=[{"main":result,"strict":m.zero_metrics(result),"first":result,"failed":True},
             {"main":result,"strict":result,"first":result,"failed":False}]
    summary=m.summarize(records)
    assert summary["samples"]==2 and summary["strict"]["selected_iou"]==.5
    assert summary["main"]["selected_iou"]==1 and summary["failure_rate"]==.5

class Toy(nn.Module):
    objective="toy-feedback"
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.tensor([1.,2.]))
    def trainable_state_dict(self):return {"weight":self.weight.detach().cpu().clone()}
    def architecture_signature(self):return {"stage":"a"}

def test_checkpoint_restores_trainable_optimizer_scheduler_and_rng(tmp_path):
    m=api("checkpoint");model=Toy()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.9)
    model.weight.square().sum().backward();optimizer.step();scheduler.step();optimizer.zero_grad()
    expected=model.weight.detach().clone();cfg={"stage":"a"};provenance={"first_sha":"fixed","data":"fixed"}
    torch.manual_seed(8)
    p=tmp_path/"state.pt"
    from ground3d_migration.runtime import capture_rng
    m.save_checkpoint(model,optimizer,scheduler,p,step=3,epoch=1,config=cfg,provenance=provenance,rng_states=[capture_rng()])
    expected_rng=torch.rand(4)
    with torch.no_grad():model.weight.add_(10)
    torch.manual_seed(99)
    assert m.load_checkpoint(model,p,cfg,provenance,optimizer=optimizer,scheduler=scheduler,restore_rng_state=True)==(3,1)
    assert torch.equal(expected,model.weight) and torch.equal(expected_rng,torch.rand(4))
    old=model.weight.detach().clone()
    with pytest.raises(ValueError):m.load_checkpoint(model,p,cfg,{"first_sha":"changed"})
    assert torch.equal(old,model.weight)
