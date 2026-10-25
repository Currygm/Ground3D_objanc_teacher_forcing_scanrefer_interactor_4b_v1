import copy
import torch
import pytest
from feedback_fixtures import source,config,batch
from region_feedback.regions import pool_regions
from region_feedback.checkpoint import save_checkpoint,load_checkpoint,import_stage_a
from region_feedback.proposals import FrozenProposalStage
from region_feedback.model import FeedbackModel
from test_feedback_runtime import Toy


def test_resume_keeps_stage_a_parent_provenance(tmp_path):
    m=Toy();opt=torch.optim.AdamW(m.parameters());sch=torch.optim.lr_scheduler.StepLR(opt,1)
    parent={"sha256":"stage-a","path":"parent"}
    path=tmp_path/"b.pt"
    from ground3d_migration.runtime import capture_rng
    save_checkpoint(m,opt,sch,path,step=1,epoch=1,config={},provenance={},rng_states=[capture_rng()],parent=parent)
    load_checkpoint(m,path,{}, {})
    assert getattr(m,"_stage_a_parent",None)==parent




def test_pool_remains_fp32_under_cpu_amp():
    f=torch.tensor([[1.1,2.3],[3.2,4.1]])
    xyz=torch.zeros(2,3);m=torch.tensor([[2.1,1.4]])
    expected=pool_regions(f,xyz,m,torch.arange(2))
    with torch.autocast("cpu",dtype=torch.bfloat16):actual=pool_regions(f,xyz,m,torch.arange(2))
    assert actual.features.dtype==torch.float32
    assert torch.allclose(expected.features,actual.features,atol=1e-6)
