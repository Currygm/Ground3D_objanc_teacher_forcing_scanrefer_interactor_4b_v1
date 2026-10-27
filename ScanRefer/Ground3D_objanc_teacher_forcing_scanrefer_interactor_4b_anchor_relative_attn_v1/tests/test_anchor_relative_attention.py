import copy
import math
import torch
import pytest
from torch import nn
from spatiallm_compressed_visual_objanc_full_attention.model.anchor_relative_attention import (
    relative_inputs, relative_cross_attention, project, geometry_delta, shuffled_centres)


def fixture(dtype=torch.float32):
    torch.manual_seed(12)
    attn=nn.MultiheadAttention(8,2,batch_first=True,dropout=0).to(dtype)
    pe=nn.Sequential(nn.Linear(3,8),nn.LayerNorm(8),nn.ReLU(),nn.Linear(8,8)).to(dtype)
    q=torch.randn(4,8,dtype=dtype,requires_grad=True)
    text_logits=torch.randn(4,6,dtype=dtype,requires_grad=True)
    masks=torch.randn(4,7,dtype=dtype,requires_grad=True)
    xyz=torch.randn(7,3,dtype=dtype)
    counts=torch.tensor([1,2,3,4,1,9,2],dtype=dtype)
    spans=torch.tensor([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]],dtype=torch.bool)
    return attn,pe,q,text_logits,masks,xyz,counts,spans


def test_weights_and_point_weighted_centres():
    a,pe,q,t,m,x,n,s=fixture()
    u,c,r,mu,valid=relative_inputs(a,q,t,m,x,n,s,['OBJ','ANC','ANC'])
    assert c.shape==(4,4) and r.shape==(4,) and valid.all()
    expected=(m.detach().sigmoid()*n)@x/(m.detach().sigmoid()*n).sum(-1,keepdim=True)
    torch.testing.assert_close(mu,expected)
    assert (c.sum(-1)<=1+1e-6).all()  # null reference mass is retained
    assert (c>0).all()  # all candidate hypotheses remain


@pytest.mark.parametrize('sem,geo',[(0.,0.),(.3,0.),(0.,-.2),(.3,-.2)])
def test_chunked_matches_explicit_query_specific_key(sem,geo):
    a,pe,q,t,m,x,n,s=fixture()
    u,c,r,mu,valid=relative_inputs(a,q,t,m,x,n,s,['OBJ','ANC','ANC'])
    qp=torch.randn_like(q);v=torch.randn(7,8);vp=pe(x)
    blocked=torch.zeros(4,7,dtype=torch.bool);blocked[0,-2:]=True
    got=relative_cross_attention(a,q+qp,v,vp,blocked,u,c,r,mu,x,pe,torch.tensor(sem),torch.tensor(geo),2,3)
    rows=[]
    for j in range(4):
        d=sum(c[j,k]*(pe(x-mu[k])-pe(x)) for k in range(4))
        key=v+vp+geo*r[j]*d
        out,_=a((q+qp+sem*r[:,None]*u)[j:j+1][None],key[None],v[None],attn_mask=blocked[j:j+1],need_weights=False)
        rows.append(out[0,0])
    torch.testing.assert_close(got,torch.stack(rows),atol=1e-6,rtol=1e-5)


def test_gradients_and_checkpoint_equivalence():
    a,pe,q,t,m,x,n,s=fixture()
    u,c,r,mu,_=relative_inputs(a,q,t,m,x,n,s,['OBJ','ANC','ANC'])
    assert not mu.requires_grad
    v=torch.randn(7,8);vp=pe(x);raw=torch.tensor(.2,requires_grad=True)
    out=relative_cross_attention(a,q,v,vp,torch.zeros(4,7,dtype=torch.bool),u,c,r,mu,x,pe,raw.tanh(),raw.tanh(),2,3)
    out.square().sum().backward()
    assert m.grad is None
    assert q.grad.abs().sum()>0 and t.grad.abs().sum()>0 and raw.grad.abs()>0
    assert pe[0].weight.grad.abs().sum()>0


def test_zero_coeff_has_startup_gradient():
    a,pe,q,t,m,x,n,s=fixture()
    u,c,r,mu,_=relative_inputs(a,q,t,m,x,n,s,['OBJ','ANC','ANC'])
    raw=torch.tensor(0.,requires_grad=True);v=torch.randn(7,8)
    out=relative_cross_attention(a,q,v,pe(x),torch.zeros(4,7,dtype=torch.bool),u,c,r,mu,x,pe,raw.tanh(),raw.tanh(),2,3)
    out.square().sum().backward();assert raw.grad.abs()>0


def test_no_anchor_and_empty_candidates():
    a,pe,q,t,m,x,n,s=fixture()
    assert relative_inputs(a,q,t,m,x,n,s[:1],['OBJ']) is None
    assert relative_inputs(a,q,t,torch.full_like(m,-100),x,n,s,['OBJ','ANC','ANC']) is None
    with pytest.raises(FloatingPointError):
        relative_inputs(a,q,t,m*float('nan'),x,n,s,['OBJ','ANC','ANC'])


def test_shuffle_stable_valid_only():
    mu=torch.arange(15).reshape(5,3).float();valid=torch.tensor([1,0,1,1,0],dtype=torch.bool)
    a=shuffled_centres(mu,valid,'sample:42');b=shuffled_centres(mu,valid,'sample:42')
    assert torch.equal(a,b) and torch.equal(a[~valid],mu[~valid])
    assert (a[valid]!=mu[valid]).any(-1).all()


def test_schedule_endpoints_and_resume():
    from anchor_runtime import PhaseScheduler
    for phase,expected in [('answer',(0.,1e-4)),('joint',(1e-4,1e-5))]:
        p=nn.Parameter(torch.tensor(1.));o=torch.optim.AdamW([{'params':[p],'name':'qwen_lora','lr':1e-4}])
        sched=PhaseScheduler(o,7,.1,phase)
        got=[]
        for _ in range(7):
            got.append(o.param_groups[0]['lr']);o.step();sched.step()
        assert got[0]==pytest.approx(expected[0]) and got[-1]==pytest.approx(expected[1])
        state=sched.state_dict();sched2=PhaseScheduler(o,7,.1,phase);sched2.load_state_dict(state)
        assert sched2.state_dict()==state
