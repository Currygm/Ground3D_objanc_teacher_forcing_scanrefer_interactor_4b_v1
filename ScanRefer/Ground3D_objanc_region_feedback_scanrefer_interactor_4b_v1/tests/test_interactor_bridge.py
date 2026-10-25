from pathlib import Path
import importlib.util
import torch
import pytest

def bridge():
    assert importlib.util.find_spec("ground3d_migration.interactor") is not None, "Interactor bridge missing"
    from ground3d_migration.interactor import GroundInteractor
    return GroundInteractor

def test_query_only_real_qformer_padding_full_memory_and_gradient():
    cls=bridge()
    torch.manual_seed(4)
    m=cls({"language_hidden_size":2560,"num_queries":32},load_embeddings=False).eval()
    a=torch.randn(3,256,requires_grad=True); b=torch.randn(7,256,requires_grad=True)
    events=[]
    h=m.qformer.register_forward_pre_hook(lambda module,args,kwargs:events.append(kwargs),with_kwargs=True)
    together=m([a,b]);h.remove()
    assert len(together)==2 and together[0].shape==(32,2560)
    assert events[0]["input_ids"] is None
    assert events[0]["encoder_hidden_states"].shape==(2,7,256)
    assert events[0]["encoder_attention_mask"].tolist()==[[True]*3+[False]*4,[True]*7]
    assert torch.allclose(together[0],m([a])[0],atol=2e-5)
    assert not torch.allclose(m([a+1])[0],together[0])
    sum(x.square().mean() for x in together).backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)
    assert all(p.grad is not None for n,p in m.named_parameters() if p.requires_grad)
    assert .015<float(m.latent_query.weight.std())<.025
    assert m.qformer.config.num_hidden_layers==6
    assert m.qformer.config.num_attention_heads==12

@pytest.mark.parametrize("s",[1999,2000,2001])
def test_no_sampling_and_independent_full_decoder(s):
    cls=bridge()
    from ground3d_migration.visual import GroundPointFeatures,GroundVisualTokens
    from ground3d_migration.language import GroundObjAncLM
    from torch import nn
    lm=GroundObjAncLM.__new__(GroundObjAncLM);nn.Module.__init__(lm)
    full=torch.arange(s).float()[:,None].expand(s,256)
    xyz=torch.randn(s,3)
    class Encoder(nn.Module):
        def forward(self,*args,**kwargs):
            assert "cap" not in kwargs
            return GroundPointFeatures(GroundVisualTokens(full,xyz),full,xyz)
    class Interactor(nn.Module):
        def forward(self,memories):
            assert memories[0] is full
            return [full[:32,:1].expand(32,2560)]
    lm.visual=Encoder();lm.interactor=Interactor()
    rows=lm.encode_batch_points({"point_cloud":[torch.zeros(1,6)],"point_to_superpoint":[torch.zeros(1,dtype=torch.long)],"superpoint_coord":[xyz]})
    assert rows[0].superpoint_hidden is full and rows[0].superpoint_xyz is xyz
    assert rows[0].compressed.hidden.shape==(32,2560)
    assert rows[0].compressed.xyz is None
