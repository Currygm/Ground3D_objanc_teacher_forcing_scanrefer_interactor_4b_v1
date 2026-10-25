import torch
from region_feedback.regions import pool_regions

def test_pool_uses_raw_point_weights_and_keeps_empty_region_finite():
    f=torch.tensor([[1.,0.],[3.,0.],[0.,5.]],requires_grad=True)
    xyz=torch.tensor([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]])
    masks=torch.tensor([[8.,8.,-8.],[-8.,-8.,-8.]],requires_grad=True)
    e=pool_regions(f,xyz,masks,torch.tensor([0,1,1,1,2]))
    assert torch.allclose(e.features[0],torch.tensor([2.5,0.]),atol=1e-6)
    assert torch.equal(e.valid_mask,torch.tensor([True,False]))
    assert torch.equal(e.features[1],torch.zeros(2))
    assert not e.features.requires_grad and masks.grad is None and f.grad is None
