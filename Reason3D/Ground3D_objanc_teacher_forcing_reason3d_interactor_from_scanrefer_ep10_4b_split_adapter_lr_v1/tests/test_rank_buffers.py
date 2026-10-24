import torch
from torch import nn
from reason3d_buffers import capture_buffers,restore_buffers
def test_rank_specific_bn_restored_without_touching_parameters():
    m=nn.Module();m.answer_lm=nn.Module();m.answer_lm.visual=nn.BatchNorm1d(3);m.answer_lm.interactor=nn.Linear(3,3)
    a=capture_buffers(m);m.answer_lm.visual.running_mean.fill_(7);b=capture_buffers(m)
    before=m.answer_lm.interactor.weight.detach().clone()
    restore_buffers(m,a);assert not m.answer_lm.visual.running_mean.any()
    restore_buffers(m,b);assert m.answer_lm.visual.running_mean.eq(7).all()
    assert torch.equal(before,m.answer_lm.interactor.weight)
