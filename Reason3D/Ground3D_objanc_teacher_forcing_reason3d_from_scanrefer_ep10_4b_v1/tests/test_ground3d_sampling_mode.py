import torch
from torch import nn
from ground3d_migration.language import GroundObjAncLM

def test_frozen_batchnorm_does_not_disable_training_sampling():
    class Visual(nn.Module):
        def __init__(self):super().__init__();self.bn=nn.BatchNorm1d(2);self.seen=[]
        def forward(self,cloud,mapping,xyz,*,cap,sample_training=None):
            self.seen.append((self.training,sample_training,cap));return None
    lm=GroundObjAncLM.__new__(GroundObjAncLM);nn.Module.__init__(lm)
    lm.visual=Visual().requires_grad_(False).eval()
    lm.training=True
    batch={'point_cloud':[torch.zeros(2001,6)],'point_to_superpoint':[torch.arange(2001)],'superpoint_coord':[torch.zeros(2001,3)]}
    lm.encode_batch_points(batch)
    assert lm.visual.seen[-1]==(False,True,2000)
    lm.training=False;lm.encode_batch_points(batch)
    assert lm.visual.seen[-1]==(False,False,2000)
