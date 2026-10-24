
"""Official SpConv feature path, independent of OpenMMLab registries."""
from dataclasses import dataclass
import torch
from torch import nn
from .core import prepare_geometry,scatter_mean,visual_indices,strict_prefix_load

@dataclass
class GroundVisualTokens:
    hidden: torch.Tensor
    xyz: torch.Tensor

@dataclass
class GroundPointFeatures:
    compressed: GroundVisualTokens
    superpoint_hidden: torch.Tensor
    superpoint_xyz: torch.Tensor

class GroundVisualEncoder(nn.Module):
    def __init__(self,state):
        super().__init__()
        # PCCM otherwise writes inline artifacts below the account home.
        from pathlib import Path
        import pccm.constants
        pccm.constants.PCCM_INLINE_LIBRARY_PATH=Path(__file__).resolve().parents[1]/"cache/pccm/inline"
        import spconv.pytorch as spconv
        from .spconv_unet import SpConvUNet
        self.input_conv=spconv.SparseSequential(spconv.SubMConv3d(6,32,kernel_size=3,padding=1,bias=False,indice_key='subm1'))
        self.unet=SpConvUNet(num_planes=[32,64,96,128,160],return_blocks=True)
        self.output_layer=spconv.SparseSequential(nn.BatchNorm1d(32,eps=1e-4,momentum=.1),nn.ReLU(inplace=True))
        self.x_mask=nn.Sequential(nn.Linear(32,256),nn.ReLU(),nn.Linear(256,256))
        self.loading_report=[strict_prefix_load(getattr(self,n),state,prefix) for n,prefix in
            [('input_conv','input_conv.'),('unet','unet.'),('output_layer','output_layer.'),('x_mask','decoder.x_mask.')]]
        self.point2token=None

    def forward(self,cloud,raw_to,centers,*,cap=2000,sample_training=None,geometry=None):
        import spconv.pytorch as spconv
        device=next(self.parameters()).device
        cloud=cloud.to(device=device,dtype=torch.float32)
        raw_to=raw_to.to(device=device,dtype=torch.long)
        geometry=prepare_geometry(cloud[:,:3],cloud[:,3:6],raw_to) if geometry is None else {k:v.to(device) for k,v in geometry.items()}
        coord=geometry['voxel_coord']
        indices=torch.cat([torch.zeros(len(coord),1,device=device,dtype=torch.long),coord],dim=1).int()
        shape=(coord.max(0).values+1).clamp_min(128).tolist()
        # Sparse encoder follows official FP32 path; language projection casts separately.
        with torch.autocast(device_type=device.type,enabled=False):
            sparse=spconv.SparseConvTensor(geometry['voxel_features'].float(),indices,shape,1)
            sparse=self.input_conv(sparse);sparse,_=self.unet(sparse);sparse=self.output_layer(sparse)
            pooled=scatter_mean(sparse.features[geometry['inverse_map']],raw_to,len(centers))
            full=self.x_mask(pooled)
        xyz=centers.to(device=device,dtype=torch.float32)
        if full.shape!=(len(xyz),256):raise ValueError('Full superpoint alignment failed')
        idx=visual_indices(xyz,training=self.training if sample_training is None else sample_training,cap=cap)
        projected=self.point2token(full[idx].to(next(self.point2token.parameters()).dtype))
        return GroundPointFeatures(GroundVisualTokens(projected,xyz[idx]),full,xyz)
