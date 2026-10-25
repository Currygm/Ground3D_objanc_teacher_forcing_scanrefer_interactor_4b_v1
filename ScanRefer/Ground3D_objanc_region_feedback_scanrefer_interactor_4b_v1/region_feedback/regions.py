"""FP32 predicted-mask pooling; no geometry or per-query token encoder."""
from dataclasses import dataclass
import torch

@dataclass
class RegionEvidence:
    features: torch.Tensor
    valid_mask: torch.Tensor

@torch.no_grad()
def pool_regions(features,xyz,masks,point_to_superpoint,threshold=.5):
    if features.ndim!=2 or xyz.shape!=(len(features),3) or masks.ndim!=2 or masks.shape[1]!=len(features):
        raise ValueError("Features/centers/masks must share the full superpoint axis")
    if not len(features) or not all(torch.isfinite(x).all() for x in (features,xyz,masks)):
        raise ValueError("Empty or nonfinite region inputs")
    with torch.autocast(device_type=features.device.type,enabled=False):
        mapping=point_to_superpoint.to(device=features.device,dtype=torch.long)
        if mapping.ndim!=1 or not len(mapping) or mapping.min()<0 or mapping.max()>=len(features):
            raise ValueError("Invalid raw-point to superpoint mapping")
        probability=masks.detach().float().sigmoid()
        counts=torch.bincount(mapping,minlength=len(features)).float()
        weight=probability*(probability>=threshold)*counts[None]
        mass=weight.sum(-1)
        pooled=weight@features.detach().float()/mass.clamp_min(1e-12)[:,None]
        valid=mass>0
        pooled[~valid]=0
        return RegionEvidence(pooled,valid)
