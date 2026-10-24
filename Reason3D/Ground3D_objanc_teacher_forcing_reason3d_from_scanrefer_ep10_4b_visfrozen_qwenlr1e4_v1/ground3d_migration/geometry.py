
"""Raw-point geometry for official Ground3D voxel and superpoint pooling."""
import torch
import hashlib,os,uuid
from pathlib import Path
from .core import prepare_geometry

def cached_geometry(xyz,rgb,labels,cache_dir=None):
    cache=Path(cache_dir or Path(__file__).resolve().parents[1]/'cache/geometry')
    cache.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(b'ground3d-voxel-0.02-raw-superpoint-v1')
    for value in [xyz,rgb,labels]:
        value=value.detach().cpu().contiguous()
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    fingerprint=digest.hexdigest();path=cache/(fingerprint+'.pt')
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('fingerprint')!=fingerprint:raise ValueError('Geometry cache identity mismatch')
        result=saved['geometry']
        if set(result)!={'voxel_coord','voxel_features','inverse_map','point_to_superpoint','superpoint_values','superpoint_coord'}:
            raise ValueError('Geometry cache keys differ')
        if len(result['inverse_map'])!=len(xyz) or len(result['point_to_superpoint'])!=len(xyz):raise ValueError('Geometry cache mapping lengths differ')
        if any(not torch.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite geometry cache')
        return result
    result=prepare_geometry(xyz,rgb,labels,voxel_size=.02)
    temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save({'fingerprint':fingerprint,'geometry':result},temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return result

def prepare_ground_points(xyz,rgb,normals,gt_loc_mask,*,num_bins,world_extent_m,random_sample,generator,
    point_to_superpoint=None,raw_superpoint_id=None):
    labels=raw_superpoint_id if raw_superpoint_id is not None else point_to_superpoint
    if labels is None:raise ValueError('Ground3D requires superpoint labels')
    g=cached_geometry(xyz,rgb,labels)
    shifted=xyz-xyz.amin(0)
    n=len(g['superpoint_coord'])
    # Keep raw points: official encoder expands voxel features back before pooling.
    return {**g,'ground_geometry':g,'point_cloud':torch.cat([xyz,rgb],dim=1),'high_xyz':shifted,
        'gt_loc_high':gt_loc_mask.float() if gt_loc_mask is not None else torch.zeros(len(xyz)),
        'superpoint_grid_coord':torch.floor(g['superpoint_coord']/.02),
        'stage0_to_superpoint':None}
