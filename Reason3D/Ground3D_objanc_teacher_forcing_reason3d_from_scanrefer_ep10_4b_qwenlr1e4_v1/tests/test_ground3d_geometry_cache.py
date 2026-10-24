import torch
import pytest
from ground3d_migration.geometry import cached_geometry

def test_geometry_cache_is_independent_and_preserves_raw_mapping(tmp_path):
    xyz=torch.tensor([[0.,0.,0.],[.005,0.,0.],[.03,0.,0.]])
    rgb=torch.zeros_like(xyz);sp=torch.tensor([5,7,7])
    first=cached_geometry(xyz,rgb,sp,tmp_path)
    assert len(first['voxel_coord'])==2
    assert first['point_to_superpoint'].tolist()==[0,1,1]
    assert len(list(tmp_path.glob('*.pt')))==1
    first_path=next(tmp_path.glob('*.pt'))
    second=cached_geometry(xyz,rgb,sp,tmp_path)
    assert all(torch.equal(first[k],second[k]) for k in first)
    # Geometry identity depends on coordinates/features/superpoints, never GT targets.
    cached_geometry(xyz,rgb+1,sp,tmp_path)
    assert len(list(tmp_path.glob('*.pt')))==2
    first_path.write_bytes(b'corrupt')
    with pytest.raises(Exception):cached_geometry(xyz,rgb,sp,tmp_path)
