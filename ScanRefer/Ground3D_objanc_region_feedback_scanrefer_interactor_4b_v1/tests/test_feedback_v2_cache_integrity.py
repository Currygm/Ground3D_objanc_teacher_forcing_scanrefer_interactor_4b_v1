import pytest
import torch
from region_feedback.cache import CacheWriter
from test_feedback_v2_cache import record

def test_unfinished_cache_cannot_silently_finalize_modified_record(tmp_path):
    w=CacheWriter(tmp_path,"val",{},1);w.write(0,record())
    path=tmp_path/"records/00000000.pt"
    r=torch.load(path,weights_only=False);r["masks"].add_(1);torch.save(r,path)
    with pytest.raises(ValueError,match="checksum"):w.finalize()

def test_cache_seed_and_generation_protocol_are_identity_inputs():
    from region_feedback.factory import cache_identity
    identity={"first_sha":"a","data":{},"base_config":{},"implementation":{},
              "generation_protocol":"unconstrained"}
    a=cache_identity({},identity)
    identity["generation_protocol"]="constrained"
    assert a!=cache_identity({},identity)

def test_v2_disables_unused_legacy_geometry_and_language_modules():
    from region_feedback import regions,language
    assert not hasattr(regions,"RegionTokenEncoder")
    assert not hasattr(language,"build_feedback_visuals")
