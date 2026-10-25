import copy
import pytest
import torch
from feedback_fixtures import source,batch
from region_feedback.proposals import FrozenProposalStage

def record():
    from region_feedback.cache import make_record
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    return make_record(p,b,0,s.answer_lm,index=0)

def test_cached_record_has_bound_objects_and_prediction_is_gt_free():
    r=record()
    assert r["evidence"]["features"].shape==(3,8)
    assert r["masks"].shape==(50,4)
    assert r["truth"].shape==(4,)
    assert "features" not in r
    assert not r["masks"].requires_grad

def test_manifest_integrity_complete_ids_and_split(tmp_path):
    from region_feedback.cache import CacheWriter,CachedDataset
    identity={"first_sha":"source","data":"data"}
    writer=CacheWriter(tmp_path,"val",identity,1)
    writer.write(0,record());writer.finalize()
    ds=CachedDataset(tmp_path,identity,expected_count=1,split="val")
    assert len(ds)==1 and ds[0]["sample_id"]=="fixture:0"
    with pytest.raises(ValueError):CachedDataset(tmp_path,{"first_sha":"other"},expected_count=1,split="val")
    with pytest.raises(ValueError):CachedDataset(tmp_path,identity,expected_count=2,split="val")
    with pytest.raises(ValueError):CachedDataset(tmp_path,identity,expected_count=1,split="train")
    path=tmp_path/"records/00000000.pt"
    value=torch.load(path,weights_only=False);value["truth"].zero_();torch.save(value,path)
    with pytest.raises(ValueError,match="checksum"):ds[0]

def test_duplicate_ids_and_incomplete_cache_rejected(tmp_path):
    from region_feedback.cache import CacheWriter,CachedDataset
    w=CacheWriter(tmp_path,"train",{},2)
    w.write(0,record())
    with pytest.raises(ValueError):w.finalize()
    r=record();r["index"]=1;w.write(1,r)
    with pytest.raises(ValueError,match="Duplicate"):w.finalize()
