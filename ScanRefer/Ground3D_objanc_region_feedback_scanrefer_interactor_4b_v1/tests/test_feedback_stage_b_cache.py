import copy
import pytest
import torch
from feedback_fixtures import source,batch
from region_feedback.proposals import FrozenProposalStage
from region_feedback.cache import make_record,validate_record,CacheWriter,CachedDataset

def full_record():
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    return make_record(p,b,0,s.answer_lm,index=0,include_full_scene=True)

def test_full_scene_cache_is_separate_and_keeps_frame_and_targets(tmp_path):
    r=full_record()
    assert r["protocol"]=="fixed_objects_full_scene_v4"
    assert r["superpoint_features"].shape==(4,8)
    assert r["superpoint_xyz"].shape==(4,3)
    assert r["supervision"]["gt_superpoint_masks"].shape==(1,4)
    assert r["supervision"]["gt_point_masks"].shape==(1,4)
    assert r["supervision"]["teacher_answer"]=="<obj>chair</obj>"
    w=CacheWriter(tmp_path,"train",{},1,full_scene=True);w.write(0,r);w.finalize()
    ds=CachedDataset(tmp_path,{},expected_count=1,split="train",full_scene=True)
    assert torch.equal(ds[0]["superpoint_features"],r["superpoint_features"])
    with pytest.raises(ValueError):CachedDataset(tmp_path,{},expected_count=1,split="train")

def test_full_scene_cache_rejects_misaligned_features_xyz_and_targets():
    r=full_record();validate_record(r,require_full_scene=True)
    for field in ["superpoint_features","superpoint_xyz"]:
        bad=copy.deepcopy(r);bad[field]=bad[field][:-1]
        with pytest.raises(ValueError):validate_record(bad,require_full_scene=True)
    bad=copy.deepcopy(r);bad["supervision"]["gt_superpoint_masks"]=torch.ones(1,3)
    with pytest.raises(ValueError):validate_record(bad,require_full_scene=True)
    bad=copy.deepcopy(r);bad["supervision"]["answer_entity_indices"]=[4]
    with pytest.raises(ValueError):validate_record(bad,require_full_scene=True)

def test_lightweight_a_record_does_not_silently_supply_stage_b_inputs():
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    r=make_record(p,b,0,s.answer_lm,index=0)
    assert r["protocol"]=="fixed_objects_textscore_v3" and "supervision" not in r
    with pytest.raises(ValueError):validate_record(r,require_full_scene=True)
