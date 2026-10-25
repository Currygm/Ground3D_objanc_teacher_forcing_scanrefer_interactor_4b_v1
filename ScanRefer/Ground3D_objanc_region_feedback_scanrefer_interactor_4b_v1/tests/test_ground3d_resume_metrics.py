
import importlib.util
import torch
import pytest

def test_resume_rng_reproduces_fps():
    assert importlib.util.find_spec('ground3d_migration.runtime') is not None, 'RNG persistence missing'
    from ground3d_migration.runtime import capture_rng,restore_rng
    from ground3d_migration.core import visual_indices
    xyz=torch.randn(2001,3)
    state=capture_rng();first=visual_indices(xyz,training=True)
    restore_rng(state);second=visual_indices(xyz,training=True)
    assert torch.equal(first,second)

def test_frozen_hash_includes_persistent_buffers():
    from runtime_helpers import frozen_digest
    model=torch.nn.BatchNorm1d(3).requires_grad_(False)
    first=frozen_digest(model)
    model.running_mean.add_(1)
    assert frozen_digest(model)!=first

def test_tie_aware_rank_and_score_stable_order():
    assert importlib.util.find_spec('ground3d_migration.diagnostics') is not None, 'Query rank diagnostics missing'
    from ground3d_migration.diagnostics import tie_aware_rank
    scores=torch.tensor([.2,.9,.9,.1])
    iou=torch.tensor([1.,1.,.5,0.])
    values=tie_aware_rank(scores,iou)
    assert values['best_query_rank']==1
    assert values['best_query_top1_rate']==1
    values=tie_aware_rank(scores,torch.tensor([0.,0.,0.,1.]))
    assert values['best_query_rank']==4
    assert values['best_query_top3_rate']==0
    assert values['best_query_top5_rate']==1
