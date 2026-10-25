
import importlib.util
import json
import torch
import pytest

def feature():
    assert importlib.util.find_spec('ground3d_migration.core') is not None, 'Ground3D migration core not implemented'
    from ground3d_migration import core
    return core

@pytest.mark.parametrize('s', [1999, 2000, 2001])
def test_sampling_alignment_full_decoder_and_eval(s):
    c = feature()
    xyz = torch.randn(s, 3)
    feat = torch.arange(s).float()[:, None].expand(s, 256)
    idx = c.visual_indices(xyz, training=True, cap=2000)
    assert len(idx) == min(s, 2000)
    assert idx.unique().numel() == len(idx)
    assert torch.equal(feat[idx, 0], idx.float())
    assert c.visual_indices(xyz, training=False, cap=2000).tolist() == list(range(s))
    assert feat.shape[0] == s

def test_fps_identical_positions_unique_and_rng_resume():
    c = feature()
    xyz = torch.zeros(2001, 3)
    state = torch.get_rng_state()
    a = c.visual_indices(xyz, training=True, cap=2000)
    torch.set_rng_state(state)
    b = c.visual_indices(xyz, training=True, cap=2000)
    assert torch.equal(a,b)
    assert a.unique().numel()==2000

def test_migrated_rows_untied_preserve_other_logits_and_optimizer():
    c = feature()
    embedding = torch.nn.Embedding(10, 4)
    head = torch.nn.Linear(4, 10, bias=False)
    input_before = embedding.weight.detach().clone()
    output_before = head.weight.detach().clone()
    rows = c.MigratedTokenRows([3,4,5,10], embedding.weight, head.weight, closing_source_id=4)
    embedding.requires_grad_(False); head.requires_grad_(False)
    assert torch.equal(rows.input_rows[0], input_before[3])
    assert torch.equal(rows.output_rows[0], output_before[3])
    assert torch.equal(rows.input_rows[3], input_before[4])
    ids = torch.tensor([[1,3,5,10]])
    h = rows.embed(ids,embedding)
    assert torch.equal(h[0,0],input_before[1])
    logits = rows.logits(h,head)
    assert logits.shape == (1,4,11)
    assert torch.allclose(logits[...,1],head(h)[...,1])
    opt=torch.optim.AdamW(rows.parameters(),lr=.1,weight_decay=.1)
    logits.sum().backward();opt.step()
    assert torch.equal(embedding.weight,input_before)
    assert torch.equal(head.weight,output_before)

def test_voxel_mean_features_and_raw_superpoints_never_merge():
    c=feature()
    xyz=torch.tensor([[0.,0,0],[.005,0,0],[.04,0,0]])
    rgb=torch.tensor([[-1.,0,1],[1.,0,-1],[0.,0,0]])
    sp=torch.tensor([5,7,7])
    g=c.prepare_geometry(xyz,rgb,sp,voxel_size=.02)
    assert g['voxel_features'].shape==(2,6)
    assert torch.equal(g['point_to_superpoint'],torch.tensor([0,1,1]))
    assert torch.equal(g['inverse_map'],torch.tensor([0,0,1]))
    assert torch.allclose(g['voxel_features'][0,:3],torch.zeros(3))
    assert g['superpoint_coord'].shape==(2,3)

def test_signature_rejects_old_architecture_and_hash_drift():
    c=feature()
    expected={'objective':'ground3d_objanc_v1','source_sha':'abc'}
    c.validate_signature(expected,dict(expected))
    with pytest.raises(ValueError):c.validate_signature(expected,{'objective':'spatiallm'})
    with pytest.raises(ValueError):c.validate_signature(expected,{**expected,'source_sha':'xyz'})

def test_closing_anc_copies_obj_close_even_inside_padded_qwen_vocabulary():
    from ground3d_migration.core import MigratedTokenRows
    input_weight=torch.arange(60,dtype=torch.float32).reshape(20,3)
    output_weight=input_weight+100
    rows=MigratedTokenRows([4,5,6,10],input_weight,output_weight,closing_source_id=5)
    assert torch.equal(rows.input_rows[3],input_weight[5])
    assert torch.equal(rows.output_rows[3],output_weight[5])
    assert torch.equal(rows.input_rows[:3],input_weight[[4,5,6]])
