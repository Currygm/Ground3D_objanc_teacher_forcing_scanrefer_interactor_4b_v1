import json
from pathlib import Path
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


def test_real_paths_configs_data_and_init_contract():
    from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
    from anchor_eval import load_subset_lookup
    for run in ['baseline','semantic','full']:
        c=load_segmentation_config(ROOT/f'configs/{run}.yaml')
        val=json.loads(Path(c['data']['val_annotations']).read_text())['annotations']
        train=json.loads(Path(c['data']['train_sources'][0]['annotations']).read_text())['annotations']
        assert (len(train),len(val))==(36569,9483)
        lookup=load_subset_lookup(c)
        assert all((str(r['scene_id']),str(r['object_id'])) in lookup for r in val)


@pytest.mark.parametrize('run',['baseline','semantic','full'])
def test_actual_parameter_groups_and_frozen_scalar_checkpoint(run):
    import yaml
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
    c=yaml.safe_load((ROOT/f'configs/{run}.yaml').read_text())
    m=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(m);m.config=c
    m.answer_lm=nn.Module();m.answer_lm.structured=nn.Linear(2,2);m.answer_lm.visual=nn.Linear(2,2);m.answer_lm.interactor=nn.Linear(2,2)
    m.answer_lm.language=nn.Module();m.answer_lm.language.lora_A=nn.Linear(2,2)
    m.text_adapter=nn.Linear(2,2);m.visual_adapter=nn.Linear(2,2)
    m.decoder=ObjAncFullCycleDecoder(visual_dim=2,text_dim=2,hidden_dim=8,num_heads=2,feedforward_dim=16,anchor_config=c['anchor_relative_attention'])
    m._checkpoint_names={n for n,p in m.named_parameters() if p.requires_grad}
    groups=m.optimizer_parameter_groups()
    actual={id(p) for g in groups for p in g['params']}
    assert actual=={id(p) for p in m.parameters() if p.requires_grad}
    scalars=[g for g in groups if g['name']=='anchor_scalars']
    assert len(scalars)==(0 if run=='baseline' else 1)
    if scalars:assert scalars[0]['weight_decay']==0 and len(scalars[0]['params'])==(2 if run=='semantic' else 4)
    assert len([n for n in m.trainable_state_dict() if n.endswith(('.anchor_sem','.anchor_geo'))])==4
    # Model-level zero connection is essential for no-ANC DDP ranks.
    m._connect_trainable_graph(torch.tensor(0.,requires_grad=True)).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_disabled_decoder_has_no_rng_or_output_change():
    import yaml
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
    cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())['anchor_relative_attention']
    kwargs=dict(visual_dim=2,text_dim=2,hidden_dim=8,num_heads=2,feedforward_dim=16)
    torch.manual_seed(42);old=ObjAncFullCycleDecoder(**kwargs);rng=torch.get_rng_state()
    torch.manual_seed(42);new=ObjAncFullCycleDecoder(**kwargs,anchor_config=cfg)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k])
    args=([torch.randn(3,2),torch.randn(5,2)],[torch.randn(4,2),torch.randn(2,2)],[torch.randn(3,3),torch.randn(5,3)],[torch.tensor([0,1,2,3]),torch.tensor([1,3])])
    a=old(*args);b=new(*args)
    assert torch.equal(a['pred_masks'],b['pred_masks']) and torch.equal(a['pred_logits'],b['pred_logits'])


def test_active_decoder_handles_different_text_and_visual_lengths():
    import yaml
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
    cfg=yaml.safe_load((ROOT/'configs/full.yaml').read_text())['anchor_relative_attention']
    model=ObjAncFullCycleDecoder(visual_dim=2,text_dim=2,hidden_dim=8,num_heads=2,feedforward_dim=16,anchor_config=cfg)
    entities=[(torch.tensor([[0,1,0],[0,0,1]],dtype=torch.bool),['OBJ','ANC']),(torch.tensor([[0,1,0,0],[0,0,1,0]],dtype=torch.bool),['OBJ','ANC'])]
    out=model([torch.randn(3,2),torch.randn(5,2)],[torch.randn(3,2),torch.randn(4,2)],[torch.randn(3,3),torch.randn(5,3)],[torch.tensor([0,1,2]),torch.tensor([0,1,2,3])],entities=entities,superpoint_counts=[torch.ones(3),torch.ones(5)],sample_ids=['a','b'])
    assert out['pred_masks'].shape==(2,50,5)
    out['pred_masks'].sum().backward()


def test_fp64_geometric_projection_identity():
    from spatiallm_compressed_visual_objanc_full_attention.model.anchor_relative_attention import geometry_delta,project
    torch.manual_seed(2)
    a=nn.MultiheadAttention(8,2,batch_first=True).double();pe=nn.Sequential(nn.Linear(3,8),nn.Tanh()).double()
    xyz=torch.randn(5,3,dtype=torch.float64);mu=torch.randn(4,3,dtype=torch.float64);c=torch.randn(3,4,dtype=torch.float64);q=torch.randn(2,3,4,dtype=torch.float64)
    out=geometry_delta(a,pe,xyz,mu,c,q,2,3)
    delta=torch.einsum('jk,knd->jnd',c,pe(xyz[None]-mu[:,None])-pe(xyz)[None])
    keys=project(a,delta,1,bias=False).reshape(3,5,2,4)
    expected=torch.einsum('hjd,jnhd->hjn',q,keys)/2
    torch.testing.assert_close(out,expected,atol=1e-10,rtol=1e-8)
