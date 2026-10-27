import copy
import json
from pathlib import Path
import torch
from torch import nn
import pytest
import yaml
from anchor_runtime import PhaseScheduler,capture_buffers,restore_buffers
from anchor_eval import unique_multiple_lookup,validate_coverage
from anchor_config import validate_anchor_config

ROOT=Path(__file__).resolve().parents[1]


def test_configs_and_isolation_contract():
    configs=[]
    for run in ['baseline','semantic','full']:
        c=yaml.safe_load((ROOT/f'configs/{run}.yaml').read_text());validate_anchor_config(c)
        c=copy.deepcopy(c);c['run']='RUN'
        for key in ['output_dir','evaluation_dir']:c['project'][key]='PATH'
        for key in ['semantic_enabled','geometry_enabled']:c['anchor_relative_attention'][key]=False
        configs.append(c)
    assert configs[0]==configs[1]==configs[2]
    bad=copy.deepcopy(configs[0]);bad['run']='baseline';bad['training']['decoder_lr']=-1
    with pytest.raises(ValueError):validate_anchor_config(bad)


def test_full_cpu_resume_buffers_momentum_rng_and_lr():
    m=nn.BatchNorm1d(2);o=torch.optim.AdamW([{'params':list(m.parameters()),'lr':1e-4,'name':'qwen_lora'}])
    s=PhaseScheduler(o,9,.1,'joint')
    for _ in range(4):
        loss=m(torch.randn(4,2)).square().sum();loss.backward();o.step();o.zero_grad();s.step()
    weights=copy.deepcopy(m.state_dict());bs=capture_buffers(m);op=copy.deepcopy(o.state_dict());ss=s.state_dict();rng=torch.get_rng_state()
    m2=nn.BatchNorm1d(2);m2.load_state_dict(weights)
    o2=torch.optim.AdamW([{'params':list(m2.parameters()),'lr':1e-4,'name':'qwen_lora'}]);s2=PhaseScheduler(o2,9,.1,'joint')
    o2.load_state_dict(op);s2.load_state_dict(ss);restore_buffers(m2,bs)
    torch.set_rng_state(rng);x=torch.randn(4,2)
    for model,opt,sch in [(m,o,s),(m2,o2,s2)]:
        model(x).square().sum().backward();opt.step();sch.step()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],atol=0,rtol=0)
    assert s.state_dict()==s2.state_dict()


def test_subsets_and_exact_denominator():
    rows=[dict(scene_id='s',object_id=1,object_name='office_chair'),dict(scene_id='s',object_id=2,object_name='chair'),dict(scene_id='s',object_id=3,object_name='bed')]
    labels=[dict(raw_category='office chair',nyu40class='chair'),dict(raw_category='chair',nyu40class='chair'),dict(raw_category='bed',nyu40class='bed')]
    lookup=unique_multiple_lookup(rows,labels)
    assert lookup[('s','1')]=='multiple' and lookup[('s','3')]=='unique'
    payloads=[{mode:dict(sample_indices=[r],sample_ids=[str(r)]) for mode in ['teacher','generated']} for r in range(3)]
    validate_coverage(payloads,3)
    payloads[1]['generated']['sample_indices']=[0]
    with pytest.raises(ValueError):validate_coverage(payloads,3)


def test_training_and_scripts_do_not_evaluate_implicitly():
    for name in ['train_answer.sh','train_joint.sh']:
        assert 'evaluate' not in (ROOT/'scripts'/name).read_text()
    launch=(ROOT/'launch_experiment.py').read_text()
    assert "'GPU_IDS'" in launch and 'UUID' in launch and 'RUN and CONFIG disagree' in launch


def test_decoder_coefficients_and_metadata_boundary():
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
    cfg=yaml.safe_load((ROOT/'configs/full.yaml').read_text())['anchor_relative_attention']
    torch.manual_seed(42);base=ObjAncFullCycleDecoder(visual_dim=8,text_dim=8,hidden_dim=8,num_heads=2,feedforward_dim=16,anchor_config={**cfg,'semantic_enabled':False,'geometry_enabled':False})
    torch.manual_seed(42);full=ObjAncFullCycleDecoder(visual_dim=8,text_dim=8,hidden_dim=8,num_heads=2,feedforward_dim=16,anchor_config=cfg)
    assert len([n for n,p in full.named_parameters() if n.endswith(('.anchor_sem','.anchor_geo'))])==4
    for k,v in base.state_dict().items():assert torch.equal(v,full.state_dict()[k])
    v=[torch.randn(5,8)];t=[torch.randn(4,8)];x=[torch.randn(5,3)];roles=[torch.tensor([0,1,2,3])]
    spans=torch.tensor([[0,1,0,0],[0,0,1,0]],dtype=torch.bool)
    base.eval();full.eval()
    a=base(v,t,x,roles)
    b=full(v,t,x,roles,entities=[(spans,['OBJ','ANC'])],superpoint_counts=[torch.ones(5)],sample_ids=['s'])
    torch.testing.assert_close(a['pred_masks'],b['pred_masks'],atol=1e-6,rtol=1e-5)
    b['pred_masks'].square().sum().backward()
    assert any(p.grad is not None and p.grad.abs()>0 for n,p in full.named_parameters() if n.endswith('.anchor_sem'))


def test_actual_checkpoint_rank_buffers_and_cross_run_rejection(tmp_path,monkeypatch):
    import train_segmentation as train
    from ground3d_migration.runtime import capture_rng
    class Tiny(nn.Module):
        objective='test'
        def __init__(self,run):
            super().__init__();self.p=nn.Parameter(torch.ones(2));self.register_buffer('bn',torch.zeros(2))
            self.config={'run':run,'model':{'model_name_or_path':'unused'}}
        def architecture_signature(self):return {'run':self.config['run']}
        def trainable_state_dict(self):return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}
    monkeypatch.setattr(train,'data_manifest',lambda c:{'hash':'fixed'})
    monkeypatch.setenv('WORLD_SIZE','2');monkeypatch.setenv('RANK','1')
    model=Tiny('full');op=torch.optim.AdamW([{'params':[model.p],'name':'qwen_lora','lr':1e-4}]);sc=PhaseScheduler(op,9,.1,'joint')
    model.p.square().sum().backward();op.step();sc.step()
    bs=[{'bn':torch.ones(2)},{'bn':torch.full((2,),2.)}]
    path=train.save_checkpoint(model,op,sc,tmp_path,1,1,model.config,manifest={},frozen_hash='x',rng_states=[capture_rng(),capture_rng()],buffer_states=bs)
    train.load_resume(model,op,sc,path,{})
    assert torch.equal(model.bn,bs[1]['bn'])
    model.config['run']='baseline'
    with pytest.raises(ValueError):train.load_resume(model,op,sc,path,{})


def test_non_qwen_legacy_lr_and_scalars():
    from train_segmentation import scheduler_for
    a=nn.Parameter(torch.ones(()));b=nn.Parameter(torch.ones(()))
    old=torch.optim.AdamW([{'params':[a],'name':'decoder','lr':1e-4}])
    new=torch.optim.AdamW([{'params':[b],'name':'decoder','lr':1e-4}])
    legacy=scheduler_for(old,9,.1);current=PhaseScheduler(new,9,.1,'joint')
    for _ in range(9):
        assert old.param_groups[0]['lr']==pytest.approx(new.param_groups[0]['lr'])
        old.step();legacy.step();new.step();current.step()


@pytest.mark.parametrize('options',[['--max-samples=100'],['--max-samples','100'],['--val-annotations','/tmp/other.json'],['--dataset-name','other'],['--eval-cache-root','/tmp/cache']])
def test_formal_evaluation_cannot_be_shortened_or_retargeted(monkeypatch,options):
    import sys
    import evaluate_segmentation as evaluator
    monkeypatch.setattr(sys,'argv',['evaluate','unused.yaml','unused.pt','unused-output',*options])
    with pytest.raises(ValueError,match='full ScanRefer'):
        evaluator.main()
