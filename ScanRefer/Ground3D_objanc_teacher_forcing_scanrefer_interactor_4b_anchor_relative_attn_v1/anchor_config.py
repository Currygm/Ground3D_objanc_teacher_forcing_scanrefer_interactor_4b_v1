"""Pinned experiment contract, with only mathematical chunk sizes tunable."""
import copy
import math
import yaml
from pathlib import Path


def validate_anchor_config(c):
    run=c.get('run');modes={'baseline':(False,False),'semantic':(True,False),'full':(True,True)}
    if run not in modes:raise ValueError('Unknown RUN')
    a=c['anchor_relative_attention'];s,g=modes[run]
    fixed=dict(layers=[8,9],semantic_enabled=s,geometry_enabled=g,scalar_init=0,scalar_bound='tanh',
               numerical_eps=1e-6,geometry_mask_detach=True,stats_dtype='float32')
    if set(a)!=set(fixed)|{'candidate_chunk','visual_chunk'} or any(a[k]!=v for k,v in fixed.items()):raise ValueError('Anchor contract differs')
    for key in ['candidate_chunk','visual_chunk']:
        if type(a[key]) is not int or a[key]<1:raise ValueError('Invalid chunk size')
    if c['model']['generation']['constrained'] is not False:raise ValueError('Unconstrained evaluation required')
    if c['data']['expected_val_samples']!=9483:raise ValueError('Expected full ScanRefer val')
    original=yaml.safe_load((Path(__file__).parent/'configs/scanrefer_teacher_forcing_univlg9.yaml').read_text())
    original['data']['expected_val_samples']=9483
    for section in ['loss','matcher','ranking','point_encoder','decoder','answer_policy','data','interactor','initialization']:
        if c[section]!=original[section]:raise ValueError('Source contract differs: '+section)
    expected=copy.deepcopy(original['training'])
    expected.update(qwen_lora_lr=1e-4,qwen_schedule='answer_linear_joint_cosine_v1')
    if c['training']!=expected:raise ValueError('Training ablation differs')
    if Path(c['project']['output_dir']).name!=run or Path(c['project']['evaluation_dir']).name!=run:raise ValueError('RUN/output mismatch')

