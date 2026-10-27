"""Thin native torchrun launcher. Never queues, auto-selects GPUs or evaluates after training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT.parent/'Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1/env/bin/python'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['answer','joint','evaluate'])
    known,args=parser.parse_known_args()
    selected=os.environ.get('RUN')
    path=Path(os.environ.get('CONFIG',ROOT/'configs'/f'{selected or "baseline"}.yaml')).resolve(strict=True)
    if not path.is_relative_to(ROOT):raise ValueError('CONFIG must be in the new project')
    config=yaml.safe_load(path.read_text())
    run=config['run']
    if selected is not None and selected!=run:raise ValueError('RUN and CONFIG disagree')
    from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
    load_segmentation_config(path)
    raw=os.environ.get('GPU_IDS','').split(',')
    if not raw or any(not x for x in raw) or len(set(raw))!=len(raw):raise ValueError('Explicit unique GPU_IDS required')
    if not (all(x.isdigit() for x in raw) or all(x.startswith('GPU-') for x in raw)):raise ValueError('Mixed or invalid GPU_IDS')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping=dict(line.strip().split(', ') for line in inventory.strip().splitlines())
    uuids=[mapping[x] if x.isdigit() else x for x in raw]
    if len(set(uuids))!=len(raw) or not set(uuids)<=set(mapping.values()):raise ValueError('Unknown/duplicate GPU UUID')
    if known.stage!='evaluate' and len(uuids)!=4:raise ValueError('Training requires exactly four GPUs')
    if known.stage=='answer' and run!='baseline':raise ValueError('Shared Answer uses RUN=baseline')
    if known.stage=='evaluate' and any(x.split('=')[0] in ['--max-samples','--val-annotations','--dataset-name','--eval-cache-root'] for x in args):raise ValueError('Official evaluation requires all 9483 samples')
    cache=ROOT/'cache/huggingface';cache.mkdir(parents=True,exist_ok=True)
    tmp=ROOT.parents[2]/'.arattn_tmp';tmp.mkdir(exist_ok=True)
    assert tmp.resolve().is_relative_to(Path('/home/huangxiang/codes/hx_l'))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids),HF_HOME=str(cache),TMPDIR=str(tmp),
             TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
    # Only runs when the user explicitly invokes this launcher, not during CPU verification.
    subprocess.run([str(PYTHON),'-c','import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()=='+str(len(uuids))],env=env,check=True)
    if known.stage=='evaluate':
        if len(args)<2:parser.error('evaluate requires CHECKPOINT OUTPUT [options]')
        target=Path(args[1]).resolve()
        if not target.is_relative_to(ROOT):raise ValueError('Evaluation output escapes project')
        if target.exists() and any(target.iterdir()):raise FileExistsError('Evaluation output must be fresh')
        entry='evaluate_segmentation_unconstrained.py';argv=[str(path),*args]
    else:
        entry='train_answer_3b.py' if known.stage=='answer' else 'train_segmentation.py';argv=[str(path),*args]
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    command=[str(PYTHON),'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(uuids)}',str(ROOT/entry),*argv]
    (logdir/f'{run}_{known.stage}_{stamp}.launch.json').write_text(json.dumps(dict(command=command,gpu_uuids=uuids,physical_ids=[k for k,v in mapping.items() if v in uuids],config=str(path)),indent=2))
    print(json.dumps(dict(command=command,gpu_uuids=uuids)),flush=True)
    # Preserve SIGINT/exit status; shell tee wrappers record stdout and stderr.
    os.chdir(ROOT);os.execve(str(PYTHON),command,env)


if __name__=='__main__':main()
