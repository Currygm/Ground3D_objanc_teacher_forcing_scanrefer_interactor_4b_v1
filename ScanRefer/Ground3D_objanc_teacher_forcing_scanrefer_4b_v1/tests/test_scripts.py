import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def test_training_wrapper_preserves_cuda_selection(tmp_path):
    stub=tmp_path/'python'
    stub.write_text('#!/bin/sh\nprintf "CUDA=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    env={**os.environ,'PYTHON':str(stub),'CUDA_VISIBLE_DEVICES':'4,5,6,7'}
    env.pop('GPU_IDS',None)
    script=ROOT/'scripts/train_joint.sh'
    assert script.exists(), '3B wrapper missing'
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'CUDA=4,5,6,7' in result.stdout
    assert '--nproc_per_node=4' in result.stdout
    assert 'scanrefer_teacher_forcing_univlg9.yaml' in result.stdout

def test_two_stage_wrapper_runs_answer_before_joint(tmp_path):
    stub=tmp_path/'python'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    env={**os.environ,'PYTHON':str(stub),'CUDA_VISIBLE_DEVICES':'4,5,6,7'}
    env.pop('GPU_IDS',None)
    script=ROOT/'scripts/train_all.sh'
    assert script.exists(),'two-stage wrapper missing'
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.index('train_answer_3b.py')<result.stdout.index('train_segmentation.py')
    assert '--answer-checkpoint' in result.stdout
    assert 'outputs/open_answer/latest.pt' in result.stdout

def test_two_stage_wrapper_stops_if_answer_fails(tmp_path):
    stub=tmp_path/'python'; log=tmp_path/'calls'
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CALL_LOG"\nexit 7\n')
    stub.chmod(0o755)
    env={**os.environ,'PYTHON':str(stub),'CALL_LOG':str(log),'GPU_IDS':'4,5,6,7'}
    script=ROOT/'scripts/train_all.sh'
    assert script.exists(),'two-stage wrapper missing'
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==7
    assert 'train_answer_3b.py' in log.read_text()
    assert 'train_segmentation.py' not in log.read_text()

def test_launchers_require_explicit_unique_device_selection(tmp_path):
    env={**os.environ};env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('GPU_IDS',None)
    for name in ['train_answer.sh','train_joint.sh','evaluate.sh']:
        result=subprocess.run(['bash',str(ROOT/'scripts'/name)],env=env,capture_output=True,text=True)
        assert result.returncode==2 and 'Explicit' in result.stderr
        result=subprocess.run(['bash',str(ROOT/'scripts'/name)],env={**env,'GPU_IDS':'1,1'},capture_output=True,text=True)
        assert result.returncode==2 and 'Duplicate' in result.stderr
