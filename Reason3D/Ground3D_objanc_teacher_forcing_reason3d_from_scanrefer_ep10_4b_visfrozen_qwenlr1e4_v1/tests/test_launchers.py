import json,os,shutil,subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize("finish",[True,False])
def test_same_gpu_train_eval_launcher_and_early_stop_guard(tmp_path,finish):
    project=Path(__file__).resolve().parents[1]
    scripts=tmp_path/"scripts";shutil.copytree(project/"scripts",scripts)
    fake=tmp_path/"fake_python"
    fake.write_text(r"""#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
root=Path(__file__).resolve().parent
args=sys.argv[1:]
if any(a.endswith('gpu_selection.py') for a in args):
 print('GPU-test1,GPU-test2,GPU-test4,GPU-test6');sys.exit(0)
if any(a.endswith('select_final_checkpoint.py') for a in args):
 if not (root/'finished').exists():sys.exit(4)
 print(root/'epoch_013_step_00002184.pt');sys.exit(0)
if args[:2]==['-B','-']:sys.exit(0)
kind='train' if any(a.endswith('train_reason3d.py') for a in args) else 'eval'
with (root/'events.jsonl').open('a') as f:f.write(json.dumps({'stage':kind,'gpus':os.environ['CUDA_VISIBLE_DEVICES'],'args':args})+'\n')
if kind=='train' and '--max-steps' not in args:(root/'finished').touch()
""")
    fake.chmod(0o755)
    env={**os.environ,"PYTHON":str(fake),"GPU_IDS":"1,2,4,6","OUTPUT_DIR":str(tmp_path/"output"),"EVAL_OUTPUT_DIR":str(tmp_path/"evaluation")}
    command=["bash",str(scripts/"train_then_eval.sh")]
    if not finish:command += ["--max-steps","1"]
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    events=[json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()]
    assert result.returncode==(0 if finish else 4),result.stdout+result.stderr
    assert [e["stage"] for e in events]==(["train","eval"] if finish else ["train"])
    assert all(e["gpus"]=="GPU-test1,GPU-test2,GPU-test4,GPU-test6" for e in events)
    if finish:
        assert str(tmp_path/'epoch_013_step_00002184.pt') in events[-1]['args']
        assert events[-1]['args'][-3:]==['--modes','teacher','generated']
