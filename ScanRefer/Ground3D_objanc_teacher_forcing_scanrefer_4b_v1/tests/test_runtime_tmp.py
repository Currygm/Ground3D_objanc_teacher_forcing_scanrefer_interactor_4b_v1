import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
@pytest.mark.parametrize("name", ["train_answer.sh", "train_joint.sh", "evaluate.sh"])
def test_launcher_supports_multiprocessing_socket(name):
    source = (ROOT / "scripts" / name).read_text()
    setup = source.split('exec "$PYTHON"', 1)[0].replace('${BASH_SOURCE[0]}', '$0')
    probe = "from multiprocessing.connection import Listener; s=Listener(); s.close(); print('socket-ok')"
    env = dict(os.environ, GPU_IDS="1,3,4,5", PYTHON=sys.executable)
    env.pop("GROUND3D_TMPDIR", None)
    result = subprocess.run(
        ["bash", "-c", setup + '\nexec "$PYTHON" -c "$1"', str(ROOT / "scripts" / name), probe],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "socket-ok" in result.stdout

@pytest.mark.parametrize("name", ["train_answer.sh", "train_joint.sh", "evaluate.sh"])
@pytest.mark.parametrize("bad_dir", ["/tmp/g3_test", str(ROOT / "cache/tmp")])
def test_launcher_rejects_unsafe_tmpdir(name, bad_dir):
    source = (ROOT / "scripts" / name).read_text()
    setup = source.split('exec "$PYTHON"', 1)[0].replace('${BASH_SOURCE[0]}', '$0')
    result = subprocess.run(
        ["bash", "-c", setup, str(ROOT / "scripts" / name)],
        env=dict(os.environ, GPU_IDS="1,3,4,5", PYTHON=sys.executable, GROUND3D_TMPDIR=bad_dir),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "TMPDIR" in result.stderr

def test_launcher_dataloader_tensor_transfer():
    name = "train_answer.sh"
    source = (ROOT / "scripts" / name).read_text()
    setup = source.split('exec "$PYTHON"', 1)[0].replace('${BASH_SOURCE[0]}', '$0')
    probe = "import torch; from torch.utils.data import DataLoader,TensorDataset; data=TensorDataset(torch.arange(8)); got=torch.cat([x[0] for x in DataLoader(data,batch_size=2,num_workers=2,timeout=15)]); assert torch.equal(got,torch.arange(8)); print('dataloader-ok')"
    env = dict(os.environ, GPU_IDS="1,3,4,5", PYTHON=sys.executable)
    env.pop("GROUND3D_TMPDIR", None)
    result = subprocess.run(
        ["bash", "-c", setup + '\nexec "$PYTHON" -c "$1"', str(ROOT / "scripts" / name), probe],
        env=env, text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dataloader-ok" in result.stdout
