"""Device validation and compatibility entrypoints for object-feedback v3."""
import os
import torch

def explicit_device():
    selected=os.getenv("CUDA_VISIBLE_DEVICES","")
    parts=selected.split(",")
    if not selected or any(not x.isdigit() for x in parts) or len(set(parts))!=len(parts):
        raise RuntimeError("Explicit, unique CUDA_VISIBLE_DEVICES IDs are required")
    if not torch.cuda.is_available():raise RuntimeError("Selected CUDA devices are unavailable")
    local=int(os.getenv("LOCAL_RANK","0"))
    if local>=torch.cuda.device_count():raise RuntimeError("Local rank exceeds selected devices")
    torch.cuda.set_device(local)
    return torch.device("cuda",local)

def train_main():
    from .runtime_v2 import train_main as run
    return run()

def evaluate_main():
    from .runtime_v2 import evaluate_main as run
    return run()
