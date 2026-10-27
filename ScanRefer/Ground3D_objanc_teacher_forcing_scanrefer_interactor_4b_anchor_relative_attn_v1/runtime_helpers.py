"""CPU provenance, atomic persistence and nonblocking training logs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import torch

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def checkpoint_manifest(config):
    from ground3d_migration.core import verify_download
    if not isinstance(config,dict):raise ValueError('Ground3D requires both official manifests')
    return {name:verify_download(config['model'][key]) for name,key in [('qwen','model_name_or_path'),('ground3d','ground3d_weights_dir')]}

def frozen_digest(model):
    digest=hashlib.sha256()
    excluded=set(model.trainable_state_dict()) if hasattr(model,'trainable_state_dict') else {n for n,p in model.named_parameters() if p.requires_grad}
    for name,tensor in model.state_dict().items():
        if name in excluded:continue
        tensor=tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def load_trainable_state(model,state):
    expected=model.trainable_state_dict() if hasattr(model,'trainable_state_dict') else {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not isinstance(state,dict) or set(state)!=set(expected):raise ValueError('Checkpoint trainable-state keys differ')
    for n,v in state.items():
        if not isinstance(v,torch.Tensor) or v.shape!=expected[n].shape or v.dtype!=expected[n].dtype or not torch.isfinite(v).all():raise ValueError('Checkpoint trainable-state shapes/values differ')
    with torch.no_grad():
        destination=model.state_dict()
        for n,v in state.items():destination[n].copy_(v)

def validate_checkpoint(checkpoint,objective,signature,manifest):
    if checkpoint.get('objective')!=objective or checkpoint.get('architecture_signature')!=signature:
        raise ValueError('Checkpoint objective or architecture differs')
    if checkpoint.get('base_manifest')!=manifest:
        raise ValueError('Frozen base checkpoint SHA256 manifest differs')

def atomic_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()

class AsyncJsonLog:
    """Stalled filesystem never stalls training; overflow disables logging."""
    def __init__(self,path):
        self.path=Path(path); self.disabled=False
        self.queue=queue.Queue(maxsize=128)
        self.worker=threading.Thread(target=self._run,daemon=True)
        self.worker.start()

    def submit(self,record):
        if self.disabled:return
        try:self.queue.put_nowait(json.dumps(record,ensure_ascii=False))
        except queue.Full:self.disabled=True

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as handle:
                while True:
                    item=self.queue.get()
                    if item is None:return
                    handle.write(item+'\n'); handle.flush()
        except OSError as exc:
            self.disabled=True
            print(f'Async metric log disabled: {exc}',flush=True)

    def close(self,timeout=1):
        try:self.queue.put_nowait(None)
        except queue.Full:self.disabled=True
        self.worker.join(timeout=timeout)
