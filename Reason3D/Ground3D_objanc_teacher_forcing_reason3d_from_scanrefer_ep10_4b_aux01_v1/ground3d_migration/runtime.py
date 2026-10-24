
import random
import numpy as np
import torch

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}

def restore_rng(state):
    if set(state)!={'python','numpy','torch','cuda'}:raise ValueError('Incomplete RNG checkpoint')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if len(state['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(state['cuda'])

def memory_report():
    if not torch.cuda.is_initialized():return {'peak_allocated_bytes':None,'peak_reserved_bytes':None}
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
