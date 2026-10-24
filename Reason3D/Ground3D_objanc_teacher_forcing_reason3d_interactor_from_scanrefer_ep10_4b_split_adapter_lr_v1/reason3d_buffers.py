"""Keep per-rank visual BN statistics across epoch-boundary resumes."""
import torch
def capture_buffers(model):
    return {n:v.detach().cpu().clone() for n,v in model.named_buffers()
            if n.startswith(("answer_lm.visual.","answer_lm.interactor."))}
def restore_buffers(model,state):
    current=capture_buffers(model)
    if not isinstance(state,dict) or set(state)!=set(current):raise ValueError("Resume rank buffer keys differ")
    destination=dict(model.named_buffers())
    with torch.no_grad():
        for name,value in state.items():
            if value.shape!=current[name].shape or value.dtype!=current[name].dtype:raise ValueError("Resume buffer protocol differs")
            destination[name].copy_(value)
