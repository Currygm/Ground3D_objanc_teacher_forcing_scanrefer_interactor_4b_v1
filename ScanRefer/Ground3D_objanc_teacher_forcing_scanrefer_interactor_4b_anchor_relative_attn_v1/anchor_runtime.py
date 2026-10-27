"""Stage-aware scheduling and checkpoint metadata for the controlled ablation."""
import math
import torch


class PhaseScheduler:
    def __init__(self,optimizer,total,minimum,phase):
        if phase not in ('answer','joint') or total<2:raise ValueError('Invalid phase schedule')
        self.optimizer=optimizer;self.total=int(total);self.minimum=float(minimum);self.phase=phase
        for group in optimizer.param_groups:group.setdefault('initial_lr',float(group['lr']))
        self.base=[float(g['initial_lr']) for g in optimizer.param_groups];self.completed=0
        self._apply()

    def _apply(self):
        t=min(self.completed,self.total-1)
        for g,base in zip(self.optimizer.param_groups,self.base):
            if g['name']=='qwen_lora':
                g['lr']=1e-4*t/(self.total-1) if self.phase=='answer' else 1e-5+4.5e-5*(1+math.cos(math.pi*t/(self.total-1)))
            else:g['lr']=base*(self.minimum+(1-self.minimum)*.5*(1+math.cos(math.pi*min(self.completed/self.total,1))))

    def step(self):self.completed+=1;self._apply()
    def state_dict(self):
        return {k:getattr(self,k) for k in ['total','minimum','phase','base','completed']}
    def load_state_dict(self,state):
        if any(state[k]!=getattr(self,k) for k in ['total','minimum','phase','base']):raise ValueError('Schedule configuration differs')
        self.completed=int(state['completed']);self._apply()


def capture_buffers(model):
    return {n:v.detach().cpu().clone() for n,v in model.named_buffers()}


def restore_buffers(model,state):
    destination=dict(model.named_buffers())
    if set(destination)!=set(state):raise ValueError('Buffer keys differ')
    with torch.no_grad():
        for n,v in state.items():
            if v.shape!=destination[n].shape or v.dtype!=destination[n].dtype:raise ValueError('Buffer schema differs')
            destination[n].copy_(v)
