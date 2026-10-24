import copy
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from torch import nn
from joint_lr_schedule import scheduler_for,learning_rate,GROUP_KEYS
from reason3d_runtime import load_segmentation_config,OBJECTIVE
from train_reason3d import save_checkpoint,load_resume
from ground3d_migration.runtime import capture_rng

ROOT=Path(__file__).resolve().parents[1]

class Toy(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.weights=nn.ParameterDict({n:nn.Parameter(torch.tensor([.7],dtype=torch.float64)) for n in GROUP_KEYS})
        self.answer_lm=SimpleNamespace(source_manifests={},tokenizer_snapshot=lambda:{"toy":"same"})
        self._training_phase="answer_only"
    def architecture_signature(self):return {"toy":"native-checkpoint-roundtrip","objective":OBJECTIVE}
    def trainable_state_dict(self):return {k:v.detach().cpu() for k,v in self.state_dict().items()}

def build(config):
    m=Toy(config);t=config["training"]
    groups=[{"name":n,"params":[m.weights[n]],"lr":t[keys[0]],"weight_decay":t["weight_decay"]} for n,keys in GROUP_KEYS.items()]
    o=torch.optim.AdamW(groups);s=scheduler_for(o,168,t)
    return m,o,s

def update(m,o,s,step):
    for n,p in m.weights.items():
        p.grad=torch.ones_like(p)*(1+step%3) if step>=504 or n in {"qwen_lora","structured_tokens"} else None
    o.step();o.zero_grad(set_to_none=True);s.step()

@pytest.mark.parametrize("epoch",[3,8])
def test_native_checkpoint_resume_keeps_momentum_schedule_and_next_updates(tmp_path,epoch):
    c=load_segmentation_config(ROOT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml")
    m,o,s=build(c);step=epoch*168
    for k in range(step):update(m,o,s,k)
    m._training_phase="answer_only" if epoch==3 else "joint"
    assert int(o.state[m.weights["qwen_lora"]]["step"])==step
    if epoch==3:assert m.weights["decoder"] not in o.state
    path=save_checkpoint(m,o,s,tmp_path,step,epoch,c,[capture_rng()])
    restored,ro,rs=build(c)
    got_step,got_epoch,states=load_resume(restored,ro,rs,path,1)
    assert (got_step,got_epoch)==(step,epoch) and len(states)==1
    assert rs.last_epoch==step and int(ro.state[restored.weights["qwen_lora"]]["step"])==step
    for ga,gb in zip(o.param_groups,ro.param_groups):assert ga["lr"]==gb["lr"]
    for k in range(step,step+105):
        update(m,o,s,k);update(restored,ro,rs,k)
    for name in GROUP_KEYS:assert torch.equal(m.weights[name],restored.weights[name])
    assert s.state_dict()==rs.state_dict()

def test_real_config_matches_baseline_except_experiment_paths_and_lr_fields():
    import yaml
    c=load_segmentation_config(ROOT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml")
    baseline=ROOT.parent/"Ground3D_objanc_teacher_forcing_reason3d_from_scanrefer_ep10_4b_v1"
    original=yaml.safe_load((baseline/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml").read_text())
    for key in ("data","model","point_encoder","decoder","loss","matcher","ranking","finetune_initialization","initialization","answer_policy"):
        assert c[key]==original[key],key
    original_training=dict(original["training"]);changed=dict(c["training"])
    original_training.pop("objective");original_training.pop("lr_schedule")
    changed.pop("objective");changed.pop("lr_schedule")
    for key in list(changed):
        if key.startswith("joint_") and key not in original_training:changed.pop(key)
    assert original_training==changed
    for name in GROUP_KEYS:
        for step in (0,1,167,168,503):
            import math
            base=original["training"][GROUP_KEYS[name][0]]
            expected=base*(.1+.9*.5*(1+math.cos(math.pi*step/2184)))
            assert learning_rate(name,step,168,c["training"])==pytest.approx(expected)

def test_native_final_checkpoint_gate_rejects_early_stop_and_accepts_final(tmp_path):
    from tools.select_final_checkpoint import select_final_checkpoint
    c=load_segmentation_config(ROOT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml")
    with pytest.raises(ValueError,match="exactly one"):select_final_checkpoint(c,tmp_path)
    m,o,s=build(c)
    for step in range(2184):update(m,o,s,step)
    m._training_phase="joint"
    path=save_checkpoint(m,o,s,tmp_path,2184,13,c,[capture_rng()])
    assert select_final_checkpoint(c,tmp_path)==path
