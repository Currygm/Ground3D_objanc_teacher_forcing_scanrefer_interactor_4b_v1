"""CPU regressions for Interactor/Qwen Answer alignment and Qwen LR warmup."""
import copy,math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from test_ground3d_qwen_causal import tiny_lm
from reason3d_runtime import ObjAncSegmentationModel
from reason3d_training_schedule import set_training_epoch,answer_only_forward
from train_reason3d import scheduler_for,save_checkpoint,load_resume
from ground3d_migration.runtime import capture_rng

class Visual(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(3,256);self.bn=nn.BatchNorm1d(256);self.grad_flags=[]
    def forward(self,cloud,mapping,xyz,**kwargs):
        self.grad_flags.append(torch.is_grad_enabled())
        return SimpleNamespace(superpoint_hidden=self.bn(self.proj(cloud)),compressed=None)

class Interactor(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(256,64);self.frozen_anchor=nn.Parameter(torch.ones(1),requires_grad=False);self.grad_flags=[]
    def forward(self,memories):
        self.grad_flags.append(torch.is_grad_enabled())
        return [self.proj(x.mean(0,keepdim=True)).expand(32,-1) for x in memories]

class Tiny(ObjAncSegmentationModel):
    def __init__(self):
        nn.Module.__init__(self);self.answer_lm=tiny_lm()
        self.answer_lm.visual=Visual();self.answer_lm.interactor=Interactor()
        self.decoder=nn.Linear(64,64);self.text_adapter=nn.Linear(64,64);self.visual_adapter=nn.Linear(256,64)
        self.answer_lm_weight=3.;self.obj_ce_weight=3.
        self.config={"training":{"answer_warmup_epochs":3,"qwen_lora_lr":1e-4,
            "structured_token_lr":1e-5,"interactor_lr":1e-4,"visual_lr":1e-5,
            "adapter_lr":1e-4,"decoder_lr":1e-4,"weight_decay":.01}}
        self._checkpoint_names={n for n,p in self.named_parameters() if p.requires_grad}
    def architecture_signature(self):return {"test":"answer_interactor_qwen_warmup"}

def batch():
    return {"point_cloud":[torch.randn(8,3,requires_grad=True)],
        "point_to_superpoint":[torch.arange(8)],"superpoint_coord":[torch.randn(8,3)],
        "question":["where is chair"],"teacher_answer":["It is <obj>red chair</obj> near <anc>table</anc>"]}

@pytest.mark.parametrize("epoch",[0,2,3,12])
def test_phase_boundaries_keep_interactor_and_lora_active(epoch):
    m=Tiny();keys=set(m.trainable_state_dict());set_training_epoch(m,epoch)
    for _ in range(2):
        m.eval();m.train()
        assert m.answer_lm.interactor.training
        assert m.answer_lm.visual.training==(epoch>=3)
    for n,p in m.named_parameters():
        answer=n.startswith(("answer_lm.structured.","answer_lm.interactor.")) or ("lora_" in n and ".visual." not in n)
        assert p.requires_grad==(n in m._checkpoint_names and (epoch>=3 or answer)),n
    assert not m.answer_lm.interactor.frozen_anchor.requires_grad
    assert set(m.trainable_state_dict())==keys

def test_real_answer_ce_reaches_interactor_lora_and_tokens_without_visual_graph():
    torch.manual_seed(12);m=Tiny();opt=torch.optim.AdamW(m.optimizer_parameter_groups())
    # Isolate gradient routing from the phase-policy test above.
    for n,p in m.named_parameters():
        answer=n.startswith(("answer_lm.structured.","answer_lm.interactor.")) or ("lora_" in n and ".visual." not in n)
        p.requires_grad_(n in m._checkpoint_names and answer)
    m._training_phase="answer_only";m.answer_lm.visual.eval();m.answer_lm.interactor.train()
    frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
    buffers=copy.deepcopy(m.answer_lm.visual.state_dict());before=m.answer_lm.interactor.proj.weight.detach().clone()
    b=batch();result=answer_only_forward(m,b)
    result["loss"].backward()
    assert m.answer_lm.visual.grad_flags==[False]
    assert m.answer_lm.interactor.grad_flags==[True]
    assert b["point_cloud"][0].grad is None
    assert m.answer_lm.interactor.proj.weight.grad is not None
    assert torch.isfinite(m.answer_lm.interactor.proj.weight.grad).all()
    assert m.answer_lm.interactor.proj.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if "lora_" in n)
    assert m.answer_lm.structured.input_rows.grad.abs().sum()>0
    assert m.answer_lm.structured.output_rows.grad.abs().sum()>0
    opt.step()
    assert not torch.equal(before,m.answer_lm.interactor.proj.weight)
    for n,v in frozen.items():torch.testing.assert_close(dict(m.named_parameters())[n],v,rtol=0,atol=0)
    for n,v in buffers.items():torch.testing.assert_close(m.answer_lm.visual.state_dict()[n],v,rtol=0,atol=0)

def test_joint_visual_graph_and_outer_eval_no_grad_are_preserved():
    m=Tiny();set_training_epoch(m,3)
    rows=m.answer_lm.encode_batch_points(batch())
    assert rows[0].compressed.hidden.requires_grad
    rows[0].compressed.hidden.square().mean().backward()
    assert m.answer_lm.visual.proj.weight.grad.abs().sum()>0
    assert m.answer_lm.interactor.proj.weight.grad.abs().sum()>0
    m.eval()
    with torch.no_grad():rows=m.answer_lm.encode_batch_points(batch())
    assert not rows[0].compressed.hidden.requires_grad
    assert m.answer_lm.visual.grad_flags[-1] is False
    assert m.answer_lm.interactor.grad_flags[-1] is False

def test_qwen_peak_is_last_answer_update_then_joint_cosine_other_groups_unchanged():
    names=["qwen_lora","structured_tokens","interactor","ground_visual","adapters","decoder"]
    bases=[1e-4,1e-5,1e-4,1e-5,1e-4,1e-4]
    opt=torch.optim.AdamW([{"name":n,"params":[nn.Parameter(torch.ones(1))],"lr":lr} for n,lr in zip(names,bases)])
    sched=scheduler_for(opt,2184,.1,answer_steps=504)
    for update in range(1,2185):
        for g,base in zip(opt.param_groups,bases):
            if g["name"]=="qwen_lora":
                factor=update/504 if update<=504 else .1+.45*(1+math.cos(math.pi*(update-504)/1680))
            else:factor=.1+.45*(1+math.cos(math.pi*(update-1)/2184))
            assert g["lr"]==pytest.approx(base*factor,rel=1e-12,abs=1e-15)
        opt.step();sched.step()
    assert opt.param_groups[0]["lr"]==pytest.approx(1e-5)

@pytest.mark.parametrize("epoch",[3,8,13])
def test_warmup_checkpoint_resume_preserves_adam_moments_and_next_rates(tmp_path,epoch):
    from test_reason3d_checkpoint import Tiny as StateTiny,configuration
    config=configuration(tmp_path);m=StateTiny(config);m._training_phase="answer_only" if epoch==3 else "joint"
    def make_opt(model):
        return torch.optim.AdamW([{"name":"qwen_lora","params":[model.linear.weight],"lr":1e-4},
            {"name":"structured_tokens","params":[model.linear.bias],"lr":1e-5}])
    opt=make_opt(m);s=scheduler_for(opt,26,.1,answer_steps=6)
    for _ in range(epoch*2):
        m.linear(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad();s.step()
    ckpt=save_checkpoint(m,opt,s,tmp_path,epoch*2,epoch,config,[capture_rng()],[{}])
    new=StateTiny(copy.deepcopy(config));no=make_opt(new);ns=scheduler_for(no,26,.1,answer_steps=6)
    step,completed,_=load_resume(new,no,ns,ckpt,1)
    assert (step,completed,ns.last_epoch)==(epoch*2,epoch,epoch*2)
    for a,b in zip(opt.state.values(),no.state.values()):
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for _ in range(3):
        assert [g["lr"] for g in opt.param_groups]==[g["lr"] for g in no.param_groups]
        opt.step();s.step();no.step();ns.step()

def test_config_identity_rates_and_native_eval_launcher():
    from reason3d_runtime import load_segmentation_config
    root=Path(__file__).resolve().parents[1]
    c=load_segmentation_config(root/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml")
    assert c["project"]["artifact_root"]==str(root)
    assert c["training"]["qwen_lora_lr"]==1e-4
    assert c["training"]["interactor_lr"]==1e-4
    assert c["training"]["lr_schedule"]=="qwen_answer_linear_warmup_joint_cosine"
    assert (c["training"]["answer_warmup_epochs"],c["training"]["joint_epochs"])==(3,10)
    assert c["loss"]["aux_weight"]==.5
    assert "scanrefer_interactor_4b_v1" in c["finetune_initialization"]["checkpoint"]
    assert c["model"]["generation"]["constrained"] is False
    assert "--modes teacher generated" in (root/"scripts/train_then_eval.sh").read_text()
