"""CPU contracts for Joint vision training and phase-specific Qwen LR."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from reason3d_runtime import ObjAncSegmentationModel
from reason3d_training_schedule import set_training_epoch
from train_reason3d import scheduler_for,save_checkpoint,load_resume
from ground3d_migration.runtime import capture_rng

class Tiny(ObjAncSegmentationModel):
    def __init__(self,config=None):
        nn.Module.__init__(self)
        self.answer_lm=nn.Module()
        self.answer_lm.visual=nn.Sequential(nn.Linear(2,2),nn.BatchNorm1d(2))
        self.answer_lm.structured=nn.Linear(2,2)
        self.answer_lm.lora_layer=nn.Linear(2,2)
        self.answer_lm.tokenizer_snapshot=lambda:{"tokens":[1,2]}
        self.answer_lm.source_manifests={"qwen":"frozen"}
        self.decoder=nn.Linear(2,2)
        self.text_adapter=nn.Linear(2,2)
        self.visual_adapter=nn.Linear(2,2)
        self._checkpoint_names={n for n,p in self.named_parameters()}
        self.config=config or {"training":{"answer_warmup_epochs":3}}
    def architecture_signature(self):
        return {"test":"visfrozen"}

def optimizer(model):
    specs=[("structured_tokens",model.answer_lm.structured,1e-5),
           ("qwen_lora",model.answer_lm.lora_layer,1e-5),
           ("ground_visual",model.answer_lm.visual,1e-5),
           ("adapters",nn.ModuleList([model.text_adapter,model.visual_adapter]),1e-4),
           ("decoder",model.decoder,1e-4)]
    return torch.optim.AdamW([{"name":n,"params":list(m.parameters()),"lr":lr} for n,m,lr in specs],weight_decay=.01)

@pytest.mark.parametrize("epoch",[0,2,3,7,12])
def test_vision_is_frozen_in_answer_and_updates_with_adapter_in_joint(epoch):
    model=Tiny();opt=optimizer(model)
    set_training_epoch(model,epoch)
    before=copy.deepcopy(model.answer_lm.visual.state_dict())
    adapter=model.visual_adapter.weight.detach().clone()
    for _ in range(2):
        model.eval();model.train()
        assert model.answer_lm.visual.training==(epoch>=3)
        assert all(p.requires_grad==(epoch>=3) for p in model.answer_lm.visual.parameters())
        x=model.answer_lm.visual(torch.randn(4,2))
        z=model.answer_lm.structured(x)+model.answer_lm.lora_layer(x)
        if epoch>=3:
            assert model.visual_adapter.training and model.decoder.training
            z=model.decoder(model.visual_adapter(z)+model.text_adapter(z))
        z.sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    after=model.answer_lm.visual.state_dict()
    if epoch>=3:
        assert any(not torch.equal(before[n],after[n]) for n in before)
        assert not torch.equal(before["1.running_mean"],after["1.running_mean"])
    else:
        for n,v in before.items():torch.testing.assert_close(after[n],v,rtol=0,atol=0)
    assert (not torch.equal(adapter,model.visual_adapter.weight))==(epoch>=3)
    state=model.trainable_state_dict()
    for n,v in model.answer_lm.visual.state_dict().items():
        torch.testing.assert_close(state["answer_lm.visual."+n],v,rtol=0,atol=0)
    assert all((p in opt.state)==(epoch>=3) for p in model.answer_lm.visual.parameters())

def test_learning_rate_boundary_is_continuous_cosine_except_qwen_multiplier():
    model=Tiny();opt=optimizer(model)
    scheduler=scheduler_for(opt,3024,.1,answer_steps=504,joint_qwen_multiplier=10.)
    for step in range(3025):
        factor=.1+.45*(1+math.cos(math.pi*step/3024))
        for g in opt.param_groups:
            base={"structured_tokens":1e-5,"qwen_lora":1e-5,"ground_visual":1e-5,"adapters":1e-4,"decoder":1e-4}[g["name"]]
            expected=base*factor*(10 if g["name"]=="qwen_lora" and step>=504 else 1)
            assert g["lr"]==pytest.approx(expected,rel=1e-12,abs=1e-15)
        if step<3024:opt.step();scheduler.step()

@pytest.mark.parametrize("epoch",[3,8,13])
def test_checkpoint_keeps_frozen_vision_and_restores_phase_lr_and_adam_moments(tmp_path,epoch):
    train=tmp_path/"train.json";val=tmp_path/"val.json"
    train.write_text("[]");val.write_text("[]")
    config={"training":{"epochs":13,"answer_warmup_epochs":3,"joint_epochs":10},
            "data":{"train_sources":[{"annotations":str(train)}],"val_annotations":str(val)},
            "finetune_initialization":{"checkpoint":"scanrefer10.pt","sha256":"parent"}}
    m=Tiny(config);opt=optimizer(m);sched=scheduler_for(opt,26,.1,answer_steps=6,joint_qwen_multiplier=10)
    set_training_epoch(m,epoch-1)
    m.answer_lm.lora_layer(torch.ones(2,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True);sched.step()
    for _ in range(epoch*2-1):opt.step();sched.step()
    path=save_checkpoint(m,opt,sched,tmp_path,epoch*2,epoch,config,[capture_rng()])
    new=Tiny(copy.deepcopy(config));newopt=optimizer(new);news=scheduler_for(newopt,26,.1,answer_steps=6,joint_qwen_multiplier=10)
    step,completed,_=load_resume(new,newopt,news,path,1)
    set_training_epoch(new,min(completed,12))
    assert step==epoch*2 and completed==epoch
    for n,v in m.state_dict().items():torch.testing.assert_close(new.state_dict()[n],v,rtol=0,atol=0)
    for a,b in zip(opt.param_groups,newopt.param_groups):assert a["lr"]==b["lr"]
    for a,b in zip(opt.state.values(),newopt.state.values()):
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    for _ in range(3):
        opt.step();sched.step();newopt.step();news.step()
        assert [g["lr"] for g in opt.param_groups]==[g["lr"] for g in newopt.param_groups]

def test_configuration_and_launcher_contract():
    import yaml
    from reason3d_runtime import load_segmentation_config
    root=Path(__file__).resolve().parents[1]
    c=load_segmentation_config(root/"configs/reason3d_answer_lm3_objce3_warmup3_joint15.yaml")
    assert (c["training"]["epochs"],c["training"]["joint_epochs"])==(18,15)
    assert c["training"]["visual_lr"]==1e-5
    assert c["training"]["joint_qwen_lora_lr"]==1e-4
    assert c["training"]["freeze_visual_encoder"] is False
    assert c["loss"]["aux_weight"]==.5
    assert c["finetune_initialization"]["source_completed_epochs"]==10
    assert str(root)==c["project"]["artifact_root"]
    s=(root/"scripts/train_then_eval.sh").read_text()
    assert "evaluate_unconstrained.sh" in s
    assert "generated_no_fallback" not in s
    assert "warmup3_joint15" in (root/"scripts/runtime.sh").read_text()
