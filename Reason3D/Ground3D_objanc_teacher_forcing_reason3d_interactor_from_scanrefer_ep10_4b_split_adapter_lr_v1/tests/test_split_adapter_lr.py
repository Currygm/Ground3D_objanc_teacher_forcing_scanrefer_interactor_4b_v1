import copy
from pathlib import Path
import pytest
import torch
from torch import nn
import yaml
from reason3d_runtime import SourceModel, load_segmentation_config
from reason3d_training_schedule import set_training_epoch
from train_reason3d import scheduler_for
from reason3d_tuning_validation import validate_training_scalars

ROOT=Path(__file__).resolve().parents[1]

class Tiny(nn.Module):
    def __init__(self,split=True):
        super().__init__()
        self.answer_lm=nn.Module()
        self.answer_lm.structured=nn.Linear(2,2)
        self.answer_lm.language=nn.Module()
        self.answer_lm.language.lora_A=nn.Linear(2,2)
        self.answer_lm.interactor=nn.Linear(2,2)
        self.answer_lm.visual=nn.Linear(2,2)
        self.text_adapter=nn.Linear(2,2)
        self.visual_adapter=nn.Linear(2,2)
        self.decoder=nn.Linear(2,2)
        self.config={"training":{"structured_token_lr":1e-5,"qwen_lora_lr":1e-4,
            "interactor_lr":1e-4,"visual_lr":1e-5,"adapter_lr":1e-4,"decoder_lr":5e-5,
            "weight_decay":.01,"answer_warmup_epochs":3,"answer_lm_weight":3.,
            "obj_ce_weight":3.,"max_grad_norm":1.,"cosine_min_lr_ratio":.1}}
        if split:self.config["training"].update(text_adapter_lr=1e-4,visual_adapter_lr=5e-5)
        self._checkpoint_names={n for n,_ in self.named_parameters()}
    def groups(self):
        return SourceModel.optimizer_parameter_groups(self)

def test_split_groups_cover_every_parameter_once_with_distinct_rates():
    m=Tiny();groups=m.groups();byname={g["name"]:g for g in groups}
    assert "text_adapter" in byname and "visual_adapter" in byname
    assert "adapters" not in byname
    assert len(groups)==7
    assert byname["text_adapter"]["lr"]==1e-4
    assert byname["visual_adapter"]["lr"]==byname["decoder"]["lr"]==5e-5
    ids=[id(p) for g in groups for p in g["params"]]
    assert len(ids)==len(set(ids))
    assert set(ids)=={id(p) for p in m.parameters()}
    assert {id(p) for p in byname["text_adapter"]["params"]}=={id(p) for p in m.text_adapter.parameters()}
    assert {id(p) for p in byname["visual_adapter"]["params"]}=={id(p) for p in m.visual_adapter.parameters()}

def test_legacy_single_adapter_group_is_retained_without_overrides():
    m=Tiny(split=False);groups={g["name"]:g for g in m.groups()}
    assert len(groups)==6 and groups["adapters"]["lr"]==1e-4
    assert {id(p) for p in groups["adapters"]["params"]}=={id(p) for a in [m.text_adapter,m.visual_adapter] for p in a.parameters()}

@pytest.mark.parametrize("key,value",[
    ("text_adapter_lr",float("nan")),("visual_adapter_lr",float("inf")),
    ("text_adapter_lr",0.),("visual_adapter_lr",-1.),("text_adapter_lr",True)])
def test_invalid_split_rate_is_rejected(key,value):
    t=Tiny().config["training"];t[key]=value
    with pytest.raises(ValueError):validate_training_scalars(t)

@pytest.mark.parametrize("key",["text_adapter_lr","visual_adapter_lr"])
def test_partial_split_configuration_is_rejected(key):
    t=Tiny().config["training"];del t[key]
    with pytest.raises(ValueError):validate_training_scalars(t)

def test_answer_freezes_both_adapters_and_joint_restores_gradients():
    m=Tiny();opt=torch.optim.AdamW(m.groups())
    frozen={n:p.detach().clone() for n,p in m.named_parameters() if n.startswith(("text_adapter.","visual_adapter."))}
    set_training_epoch(m,0)
    assert not m.text_adapter.training and not m.visual_adapter.training
    assert m.answer_lm.interactor.training
    sum(p.square().sum() for p in m.parameters() if p.requires_grad).backward()
    assert all(p.grad is None for a in [m.text_adapter,m.visual_adapter] for p in a.parameters())
    opt.step();opt.zero_grad()
    for n,v in frozen.items():torch.testing.assert_close(dict(m.named_parameters())[n],v,rtol=0,atol=0)
    set_training_epoch(m,3)
    assert all(p.requires_grad for a in [m.text_adapter,m.visual_adapter] for p in a.parameters())
    sum(p.square().sum() for p in m.parameters() if p.requires_grad).backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for a in [m.text_adapter,m.visual_adapter] for p in a.parameters())

def test_split_rates_follow_original_cosine_and_optimizer_resume():
    m=Tiny();opt=torch.optim.AdamW(m.groups())
    sched=scheduler_for(opt,26,.1,answer_steps=6)
    for _ in range(10):
        sum(p.square().sum() for p in m.parameters()).backward()
        opt.step();opt.zero_grad();sched.step()
    byname={g["name"]:g for g in opt.param_groups}
    assert "text_adapter" in byname and "visual_adapter" in byname
    assert byname["text_adapter"]["lr"]==pytest.approx(2*byname["visual_adapter"]["lr"])
    other=Tiny();new=torch.optim.AdamW(other.groups());ns=scheduler_for(new,26,.1,answer_steps=6)
    new.load_state_dict(copy.deepcopy(opt.state_dict()));ns.load_state_dict(copy.deepcopy(sched.state_dict()))
    for g1,g2 in zip(opt.param_groups,new.param_groups):
        assert g1["name"]==g2["name"] and g1["lr"]==g2["lr"]
    for old,state in zip(opt.state.values(),new.state.values()):
        for k in old:torch.testing.assert_close(old[k],state[k],rtol=0,atol=0)
    for _ in range(5):
        assert [g["lr"] for g in opt.param_groups]==[g["lr"] for g in new.param_groups]
        opt.step();sched.step();new.step();ns.step()

def test_default_new_project_config_has_only_requested_settings():
    c=load_segmentation_config(ROOT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml")
    t=c["training"]
    assert c["project"]["artifact_root"]==str(ROOT)
    assert c["project"]["output_dir"].startswith(str(ROOT)+"/outputs/")
    assert t["text_adapter_lr"]==1e-4 and t["visual_adapter_lr"]==t["decoder_lr"]==5e-5
    assert (t["answer_warmup_epochs"],t["joint_epochs"],t["answer_lm_weight"],t["obj_ce_weight"])==(3,10,3.,3.)
    assert c["loss"]["aux_weight"]==c["ranking"]["weight"]==.5
    assert t["qwen_lora_lr"]==t["interactor_lr"]==1e-4
    assert c["finetune_initialization"]["source_completed_epochs"]==10
    assert c["model"]["generation"]["constrained"] is False
