import torch
from torch import nn
from test_ground3d_qwen_causal import tiny_lm
from ground3d_migration.visual import GroundVisualTokens

def test_answer_and_obj_ce_share_causal_logits():
    lm=tiny_lm(); calls=[]
    hook=lm.language.get_base_model().model.language_model.register_forward_hook(lambda *a:calls.append(1))
    result=lm._causal(["where is chair"],["It is <obj>red chair</obj> near <anc>table</anc>"],[GroundVisualTokens(torch.randn(32,64),None)])
    hook.remove();assert calls==[1]
    assert result["metrics"]["obj_token_count"]==2
    (3*result["loss"]+3*result["losses"]["answer_obj_ce"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in lm.named_parameters() if "lora_" in n)

def test_warmup_freezes_interactor_and_joint_restores_it():
    from reason3d_training_schedule import set_training_epoch
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.answer_lm=nn.Module()
            self.answer_lm.structured=nn.Linear(2,2);self.answer_lm.language=nn.Module();self.answer_lm.language.lora_A=nn.Linear(2,2)
            self.answer_lm.visual=nn.BatchNorm1d(2);self.answer_lm.interactor=nn.Linear(2,2)
            self.decoder=nn.Linear(2,2);self.text_adapter=nn.Linear(2,2);self.visual_adapter=nn.Linear(2,2)
            self._checkpoint_names={n for n,p in self.named_parameters()};self.config={"training":{"answer_warmup_epochs":3}}
    m=Model();assert set_training_epoch(m,0)=="answer_only"
    assert not m.answer_lm.interactor.training
    assert not any(p.requires_grad for p in m.answer_lm.interactor.parameters())
    assert not m.answer_lm.visual.training
    assert set_training_epoch(m,3)=="joint"
    assert all(p.requires_grad for p in m.parameters())
