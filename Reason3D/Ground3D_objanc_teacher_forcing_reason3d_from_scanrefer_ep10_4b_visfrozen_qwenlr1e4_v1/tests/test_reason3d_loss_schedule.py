from types import SimpleNamespace
import pytest
import torch
from torch import nn


def test_obj_mask_uses_causal_prediction_positions_and_excludes_other_tokens():
    from reason3d_obj_loss import obj_prediction_mask,obj_content_loss
    parsed=[SimpleNamespace(entities=[SimpleNamespace(role="OBJ"),SimpleNamespace(role="ANC")])]
    spans=torch.tensor([[False,True,True,False,False],[False,False,False,False,True]])
    maps=[(None,torch.arange(4,9),None,spans)]
    target=torch.full((1,10),-100);target[0,3:9]=torch.tensor([1,2,3,4,5,6])
    mask=obj_prediction_mask(maps,parsed,target)
    assert mask.nonzero().tolist()==[[0,4],[0,5]]
    selected=target.ne(-100);logits=torch.randn(6,9,requires_grad=True)
    loss,metrics=obj_content_loss(logits,target[selected],mask[selected])
    expected=torch.nn.functional.cross_entropy(logits[1:3].float(),torch.tensor([2,3]))
    torch.testing.assert_close(loss,expected)
    loss.backward();assert logits.grad[[0,3,4,5]].count_nonzero()==0
    assert logits.grad[1:3].abs().sum()>0 and metrics["obj_token_count"]==2


def test_real_causal_forward_shares_logits_and_obj_loss_reaches_lora():
    from test_ground3d_qwen_causal import tiny_lm
    from ground3d_migration.visual import GroundVisualTokens
    lm=tiny_lm();calls=[]
    hook=lm.language.get_base_model().model.language_model.register_forward_hook(lambda *args:calls.append(1))
    output=lm._causal(["where is chair"],["It is <obj>red chair</obj> near <anc>table</anc>"],[GroundVisualTokens(torch.randn(3,64),torch.zeros(3,3))])
    hook.remove();assert len(calls)==1
    assert output["metrics"]["obj_token_count"]==2
    (3*output["loss"]+3*output["losses"]["answer_obj_ce"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in lm.named_parameters() if "lora_" in n)


def test_answer_then_joint_restores_exact_trainable_boundary_and_frozen_visual_bn():
    from reason3d_training_schedule import set_training_epoch
    class LM(nn.Module):
        def __init__(self):
            super().__init__();self.structured=nn.Linear(2,2);self.lora_layer=nn.Linear(2,2);self.visual=nn.BatchNorm1d(2)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.answer_lm=LM();self.decoder=nn.Linear(2,2);self.text_adapter=nn.Linear(2,2);self.visual_adapter=nn.Linear(2,2)
            self._checkpoint_names={n for n,p in self.named_parameters()};self.config={"training":{"answer_warmup_epochs":3}}
    model=Model();assert set_training_epoch(model,2)=="answer_only"
    assert not model.answer_lm.visual.training
    assert all(p.requires_grad==(n.startswith("answer_lm.structured.") or "lora_" in n) for n,p in model.named_parameters())
    assert set_training_epoch(model,3)=="joint"
    assert not model.answer_lm.visual.training
    assert all(p.requires_grad==(not n.startswith("answer_lm.visual.")) for n,p in model.named_parameters())


def test_checkpoint_cadence_keeps_final_answer_and_joint5_joint10():
    from reason3d_stage_checkpoints import should_save_epoch
    assert [e for e in range(1,14) if should_save_epoch(e,3,13,5)]==[3,8,13]
