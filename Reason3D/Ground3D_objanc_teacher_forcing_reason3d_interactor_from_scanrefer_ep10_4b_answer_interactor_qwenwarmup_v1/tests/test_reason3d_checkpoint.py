import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from ground3d_migration.runtime import capture_rng
from train_reason3d import save_checkpoint,load_resume,scheduler_for

class Tiny(nn.Module):
    def __init__(self,config):
        super().__init__();self.linear=nn.Linear(2,2);self.register_buffer("visual_bn",torch.tensor([2.]))
        self.config=config;self.answer_lm=SimpleNamespace(tokenizer_snapshot=lambda:{"tokens":[1,2]},source_manifests={"qwen":"frozen"})
    def architecture_signature(self):return {"test":"ground3d_reason3d"}
    def trainable_state_dict(self):return self.state_dict()


def configuration(tmp_path):
    train=tmp_path/"train.json";val=tmp_path/"val.json";train.write_text("[]");val.write_text("[]")
    return {"project":{"artifact_root":str(tmp_path)},"training":{"epochs":13,"answer_warmup_epochs":3,"joint_epochs":10},
        "data":{"train_sources":[{"annotations":str(train)}],"val_annotations":str(val)},"finetune_initialization":{"checkpoint":"scanrefer10.pt","sha256":"parent"}}


@pytest.mark.parametrize("epoch",[3,8,13])
def test_real_checkpoint_restores_all_state_scheduler_and_optimizer(tmp_path,epoch):
    config=configuration(tmp_path);model=Tiny(config);model._training_phase="answer_only" if epoch==3 else "joint"
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5);sched=scheduler_for(opt,26,.1)
    model.linear(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad();sched.step()
    for _ in range(epoch*2-1):opt.step();sched.step()
    path=save_checkpoint(model,opt,sched,tmp_path,epoch*2,epoch,config,[capture_rng()],[{}])
    new=Tiny(copy.deepcopy(config));newopt=torch.optim.AdamW(new.parameters(),lr=1e-5);news=scheduler_for(newopt,26,.1)
    step,completed,rng=load_resume(new,newopt,news,path,1)
    assert (step,completed,news.last_epoch)==(epoch*2,epoch,epoch*2)
    for n,v in model.state_dict().items():torch.testing.assert_close(new.state_dict()[n],v,rtol=0,atol=0)
    assert len(newopt.state)==len(opt.state)
    opt.step();sched.step();newopt.step();news.step()
    assert opt.param_groups[0]["lr"]==newopt.param_groups[0]["lr"]
    with pytest.raises(ValueError,match="world"):load_resume(new,newopt,news,path,2)
    new.config["training"]["epochs"]=14
    with pytest.raises(ValueError,match="config"):load_resume(new,newopt,news,path,1)


def test_auto_evaluation_rejects_early_checkpoint_and_accepts_only_completed_final(tmp_path):
    from tools.select_final_checkpoint import select_final_checkpoint
    config=configuration(tmp_path);model=Tiny(config);opt=torch.optim.AdamW(model.parameters(),lr=1e-5);sched=scheduler_for(opt,26,.1)
    model._training_phase="answer_only"
    save_checkpoint(model,opt,sched,tmp_path,6,3,config,[capture_rng()],[{}])
    with pytest.raises(ValueError,match="final checkpoint"):select_final_checkpoint(config,tmp_path)
    model._training_phase="joint"
    path=save_checkpoint(model,opt,sched,tmp_path,26,13,config,[capture_rng()],[{}])
    assert select_final_checkpoint(config,tmp_path)==path


def test_full_task_state_validation_rejects_nonfinite_without_partial_copy(tmp_path):
    from reason3d_runtime import restore_task_state
    model=Tiny(configuration(tmp_path));before=copy.deepcopy(model.state_dict());state=copy.deepcopy(before)
    state["linear.weight"].add_(5);state["linear.bias"][0]=float("nan")
    with pytest.raises(ValueError,match="tensor"):restore_task_state(model,state)
    for n,v in before.items():torch.testing.assert_close(model.state_dict()[n],v,rtol=0,atol=0)
