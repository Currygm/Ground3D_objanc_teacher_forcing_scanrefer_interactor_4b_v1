from pathlib import Path
import copy,json,math
from types import SimpleNamespace
import pytest,torch,yaml
from torch import nn
from reason3d_runtime import load_segmentation_config,OBJECTIVE
from aux_loss_contract import parent_signature_for_loading
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml"
BASELINE=ROOT.parent/"Ground3D_objanc_teacher_forcing_reason3d_from_scanrefer_ep10_4b_v1"

def test_only_training_change_is_aux_weight_and_new_objective_paths():
    c=load_segmentation_config(CONFIG)
    old=yaml.safe_load((BASELINE/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml").read_text())
    for key in ("data","model","point_encoder","decoder","matcher","ranking","finetune_initialization","initialization","answer_policy"):
        assert c[key]==old[key],key
    assert c["loss"]=={**old["loss"],"aux_weight":.1}
    t=dict(c["training"]);ot=dict(old["training"]);t.pop("objective");ot.pop("objective")
    assert t==ot and t["lr_schedule"]=="continuous_13_epochs"
    assert not any(k.startswith("joint_") and k not in ot for k in t)

def test_parent_full_signature_preserves_every_other_field():
    parent=json.loads((ROOT/"docs/source_checkpoint.json").read_text())["architecture_signature"]
    native=copy.deepcopy(parent);native["loss"]["aux_weight"]=.1
    assert parent_signature_for_loading(native)==parent
    assert native["loss"]["aux_weight"]==.1 and parent["loss"]["aux_weight"]==.5
    native["decoder"]["num_queries"]+=1
    assert parent_signature_for_loading(native)!=parent

def test_actual_auxiliary_sum_and_gradients_are_point_one():
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
    m=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(m)
    c=load_segmentation_config(CONFIG)
    m.loss_cfg=c["loss"];m.ranking_cfg=c["ranking"];m.answer_lm_weight=3.
    primary=torch.tensor(2.,requires_grad=True);ranking=torch.tensor(4.,requires_grad=True);lm=torch.tensor(5.,requires_grad=True)
    layers=[torch.tensor(float(i),requires_grad=True) for i in range(1,9)]
    total,aux=m._total_loss(primary,layers,ranking,lm)
    assert aux.item()==36 and total.item()==pytest.approx(22.6)
    total.backward()
    assert primary.grad.item()==1 and ranking.grad.item()==.5 and lm.grad.item()==3
    assert all(x.grad.item()==pytest.approx(.1) for x in layers)

def test_actual_scheduler_still_uses_original_thirteen_epoch_cosine():
    from train_reason3d import scheduler_for
    p=nn.Parameter(torch.ones(1));o=torch.optim.AdamW([p],lr=1e-5);s=scheduler_for(o,2184,.1)
    for step in (0,100,503,504,604,1344,2184):
        expected=.1+.9*.5*(1+math.cos(math.pi*step/2184))
        assert s.lr_lambdas[0](step)==pytest.approx(expected)

def test_native_checkpoint_roundtrip_keeps_aux01_and_rejects_aux05(tmp_path):
    from train_reason3d import save_checkpoint,load_resume,scheduler_for
    from evaluate_reason3d_unconstrained import load_checkpoint
    from ground3d_migration.runtime import capture_rng
    class Toy(nn.Module):
        def __init__(self,c):
            super().__init__();self.weight=nn.Parameter(torch.tensor([.7],dtype=torch.float64));self.config=c
            self._training_phase="answer_only"
            self.answer_lm=SimpleNamespace(source_manifests={},tokenizer_snapshot=lambda:{"native":"tokenizer"})
        def architecture_signature(self):return {"objective":OBJECTIVE,"loss":dict(self.config["loss"])}
        def trainable_state_dict(self):return {k:v.detach().cpu() for k,v in self.state_dict().items()}
    c=load_segmentation_config(CONFIG);m=Toy(c);o=torch.optim.AdamW(m.parameters(),lr=1e-5);s=scheduler_for(o,2184,.1)
    for _ in range(504):m.weight.grad=torch.ones_like(m.weight);o.step();o.zero_grad(set_to_none=True);s.step()
    path=save_checkpoint(m,o,s,tmp_path,504,3,c,[capture_rng()])
    restored=Toy(c);ro=torch.optim.AdamW(restored.parameters(),lr=1e-5);rs=scheduler_for(ro,2184,.1)
    step,epoch,states=load_resume(restored,ro,rs,path,1)
    assert (step,epoch)==(504,3) and rs.last_epoch==504 and len(states)==1
    assert int(ro.state[restored.weight]["step"])==504 and torch.equal(m.weight,restored.weight)
    assert load_checkpoint(Toy(c),path)["architecture_signature"]["loss"]["aux_weight"]==.1
    wrong=copy.deepcopy(c);wrong["loss"]["aux_weight"]=.5
    with pytest.raises(ValueError):load_checkpoint(Toy(wrong),path)
