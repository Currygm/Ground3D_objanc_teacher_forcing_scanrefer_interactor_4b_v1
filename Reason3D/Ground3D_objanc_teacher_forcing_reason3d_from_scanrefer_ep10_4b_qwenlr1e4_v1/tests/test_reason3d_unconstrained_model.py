import torch
import pytest
from torch import nn
from types import SimpleNamespace
from test_ground3d_qwen_causal import tiny_lm as original_tiny_lm
from reason3d_unconstrained_model import install_language, ObjAncSegmentationModel

def tiny_lm():
    lm=original_tiny_lm(); install_language(lm); return lm
from ground3d_migration.visual import GroundVisualTokens
from spatiallm_compressed_visual_objanc_full_attention.model.llm import ROLE_OBJ,ROLE_OTHER,ROLE_QUESTION

@pytest.mark.parametrize("answer",["", "no tagged object", "<obj>chair", "<obj>chair</obj><obj>table</obj>"])
def test_failed_answer_appends_zero_own_hidden_and_single_obj(answer):
    lm=tiny_lm()
    calls=[]
    hook=lm.language.get_base_model().model.language_model.register_forward_hook(lambda *args:calls.append(1))
    out=lm._causal(["where is chair"],[answer],[GroundVisualTokens(torch.randn(32,64),None)],synthetic_obj=[True],compute_loss=False)
    hook.remove()
    c=out["context"];n=len(lm.tokenizer.encode(answer,add_special_tokens=False));q=len(c.question_hidden[0])
    assert len(calls)==1 and c.answer_hidden[0].shape==(n+1,64)
    assert torch.equal(c.answer_hidden[0][-1],torch.zeros(64))
    assert c.token_roles[0].tolist()==[ROLE_QUESTION]*q+[ROLE_OTHER]*n+[ROLE_OBJ]
    assert c.entity_roles==[("OBJ",)]
    assert c.phrase_spans[0].shape==(1,q+n+1)
    assert torch.where(c.phrase_spans[0][0])[0].tolist()==[q+n]
    assert not c.parsed_answers[0].valid

def test_wrong_but_valid_phrase_does_not_read_teacher_or_fallback():
    m=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(m)
    class LM(nn.Module):
        def generate_one(self,*args,**kwargs):
            return {"answer":"<obj>table</obj>","format_failure":False,"budget_forced":False}
    m.answer_lm=LM()
    batch={"question":["where"],"sample_id":["x"],"grounding_roles":[["OBJ"]],"grounding_phrases":[[["chair"]]]}
    result=m._choose_answers(batch,[SimpleNamespace(compressed=None)],"generated")
    assert result[0]==["<obj>table</obj>"] and result[1]==[[0]] and result[3]==["generated"]

def test_format_failure_preserves_actual_text_without_teacher():
    m=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(m)
    class LM(nn.Module):
        def generate_one(self,*args,**kwargs):
            return {"answer":"","format_failure":True,"budget_forced":True}
    m.answer_lm=LM()
    result=m._choose_answers({"question":["where"],"sample_id":["x"]},[SimpleNamespace(compressed=None)],"generated")
    assert result[0]==[""] and result[1:3]==([[0]],[[0]])
    assert result[3]==["zero_obj:budget_failure"]

@pytest.mark.parametrize("answer",["","<obj>table</obj>"])
def test_real_tiny_qwen_to_mask_decoder_no_gt_replay_once(answer):
    from pathlib import Path
    import yaml
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
    lm=tiny_lm()
    lm.generate_one=lambda *a,**k:{"answer":answer,"format_failure":not bool(answer),"budget_forced":False}
    m=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(m)
    c=yaml.safe_load((Path(__file__).resolve().parent.parent/"configs/reason3d_answer_lm3_objce3_warmup3_joint15.yaml").read_text())
    m.config=c;m.answer_lm=lm;m.text_adapter=nn.Linear(64,2048);m.visual_adapter=nn.Linear(256,512)
    m.decoder=ObjAncFullCycleDecoder(visual_dim=512,text_dim=2048,hidden_dim=16,num_queries=50,num_layers=9,num_heads=4,feedforward_dim=32)
    m.loss_cfg=c["loss"];m.matcher_cfg=c["matcher"];m.ranking_cfg=c["ranking"]
    m.mask_threshold=.5;m.answer_lm_weight=.1;m.obj_ce_weight=3.
    full=torch.randn(3,256)
    rows=[SimpleNamespace(compressed=GroundVisualTokens(torch.randn(32,64),None),superpoint_hidden=full,superpoint_xyz=torch.randn(3,3))]
    calls=[]
    def encode(batch):calls.append("visual");return rows
    m._encode_points=encode
    m.eval()
    batch={"point_cloud":[torch.zeros(3,6)],"question":["where is chair"],"sample_id":["x"],
        "grounding_roles":[["OBJ"]],"grounding_phrases":[[["chair"]]],
        "gt_superpoint_masks":[torch.tensor([[1.,0.,1.]])],"gt_point_masks":[torch.tensor([[1.,0.,1.]])],
        "point_to_superpoint":[torch.arange(3)]}
    out=m(batch,epoch=0,optimizer_step=0,steps_per_epoch=1,answer_mode="generated")
    assert out["chosen_answers"]==[answer] and out["synthetic_obj"]==[not bool(answer)]
    assert calls==["visual"] and lm.call_counts=={"causal_forward":1}
    assert out["raw_logits"][0].shape==(3,) and torch.isfinite(out["raw_logits"][0]).all()
    assert out["metrics"]["interactor_calls"]==0
    assert out["metrics"]["llm_visual_tokens"]==32
    assert out["query_scores"].shape==(1,50)
    assert out["losses"]["answer_lm"]==0

def test_install_keeps_weights_and_native_no_interactor_visual_path():
    from ground3d_migration.language import GroundObjAncLM
    from reason3d_runtime import ObjAncSegmentationModel as Original
    lm=original_tiny_lm()
    before={k:v.clone() for k,v in lm.state_dict().items()}
    visual=lm.visual
    native_encoder=lm.encode_batch_points.__func__
    install_language(lm)
    assert lm.visual is visual
    assert lm.encode_batch_points.__func__ is native_encoder
    assert lm.encode_batch_points.__func__ is GroundObjAncLM.encode_batch_points
    assert set(lm.state_dict())==set(before)
    assert all(torch.equal(v,lm.state_dict()[k]) for k,v in before.items())
    assert ObjAncSegmentationModel.architecture_signature is Original.architecture_signature
    assert ObjAncSegmentationModel.trainable_state_dict is Original.trainable_state_dict

def test_teacher_replay_and_losses_unchanged():
    torch.manual_seed(23)
    lm=original_tiny_lm()
    visuals=[GroundVisualTokens(torch.randn(7,64),None)]
    question=["where is red chair"]; answer=["<obj>red chair</obj> near <anc>table</anc>"]
    before=lm._causal(question,answer,visuals)
    install_language(lm)
    after=lm._causal(question,answer,visuals)
    assert torch.equal(before["loss"],after["loss"])
    assert torch.equal(before["losses"]["answer_obj_ce"],after["losses"]["answer_obj_ce"])
    assert torch.equal(before["context"].answer_hidden[0],after["context"].answer_hidden[0])
    assert torch.equal(before["context"].question_hidden[0],after["context"].question_hidden[0])
