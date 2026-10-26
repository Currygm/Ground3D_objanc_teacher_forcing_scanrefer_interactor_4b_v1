import copy
import torch
from feedback_fixtures import source,batch
from region_feedback.proposals import FrozenProposalStage
from region_feedback.cache import make_record

def fixture():
    s=source();b=batch();p=FrozenProposalStage(s,"constrained").predict(b)[0]
    return s,make_record(p,b,0,s.answer_lm,index=0)

def test_second_teacher_completion_never_replaces_first_pass_evidence():
    from region_feedback.object_context import object_text_context
    from region_feedback.object_prompt import render_object_prompt
    s,r=fixture();projection=torch.nn.Sequential(torch.nn.LayerNorm(8),torch.nn.Linear(8,64))
    prompt,info=render_object_prompt(s.answer_lm,r,projection,"visual",task="segmentation")
    assert "Mark exactly one target" in info["system_text"]
    assert "red chair" not in info["user_text"]
    context,lengths,lm_loss=object_text_context(s.answer_lm,[r],projection,"visual",
        answers=["<obj>red chair</obj>"],synthetic=[False],task="segmentation",compute_lm_loss=True)
    assert context.parsed_answers[0].obj.phrase=="red chair"
    assert r["answer"].startswith("<obj>chair</obj>")
    assert torch.isfinite(lm_loss) and lm_loss.requires_grad

def test_actual_greedy_second_generation_uses_object_prompt_and_eos():
    from region_feedback.generation import generate_object_answer
    s,r=fixture();lm=s.answer_lm
    projection=torch.nn.Sequential(torch.nn.LayerNorm(8),torch.nn.Linear(8,64))
    answer="<obj>chair</obj> near <anc>table</anc>"
    expected=lm.tokenizer.encode(answer,add_special_tokens=False)+[lm.eos_token_id]
    state=[0];original=lm.structured.logits
    def force(hidden,head):
        logits=hidden.new_full((hidden.shape[0],lm.structured.vocab_size),-1000.)
        logits[:,expected[state[0]]]=1000.;state[0]+=1;return logits
    lm.structured.logits=force
    result=generate_object_answer(lm,r,projection,"visual",max_new_tokens=64)
    lm.structured.logits=original
    assert result["answer"]==lm.tokenizer.decode(expected[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    assert not result["format_failure"]
    assert result["generated_token_ids"]==expected and result["stop_reason"]=="eos"
    assert not result["format_constraints"]

def test_second_text_can_recover_from_invalid_first_pass():
    from region_feedback.object_context import object_text_context
    s,r=fixture();r["failed"]=True;r["answer"]="invalid first answer"
    projection=torch.nn.Sequential(torch.nn.LayerNorm(8),torch.nn.Linear(8,64))
    context,_=object_text_context(s.answer_lm,[r],projection,"visual",
        answers=["<obj>chair</obj>"],synthetic=[False],task="segmentation")
    assert context.entity_roles==[("OBJ",)]
    assert context.parsed_answers[0].valid
    assert context.answer_hidden[0].abs().sum()>0
