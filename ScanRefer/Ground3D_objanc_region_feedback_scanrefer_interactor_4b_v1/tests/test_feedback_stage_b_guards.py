import copy
import pytest
import torch

def test_stage_b_initialization_is_explicit_and_unambiguous():
    from region_feedback.runtime_v2 import validate_initialization
    with pytest.raises(ValueError):validate_initialization("b")
    with pytest.raises(ValueError):validate_initialization("b",resume="b.pt",stage_a_checkpoint="a.pt")
    with pytest.raises(ValueError):validate_initialization("a",from_first_stage=True)
    validate_initialization("a")
    validate_initialization("b",stage_a_checkpoint="a.pt")
    validate_initialization("b",from_first_stage=True)
    validate_initialization("b",resume="b.pt")

def test_teacher_context_padding_matches_single_rows_and_token_weighted_ce():
    from test_feedback_stage_b_model import build_b
    from region_feedback.object_context import object_text_context
    s,r,cfg,m=build_b();m.eval()
    other=copy.deepcopy(r);other["question"]="chair"
    answers=["<obj>chair</obj>","<obj>red chair</obj> near <anc>table</anc>"]
    batched,_,loss=object_text_context(m.language,[r,other],m.object_projection,"visual",
        answers=answers,synthetic=[False,False],task="segmentation",compute_lm_loss=True)
    expected=0.;tokens=0
    for i,record in enumerate([r,other]):
        single,_,value=object_text_context(m.language,[record],m.object_projection,"visual",
            answers=[answers[i]],synthetic=[False],task="segmentation",compute_lm_loss=True)
        assert torch.allclose(batched.question_hidden[i],single.question_hidden[0],atol=1e-5)
        assert torch.allclose(batched.answer_hidden[i],single.answer_hidden[0],atol=1e-5)
        count=len(m.language.tokenizer.encode(answers[i],add_special_tokens=False))+1
        expected=expected+value*count;tokens+=count
    assert torch.allclose(loss,expected/tokens,atol=1e-5)


def test_second_generation_keeps_raw_budget_failure_without_text_repair():
    from test_feedback_stage_b_model import build_b
    from region_feedback.generation import generate_object_answer
    s,r,cfg,m=build_b();m.eval()
    word=m.language.tokenizer.convert_tokens_to_ids("where")
    original=m.language.structured.logits
    def force(hidden,head):
        logits=hidden.new_full((hidden.shape[0],m.language.structured.vocab_size),-1000.)
        logits[:,word]=1000.;return logits
    m.language.structured.logits=force
    result=generate_object_answer(m.language,r,m.object_projection,"visual",max_new_tokens=1)
    m.language.structured.logits=original
    assert result["format_failure"] and result["stop_reason"]=="max_new_tokens"
    assert result["answer"]=="where" and result["generated_token_ids"]==[word]
