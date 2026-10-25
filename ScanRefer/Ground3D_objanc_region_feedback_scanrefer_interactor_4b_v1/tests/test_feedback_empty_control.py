import copy
import torch
from test_feedback_v2_model import setup
from region_feedback.model import FeedbackModel
from region_feedback.object_context import object_text_context

def test_empty_control_hides_entity_words_in_replayed_assistant_answer():
    s,r,cfg=setup(mode="empty")
    m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder).eval()
    other=copy.deepcopy(r)
    other["answer"]=r["answer"].replace("chair","door")
    other["evidence"]["phrases"]=("door",)+r["evidence"]["phrases"][1:]
    a,_=object_text_context(m.language,[r],m.object_projection,"empty")
    b,_=object_text_context(m.language,[other],m.object_projection,"empty")
    assert a.answer_hidden[0].shape==b.answer_hidden[0].shape
    assert torch.allclose(a.question_hidden[0],b.question_hidden[0],atol=1e-6)
    assert torch.allclose(a.answer_hidden[0],b.answer_hidden[0],atol=1e-6)
    text_a,_=object_text_context(m.language,[r],m.object_projection,"text")
    text_b,_=object_text_context(m.language,[other],m.object_projection,"text")
    assert not torch.allclose(text_a.answer_hidden[0],text_b.answer_hidden[0])
