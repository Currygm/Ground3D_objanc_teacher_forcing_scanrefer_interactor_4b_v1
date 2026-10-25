import torch
from region_feedback.object_prompt import object_prompt_hidden,render_object_prompt
from region_feedback.model import FeedbackModel
from test_feedback_v2_model import setup

def test_real_qwen_object_prefix_padding_matches_individual_readout():
    s,r,cfg=setup();m=FeedbackModel(s.answer_lm,cfg,text_adapter=s.text_adapter,decoder=s.decoder).eval()
    import copy
    short=copy.deepcopy(r);short["question"]="chair";short["answer"]="<obj>chair</obj>"
    # A literal point marker in generated text is never mistaken for an insertion slot.
    r["answer"]+=" <|point|>"
    a,lengths=object_prompt_hidden(m.language,[r,short],m.object_projection,"visual")
    for i,item in enumerate([r,short]):
        b,_=object_prompt_hidden(m.language,[item],m.object_projection,"visual")
        assert torch.allclose(a[i],b[0],atol=1e-5)
    a.square().mean().backward()
    assert m.object_projection[-1].weight.grad.abs().sum()>0
    assert lengths[0]>lengths[1]
