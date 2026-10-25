import copy
import importlib
import importlib.util
import torch
from feedback_fixtures import source,batch,config

def api(name):
    assert importlib.util.find_spec("region_feedback."+name) is not None, name+" implementation missing"
    return importlib.import_module("region_feedback."+name)


def test_proposals_use_all_predicted_anchors_and_ignore_annotation_changes():
    first=api("proposals").FrozenProposalStage(source(),generation_protocol="constrained")
    a=batch();b=copy.deepcopy(a);b["teacher_answer"]=["LEAK"]
    b["grounding_roles"]=[["ANC"]];b["grounding_phrases"]=[[["wrong"]]]
    x=first.predict(a)[0];y=first.predict(b)[0]
    assert x.entity_roles==("OBJ","ANC","ANC")
    assert x.entity_scores.shape==(50,3)
    assert torch.equal(x.masks,y.masks) and torch.equal(x.entity_scores,y.entity_scores)
    inference={k:a[k] for k in ["point_cloud","point_to_superpoint","superpoint_coord","question"]}
    z=first.predict(inference)[0]
    assert torch.equal(x.masks,z.masks)




def test_invalid_first_answer_uses_actual_text_and_zero_obj_without_gt():
    s=source();s.answer_lm.test_answer="where <|point|>"
    first=api("proposals").FrozenProposalStage(s,generation_protocol="constrained")
    p=first.predict(batch())[0]
    assert p.failed and p.answer=="where <|point|>"
    assert p.entity_roles==("OBJ",) and p.entity_scores.shape==(50,1)
    assert torch.isfinite(p.masks).all()
