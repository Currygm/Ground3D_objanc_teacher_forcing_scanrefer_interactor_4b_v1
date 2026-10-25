import copy
import torch
from feedback_fixtures import source,batch
from region_feedback.proposals import FrozenProposalStage

def test_evidence_is_one_per_predicted_entity_not_query():
    from region_feedback.objects import select_object_evidence
    first=FrozenProposalStage(source(),"constrained")
    p=first.predict(batch())[0]
    evidence=select_object_evidence(p)
    assert evidence.features.shape==(3,8)
    assert evidence.roles==("OBJ","ANC","ANC")
    assert evidence.phrases==("chair","table","door")
    assert torch.equal(evidence.source_query_ids,p.entity_scores.argmax(0))
    assert len(p.masks)==50
    p.entity_scores[:]=0;p.entity_scores[7,:]=10
    repeated=select_object_evidence(p)
    assert repeated.source_query_ids.tolist()==[7,7,7]

def test_invalid_and_empty_objects_never_synthesize_gt_evidence():
    from region_feedback.objects import select_object_evidence
    s=source();s.answer_lm.test_answer="bad <|point|>"
    first=FrozenProposalStage(s,"constrained")
    p=first.predict(batch())[0]
    e=select_object_evidence(p)
    assert len(e.features)==0 and not e.roles
    s=source();p=FrozenProposalStage(s,"constrained").predict(batch())[0]
    p.masks.fill_(-100)
    e=select_object_evidence(p)
    assert len(e.features)==3 and not e.valid_mask.any()
    assert torch.isfinite(e.features).all()
    p2=copy.deepcopy(p);p2.answer="<obj>chair</obj>"
    try:select_object_evidence(p2)
    except ValueError:pass
    else:raise AssertionError("entity binding mismatch must fail")
