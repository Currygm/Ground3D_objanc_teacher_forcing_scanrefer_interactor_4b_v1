from types import SimpleNamespace
import torch
from torch import nn
import pytest
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel

def stage():
    model=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel); nn.Module.__init__(model)
    return model

def test_raw_generated_preserves_wrong_obj_without_teacher():
    m=stage()
    raw=getattr(m,'_choose_raw_answers',None)
    assert callable(raw),'raw generated conditioning is missing'
    batch={'teacher_answer':['<obj>chair</obj>'],'grounding_roles':[['OBJ','ANC']],
           'grounding_phrases':[[['chair'],['desk']]]}
    answer='<obj>table</obj> next to <anc>desk</anc>'
    answers,rows,indices,reasons,attempted,ratio=raw(batch,[answer])
    assert answers==[answer] and answers!=batch['teacher_answer']
    assert rows==[[0,1]] and indices==[[0,1]]
    assert reasons==['raw_generated']

def test_malformed_raw_answer_is_explicitly_unevaluable():
    m=stage(); raw=getattr(m,'_choose_raw_answers',None)
    assert callable(raw)
    with pytest.raises(ValueError):raw({'grounding_roles':[['OBJ']],'grounding_phrases':[[['chair']]]},['no tagged object'])

def test_prediction_metrics_and_exact_eval_sampler():
    import evaluate_segmentation as ev
    book=ev.MetricBook()
    book.add('OBJ',.5,.5,.8,1,1,2,2,{1:.5,3:.8,5:.8,10:.8,50:.8},
             segmentation={'precision':.5,'recall':1.,'dice':2/3})
    result=book.summary()
    assert result['obj/selected_acc@0.50']==1
    assert result['obj/best_iou_query_recall']==0
    assert result['obj/precision']==.5
    assert result['obj/recall50_top3']==1
    shards=[list(ev.ExactDistributedEvalSampler(range(7),r,4)) for r in range(4)]
    assert sorted(x for shard in shards for x in shard)==list(range(7))

