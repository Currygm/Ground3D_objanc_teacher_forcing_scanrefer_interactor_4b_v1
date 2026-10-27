import pytest
from evaluate_segmentation import MetricBook,merge_mode_payloads

def test_failed_raw_obj_contributes_zero_and_full_denominator():
    from evaluate_segmentation import add_failed_raw_obj
    b=MetricBook()
    b.add("OBJ",.8,.9,.95,0,0,0,1,{k:.95 for k in [1,3,5,10,50]},segmentation={"precision":1,"recall":1,"dice":1})
    for prefix in ('obj','all'):b._add_value(prefix+'/tie_aware_best_query_reciprocal_rank',1.)
    add_failed_raw_obj(b)
    s=b.summary()
    assert s["obj/selected_iou"]==pytest.approx(.4)
    assert s["obj/selected_acc@0.50"]==.5
    assert s["obj/qmatch"]==.5
    assert b.counts["obj/selected_iou"]==2
    assert b.counts["obj/tie_aware_best_query_reciprocal_rank"]==2
    assert s["obj/tie_aware_best_query_reciprocal_rank"]==.5
    assert s["obj/best_query_rank"]==26

def test_shard_summary_preserves_full_raw_zero_obj_subsets_and_counts():
    from evaluate_segmentation import add_failed_raw_obj
    strict=MetricBook();add_failed_raw_obj(strict)
    zero=MetricBook();zero.add("OBJ",.5,.5,.7,0,0,2,3,{k:.7 for k in [1,3,5,10,50]})
    payload={"metrics":zero.state_dict(),"visual_max":{"llm_tokens":32,"superpoints":2001},
       "answer_counts":{"samples":1,"format_valid":0},"fallback":{"zero_obj:format_failure":1},"examples":[],
       "raw_metrics":strict.state_dict(),"raw_counts":{"attempted":1,"evaluable":0,"format_unevaluable":1},
       "normal_metrics":MetricBook().state_dict(),"zero_obj_metrics":zero.state_dict()}
    s,_=merge_mode_payloads([payload],10)
    assert s["raw_generated_no_fallback"]["samples"]==1
    assert s["raw_generated_no_fallback"]["obj/selected_iou"]==0
    assert s["raw_generated_no_fallback"]["coverage"]==0
    assert s["generated_with_zero_obj"]["obj/selected_iou"]==.5
    assert s["zero_obj_subset"]["samples"]==1
    assert s["normal_generated_subset"]["samples"]==0
    assert s["teacher_fallbacks"]==0
