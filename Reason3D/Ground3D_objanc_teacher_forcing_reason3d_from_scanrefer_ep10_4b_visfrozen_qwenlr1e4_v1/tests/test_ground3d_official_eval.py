"""CPU tests for official Ground3D inference and matched scoring."""
import importlib.util
import pytest
import numpy as np

def api():
    assert importlib.util.find_spec("ground3d_official_eval"), "Official adapter missing"
    import ground3d_official_eval
    return ground3d_official_eval

def test_generated_prediction_finishes_before_truth_access():
    m=api(); events=[]
    class Dataset:
        rows=[dict(source="scannet",level="part_qa",sample_id="s",target_phrase="leg")]
        def inference(self,i):
            events.append("input")
            return dict(question="Where is the support?",scene_name="s_frame_0")
        def truth(self,i):
            events.append("truth")
            return np.array([True,False,True])
    def infer(x):
        assert set(x)=={"question","scene_name"}
        events.append("prediction")
        return dict(answer="<p>leg</p><SEG>",mask=np.array([True,False,True]),reason=None,query_count=1,mask_official04=np.array([True,True,True]))
    r,native=m.evaluate_both(Dataset(),0,infer)
    assert events==["input","prediction","truth"]
    assert r["iou"]==1 and r["valid"] and r["obj_phrase_correct"]
    assert native["iou"]==pytest.approx(2/3)

def test_no_phrase_is_zero_even_when_official_zero_query_predicts_full_mask():
    m=api()
    r=m.score_prediction(dict(answer="No answer",mask=np.ones(3,dtype=bool),reason=None,query_count=1),
                         np.array([True,False,True]),"leg")
    assert r["iou"]==0 and not r["valid"] and r["failure_reason"]=="missing_grounded_phrase"

def test_first_predicted_phrase_and_no_gt_based_matching():
    m=api()
    # A wrong phrase is scored as a valid prediction, and never rejected against GT.
    r=m.score_prediction(dict(answer="<p>table</p><SEG><p>leg</p><SEG>",mask=np.array([True,False,False]),reason=None,query_count=2),
                         np.array([False,True,True]),"leg")
    assert r["valid"] and r["iou"]==0 and not r["obj_phrase_correct"]

def test_missing_checkpoint_task_weight_fails_closed():
    m=api()
    import torch
    model={"decoder.query_proj.weight":torch.zeros(2,2),
           "decoder.reason_model.base_model.model.model.language_model.layers.0.self_attn.q_proj.base_layer.weight":torch.zeros(2,2)}
    with pytest.raises(ValueError,match="Missing task"):
        m.validate_state(model, {})
    report=m.validate_state(model,{"decoder.query_proj.weight":torch.ones(2,2)})
    assert len(report["allowed_base_missing"])==1
    with pytest.raises(ValueError,match="Unexpected"):
        m.validate_state(model,{**model,"bad":torch.ones(1)})

def test_partial_embedding_and_lora_mismatch_fail():
    m=api()
    import torch
    with pytest.raises(ValueError,match="Missing task"):
        m.validate_state({"decoder.reason_model.base_model.model.lm_head.weight":torch.zeros(3,2)}, {})
    with pytest.raises(ValueError,match="shape"):
        m.validate_state({"decoder.reason_model.x.lora_A.default.weight":torch.zeros(32,2)},
                         {"decoder.reason_model.x.lora_A.default.weight":torch.zeros(16,2)})

def test_invalid_sample_stays_in_summary_denominator():
    m=api()
    from ground3d_functional import summarize
    bad=m.score_prediction(dict(answer="",mask=None,reason=None,query_count=0),np.array([True]),"x")
    good=m.score_prediction(dict(answer="<p>x</p><SEG>",mask=np.array([True]),reason=None,query_count=1),np.array([True]),"x")
    rows=[dict(r,evaluation_index=i,sample_id=str(i),source="scannet",level="object_qa",mode="generated_no_fallback") for i,r in enumerate([bad,good])]
    result=summarize(rows,expected_indices=range(2),modes=["generated_no_fallback"])
    assert result["metrics"]["generated_no_fallback"]["samples"]==2
    assert result["metrics"]["generated_no_fallback"]["mIoU"]==.5

def test_spconv_layout_metadata_survives_prefix_normalization():
    m=api()
    from collections import OrderedDict
    import torch
    state=OrderedDict([("module.conv.weight",torch.arange(6.).reshape(2,3))])
    state._metadata=OrderedDict([("",{"version":1}),("module",{"version":1}),("module.conv",{"version":2})])
    assert hasattr(m,"normalize_state"), "SpConv metadata-preserving checkpoint normalizer missing"
    normalized=m.normalize_state(state)
    assert normalized._metadata["conv"]=={"version":2}
    class Versioned(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2,3))
        def _load_from_state_dict(self,state_dict,prefix,local_metadata,*args):
            assert local_metadata.get("version")==2, "SpConv would apply the legacy kernel permutation"
            return super()._load_from_state_dict(state_dict,prefix,local_metadata,*args)
    model=torch.nn.Module();model.conv=Versioned()
    model.load_state_dict(normalized,strict=True)
    assert torch.equal(model.conv.weight,state["module.conv.weight"])

def test_sparse_metadata_missing_or_non_v2_fails_before_weight_load():
    m=api()
    import torch
    import spconv.pytorch as spconv
    model=torch.nn.Sequential(spconv.SubMConv3d(3,4,3,bias=False))
    state=model.state_dict()
    assert hasattr(m,"validate_sparse_metadata"), "SpConv metadata validation missing"
    with pytest.raises(ValueError,match="SpConv.*metadata"):
        m.validate_sparse_metadata(model,dict(state))
    state._metadata["0"]["version"]=1
    with pytest.raises(ValueError,match="SpConv.*metadata"):
        m.validate_sparse_metadata(model,state)
    state._metadata["0"]["version"]=2
    normalized=m.normalize_state(state)
    assert normalized._metadata["0"]["version"]==2
    assert m.validate_sparse_metadata(model,normalized)==1
