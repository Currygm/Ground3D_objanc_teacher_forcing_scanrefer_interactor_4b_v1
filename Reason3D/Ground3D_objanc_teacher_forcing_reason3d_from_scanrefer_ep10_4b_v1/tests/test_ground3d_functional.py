"""CPU contract checks for Ground3D functional adaptation."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

def api():
    assert importlib.util.find_spec("ground3d_functional") is not None, "Ground3D functional adapter is not implemented"
    import ground3d_functional
    return ground3d_functional

def example():
    return {"question":"Where can I write?", "answer":"Use the <p>desk surface</p><SEG>.", "object_id":[1]}

def test_format_and_label_spaces():
    m=api()
    for level,field in [("object_qa","object_label"),("part_qa","part_label")]:
        row=m.convert_qa(example(),source="scannet",scene="scene0000_00",frame=0,level=level,source_file="/source.json",entity_key="1",qa_index=0)
        assert row["teacher_answer"]=="<obj>desk surface</obj>"
        assert row["target_label_field"]==field
        assert row["target_ids"]==[1]
        assert row["original_answer"]==example()["answer"]
        assert row["question"]==example()["question"]
    q=example();q["object_id"]=[0]
    assert m.convert_qa(q,source="scannet",scene="s",frame=0,level="object_qa",source_file="f",entity_key="0",qa_index=0)["target_ids"]==[0]

def test_refuse_ambiguous_or_unsafe_answers():
    m=api()
    for answer,ids in [("No object.",[]),("<p>a</p><SEG> <p>b</p><SEG>",[1,2]),("<p></p><SEG>",[1]),("<p><obj>x</obj></p><SEG>",[1])]:
        q={"question":"q","answer":answer,"object_id":ids}
        with pytest.raises(ValueError):
            m.convert_qa(q,source="scannet",scene="s",frame=0,level="part_qa",source_file="f",entity_key="1",qa_index=0)

def write_frame(root):
    frame=root/"scannet"/"processed_data"/"s";frame.mkdir(parents=True)
    np.save(frame/"frame_0_vert.npy",np.array([[0,0,0,255,0,0],[1,0,0,0,255,0],[2,0,0,0,0,255]],dtype=np.float32))
    np.save(frame/"frame_0_sp_label.npy",np.array([10,10,30]))
    np.save(frame/"frame_0_object_label.npy",np.array([1,1,9]))
    np.save(frame/"frame_0_part_label.npy",np.array([2,1,2]))
    return frame

def test_native_superpoints_and_part_truth_are_separate(tmp_path):
    m=api();write_frame(tmp_path)
    row=m.convert_qa(example(),source="scannet",scene="s",frame=0,level="part_qa",source_file="f",entity_key="1",qa_index=0)
    m.validate_frame(row,tmp_path)
    dataset=m.FunctionalDataset([row],data_root=tmp_path)
    sample=dataset[0]
    assert set(sample)=={"point_cloud","ground_geometry","point_to_superpoint","superpoint_coord","question","sample_id"}
    assert sample["point_to_superpoint"].tolist()==[0,0,1]
    assert sample["ground_geometry"]["superpoint_values"].tolist()==[10,30]
    assert dataset.truth(0).tolist()==[False,True,False]
    np.save(tmp_path/"scannet/processed_data/s/frame_0_part_label.npy",np.array([2,2,2]))
    with pytest.raises(ValueError,match="target_has_no_points"):m.validate_frame(row,tmp_path)

def test_prediction_precedes_gt_and_failures_count_in_denominator():
    m=api();events=[]
    class FakeData:
        rows=[{"source":"scannet","level":"part_qa","sample_id":"s","teacher_answer":"<obj>desk</obj>","target_phrase":"desk"}]
        def __getitem__(self,i):
            events.append("visual")
            return {"question":"q","point_cloud":None,"sample_id":"s"}
        def truth(self,i):
            import torch
            events.append("truth")
            return torch.tensor([True,False])
    def predict(model,batch):
        assert "teacher_answer" not in batch and "gt_point_masks" not in batch
        events.append("predict")
        return {"valid":False,"answer":"","reason":"format_failure","format_valid":False,"budget_forced":False,"selected_query":None}
    row=m.evaluate_sample(None,FakeData(),0,"generated_no_fallback",predict=predict)
    assert events==["visual","predict","truth"]
    assert row["iou"]==0 and not row["valid"]
    good={**row,"valid":True,"iou":0.8,"acc25":1.,"acc50":1.}
    summary=m.aggregate([row,good])
    assert summary["samples"]==2 and summary["mIoU"]==pytest.approx(.4)
    assert summary["Acc@0.50"]==pytest.approx(.5)
    assert summary["failure_rate"]==pytest.approx(.5)

def test_evaluation_cli_help_without_loading_model():
    api()
    import subprocess,sys
    script=Path(__file__).resolve().parents[1]/"evaluate_ground3d_functional.py"
    assert script.is_file(), "Ground3D evaluator is not implemented"
    p=subprocess.run([sys.executable,"-B",str(script),"--help"],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert "generated_no_fallback" in p.stdout and "--checkpoint" in p.stdout

def test_four_way_summary_and_exact_coverage():
    m=api()
    row={"evaluation_index":0,"source":"scannet","level":"object_qa","mode":"generated_no_fallback",
        "iou":.8,"acc25":1.,"acc50":1.,"valid":True,"failure_reason":None,"format_valid":True,"obj_phrase_correct":True}
    assert hasattr(m,"summarize"), "Ground3D subset summary is not implemented"
    summary=m.summarize([row],expected_indices=[0],modes=["generated_no_fallback"])
    assert summary["by_dataset"]["scannet"]["generated_no_fallback"]["mIoU"]==pytest.approx(.8)
    assert summary["by_level"]["object_qa"]["generated_no_fallback"]["samples"]==1
    with pytest.raises(ValueError,match="coverage"):m.summarize([row,row],expected_indices=[0],modes=["generated_no_fallback"])

def test_generated_phrase_mismatch_does_not_trigger_gt_rejection():
    m=api()
    import torch
    class Data:
        rows=[{"source":"scannet","level":"object_qa","sample_id":"s","target_phrase":"desk"}]
        def __getitem__(self,i):
            return {"question":"q","point_to_superpoint":torch.tensor([0,0,1])}
        def truth(self,i):return torch.tensor([True,True,False])
    def prediction(model,batch):
        return {"valid":True,"answer":"<obj>chair</obj>","reason":None,"format_valid":True,
            "selected_query":0,"superpoint_logits":torch.tensor([[20.,-20.],[-20.,20.]])}
    result=m.evaluate_sample(None,Data(),0,predict=prediction)
    assert result["valid"] and result["iou"]==1.
    assert not result["obj_phrase_correct"]

def test_prepare_fast_path_still_validates_target_labels(tmp_path):
    m=api();write_frame(tmp_path)
    row=m.convert_qa(example(),source="scannet",scene="s",frame=0,level="object_qa",source_file="f",entity_key="1",qa_index=0)
    assert m.validate_frame(row,tmp_path,check_values=False)["points"]==3
    row["target_ids"]=[99]
    with pytest.raises(ValueError,match="target_has_no_points"):
        m.validate_frame(row,tmp_path,check_values=False)

def test_parallel_preparation_is_deterministic_and_read_only(tmp_path):
    m=api();write_frame(tmp_path)
    source=tmp_path/"scannet";(source/"part_ground3d_val.txt").write_text("s\n")
    files=[]
    for level,task in [("object_qa","functional_object_grounding"),("part_qa","functional_part_grounding")]:
        folder=source/"refined_qa_data"/level/"s"/"final_json_outputs";folder.mkdir(parents=True)
        path=folder/("0_object_mapping.json" if level=="object_qa" else "0_part_mapping.json")
        path.write_text(json.dumps({"1":{task:[example()]}}));files.append((path,path.read_bytes()))
    serial=m.prepare("val",["scannet"],data_root=tmp_path,output=tmp_path/"serial",workers=1)
    parallel=m.prepare("val",["scannet"],data_root=tmp_path,output=tmp_path/"parallel",workers=4)
    assert serial["samples"]==parallel["samples"]==2
    assert serial["annotation_sha256"]==parallel["annotation_sha256"]
    assert serial["counts"]==parallel["counts"]
    assert all(path.read_bytes()==raw for path,raw in files)

def test_teacher_truncated_obj_span_is_a_zero_score_failure():
    m=api()
    import torch
    from types import SimpleNamespace
    class Model:
        text_adapter=torch.nn.Linear(2,2)
        visual_adapter=torch.nn.Linear(2,2)
        def _encode_points(self,b):
            return [SimpleNamespace(compressed=None,superpoint_hidden=torch.zeros(1,2),superpoint_xyz=torch.zeros(1,3))]
        answer_lm=SimpleNamespace(encode_full_context=lambda *a,**kw:SimpleNamespace(token_roles=torch.zeros(1,1,dtype=torch.long)))
        def _positive_maps(self,c,idx):
            return [torch.zeros(2,2)],torch.tensor([[[0.,1.]]]),torch.ones(1,1,dtype=torch.bool)
        def decoder(self,*a,**kw):
            return {"pred_logits":torch.zeros(1,2,1),"pred_masks":torch.zeros(1,2,1)}
    prediction=m.infer_teacher(Model(),{"point_cloud":[torch.zeros(1,6)],"question":["q"]},"<obj>desk</obj>")
    assert not prediction["valid"] and prediction["reason"]=="obj_span_truncated"
