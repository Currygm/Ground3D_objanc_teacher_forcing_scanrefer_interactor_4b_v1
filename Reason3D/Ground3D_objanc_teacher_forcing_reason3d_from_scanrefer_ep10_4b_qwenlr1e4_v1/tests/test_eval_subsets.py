import copy
import importlib
import json
import sys
from collections import Counter
import pytest
from torch.utils.data import Dataset, ConcatDataset, Subset


class MetadataDataset(Dataset):
    def __init__(self, source, count, load=False):
        self.annotations=[{"type":source} for _ in range(count)];self.load=load
    def __len__(self):return len(self.annotations)
    def __getitem__(self,index):
        if not self.load:raise AssertionError("Grouping must not load scene tensors")
        return {"sample_id":str(index)}


def test_grouping_uses_annotation_type_and_respects_subsets():
    from reason3d_eval_subsets import source_indices
    dataset=ConcatDataset([MetadataDataset("scannet",2),MetadataDataset("mp3d",3)])
    assert source_indices(dataset)=={"scannet":[0,1],"mp3d":[2,3,4]}
    assert source_indices(Subset(dataset,[4,0,2]))=={"scannet":[1],"mp3d":[0,2]}
    assert source_indices(Subset(dataset,[0]))=={"scannet":[0],"mp3d":[]}
    with pytest.raises(ValueError,match="source"):source_indices(MetadataDataset("unknown",1))


def raw_payload(module,mode,indices):
    from reason3d_no_fallback import zero_metrics
    values=[.9,0.,.2,.4,.6,.8,0.,0.]
    if mode=="generated_no_fallback":
        rows=[]
        for i in indices:
            metrics=zero_metrics();metrics["obj/selected_iou"]=values[i]
            rows.append({"evaluation_index":i,"metrics":metrics,"valid":i not in (1,7),"failure_reason":"format_failure" if i in (1,7) else None,
                         "format_valid":i not in (1,7),"budget_forced":False,"obj_phrase_correct":i%2==0})
        return {"rows":rows}
    book=module.MetricBook()
    for i in indices:book.add("OBJ",values[i],values[i],values[i],0,0,0,1,{k:values[i] for k in (1,3,5,10)})
    return {"metrics":book.state_dict(),"answer_counts":{"samples":len(indices),"format_valid":len(indices),"obj_correct":sum(i%2==0 for i in indices),
            "anc_tp":sum(i%2 for i in indices),"anc_pred":sum(i+1 for i in indices),"anc_true":sum(i+2 for i in indices)},
            "fallback":{mode:len(indices)},"examples":[{"evaluation_index":i,"sample_id":str(i)} for i in indices]}


def test_distributed_merging_matches_original_totals_and_keeps_failures():
    from reason3d_eval_subsets import merge_source_payloads
    from reason3d_no_fallback import merge_no_fallback_payloads
    module=importlib.import_module("evaluate_reason3d")
    modes=["teacher","generated","generated_no_fallback"]
    gathered=[]
    for rank in range(4):
        gathered.append({mode:{source:raw_payload(module,mode,[i for i in range(rank,8,4) if (i<2)==(source=="scannet")]) for source in ["scannet","mp3d"]} for mode in modes})
    untouched=copy.deepcopy(gathered)
    overall,examples,by_dataset=merge_source_payloads(gathered,modes,{"scannet":2,"mp3d":6},module.merge_mode_payloads,merge_no_fallback_payloads,20)
    for mode in modes:
        merge=merge_no_fallback_payloads if mode=="generated_no_fallback" else module.merge_mode_payloads
        expected=merge([raw_payload(module,mode,list(range(rank,8,4))) for rank in range(4)],20)[0]
        assert overall[mode]==expected
    assert overall["generated_no_fallback"]["obj/selected_iou"]==pytest.approx(2.9/8)
    assert by_dataset["scannet"]["generated_no_fallback"]["obj/selected_iou"]==pytest.approx(.45)
    assert by_dataset["mp3d"]["generated_no_fallback"]["samples"]==6
    assert by_dataset["mp3d"]["generated_no_fallback"]["failure_rate"]==pytest.approx(1/6)
    assert gathered==untouched
    with pytest.raises(ValueError,match="coverage"):
        merge_source_payloads(gathered,modes,{"scannet":3,"mp3d":5},module.merge_mode_payloads,merge_no_fallback_payloads,20)


@pytest.mark.parametrize("entry",["evaluate_reason3d"])
def test_all_entry_points_write_overall_and_two_source_summaries(tmp_path,monkeypatch,entry):
    module=importlib.import_module(entry)
    config={"project":{"artifact_root":str(tmp_path)},"data":{"dataset_name":"reason3d","val_annotations":"validation.json","num_workers":0},"training":{"amp":False},"validation":{"mask_threshold":.5}}
    checkpoint=tmp_path/"checkpoint.pt";checkpoint.touch();output=tmp_path/"evaluation"
    class Model:
        def __init__(self,config):pass
        def to(self,device):return self
        def eval(self):return self
    calls=Counter()
    def evaluate(model,loader,mode,*args,**kwargs):
        indices=[]
        for batch in loader:
            index=int(batch["evaluation_index"][0]);indices.append(index);calls[mode,index]+=1
        return raw_payload(module,mode,indices)
    monkeypatch.setattr(module,"load_segmentation_config",lambda path:copy.deepcopy(config))
    monkeypatch.setattr(module,"distributed_context",lambda:(0,1,0,False))
    monkeypatch.setattr(module,"build_segmentation_dataset",lambda *args:ConcatDataset([MetadataDataset("scannet",2,True),MetadataDataset("mp3d",3,True)]))
    monkeypatch.setattr(module,"ObjAncSegmentationModel",Model)
    monkeypatch.setattr(module,"load_checkpoint",lambda *args:{"step":2184,"completed_epochs":13})
    monkeypatch.setattr(module.torch.cuda,"is_available",lambda:False)
    monkeypatch.setattr(module,"evaluate_mode",evaluate)
    monkeypatch.setattr(module,"evaluate_no_fallback",lambda model,loader,*args,**kwargs:evaluate(model,loader,"generated_no_fallback"))
    monkeypatch.setattr(sys,"argv",[entry,str(tmp_path/"config.yaml"),str(checkpoint),str(output)])
    module.main()
    summary=json.loads((output/"summary.json").read_text())
    assert summary["validation_samples"]==5
    assert summary["validation_samples_by_dataset"]=={"scannet":2,"mp3d":3}
    for source,count in [("scannet",2),("mp3d",3)]:
        subset=json.loads((output/f"summary_{source}.json").read_text())
        assert subset["validation_samples"]==count
        assert subset["generated_no_fallback"]["samples"]==count
        assert subset["teacher_answer"]==summary["by_dataset"][source]["teacher_answer"]
    assert len(calls)==15 and all(count==1 for count in calls.values())
