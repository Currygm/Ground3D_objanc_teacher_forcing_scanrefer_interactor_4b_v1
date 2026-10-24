from types import SimpleNamespace
import torch
from torch import nn
from reason3d_no_fallback import infer_generated_no_fallback


def test_wrong_generated_obj_is_selected_without_any_gt_access():
    class Guarded(dict):
        def __getitem__(self,key):
            if key not in {"question","point_cloud"}:raise AssertionError("GT accessed during inference: "+key)
            return super().__getitem__(key)
    model=nn.Module();model.text_adapter=nn.Linear(2,2);model.visual_adapter=nn.Linear(2,2)
    model._encode_points=lambda batch:[SimpleNamespace(compressed="visual",superpoint_hidden=torch.ones(2,2),superpoint_xyz=torch.zeros(2,3))]
    calls=[]
    def replay(points,questions,answers,**kwargs):
        calls.append(answers);assert answers==["<obj>chair</obj>"]
        return SimpleNamespace(token_roles=[torch.tensor([0,1,1])])
    model.answer_lm=SimpleNamespace(generate_one=lambda *a,**kw:{"answer":"<obj>chair</obj>","format_failure":False,"budget_forced":False},encode_full_context=replay)
    model._positive_maps=lambda context,indices:([torch.ones(3,2)],torch.tensor([[[True,False,False]]]),torch.tensor([[True]]))
    class Decoder(nn.Module):
        def forward(self,*args):
            return {"pred_logits":torch.tensor([[[-2.,0.,0.],[2.,0.,0.]]]),"pred_masks":torch.zeros(1,2,2)}
    model.decoder=Decoder()
    result=infer_generated_no_fallback(model,Guarded(question=["What can I sit on?"],point_cloud=[torch.zeros(2,6)]))
    assert result["valid"] and result["selected_query"]==1 and len(calls)==1
