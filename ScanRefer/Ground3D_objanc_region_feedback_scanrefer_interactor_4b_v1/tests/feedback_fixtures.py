from types import MethodType,SimpleNamespace
import torch
from torch import nn
from test_ground3d_qwen_causal import tiny_lm
from ground3d_migration.visual import GroundVisualTokens
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer

def generate(self,question,point_cloud=None,*,visual=None):
    answer=getattr(self,"test_answer","<obj>chair</obj> near <anc>table</anc> and <anc>door</anc>")
    parsed=parse_answer(answer)
    return {"answer":answer,"parsed":parsed,"format_failure":not parsed.valid,"budget_forced":False}

def encode(self,batch):
    self.encode_calls+=1
    output=[]
    for cloud,xyz in zip(batch["point_cloud"],batch["superpoint_coord"]):
        f=self.frontend(cloud)
        prefix=self.scene_projection(f.mean(0,keepdim=True)).expand(3,-1)
        output.append(SimpleNamespace(superpoint_hidden=f,superpoint_xyz=xyz,
                                      compressed=GroundVisualTokens(prefix,None)))
    return output

def source():
    torch.manual_seed(27)
    model=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(model)
    model.answer_lm=tiny_lm()
    model.answer_lm.generate_one=MethodType(generate,model.answer_lm)
    model.text_adapter=nn.Linear(64,16);model.visual_adapter=nn.Linear(8,8)
    model.frontend=nn.Linear(6,8);model.scene_projection=nn.Linear(8,64)
    model.decoder=ObjAncFullCycleDecoder(visual_dim=8,text_dim=16,hidden_dim=16,
        num_queries=50,num_layers=9,num_heads=4,feedforward_dim=32,max_text_tokens=384)
    model.loss_cfg={"span_weight":6.,"mask_bce_weight":5.,"dice_weight":2.,"focal_alpha":.25,"focal_gamma":2.,"aux_weight":.5}
    model.matcher_cfg={"span_weight":2.,"mask_bce_weight":15.,"dice_weight":6.}
    model.ranking_cfg={"temperature":.1,"weight":.5}
    model.mask_threshold=.5;model.answer_lm_weight=.1;model.encode_calls=0
    model._encode_points=MethodType(encode,model)
    model.config={"loss":model.loss_cfg,"matcher":model.matcher_cfg,"ranking":model.ranking_cfg}
    return model

def batch():
    cloud=torch.tensor([[0.,0.,0.,1.,0.,0.],[1.,0.,0.,0.,1.,0.],[2.,0.,0.,0.,0.,1.],[3.,0.,0.,1.,1.,0.]])
    return {"point_cloud":[cloud],"point_to_superpoint":[torch.arange(4)],
        "superpoint_coord":[cloud[:,:3]],"question":["where is chair near table"],
        "teacher_answer":["<obj>chair</obj>"],"grounding_object_ids":[[0]],
        "grounding_roles":[["OBJ"]],"grounding_phrases":[[["chair"]]],
        "answer_entity_indices":[[0]],"gt_point_masks":[torch.tensor([[1.,1.,0.,0.]])],
        "gt_superpoint_masks":[torch.tensor([[1.,1.,0.,0.]])],
        "sample_id":["fixture:0"],"scene_id":["fixture"]}

def config(stage="a",mode="visual_geometry"):
    return {"feedback":{"stage":stage,"evidence_mode":mode,"input_dim":8,"hidden_dim":16,
        "language_dim":64,"num_queries":50,"num_heads":4,"region_refinement_layers":0,
        "prior_dropout":0.,"prior_corruption":0.,"ranking_temperature":.1,
        "ranking_weight":.5,"answer_lm_weight":.1},
        "optimizer":{"new_lr":1e-3,"qwen_lora_lr":1e-3,"structured_token_lr":1e-3,
                     "decoder_lr":1e-3,"weight_decay":0.}}
