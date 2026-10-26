"""Full second-pass UniVLG9 segmentation with one Qwen and predicted-object priors."""
import torch
from torch import nn
from .model import PROTOCOL,FeedbackModel
from .object_context import object_text_context
from .object_prompt import SEGMENTATION_PROMPT_PROTOCOL
from .generation import generate_object_answer
from .proposals import predicted_entity_scores
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel

class SegmentationObjective:
    _positive_maps=staticmethod(ObjAncSegmentationModel._positive_maps)
    _layer_loss=ObjAncSegmentationModel._layer_loss
    _ranking=ObjAncSegmentationModel._ranking
    def __init__(self,config,feedback):
        self.loss_cfg=dict(config["loss"]);self.matcher_cfg=dict(config["matcher"])
        self.ranking_cfg={**config["ranking"],"temperature":feedback["ranking_temperature"]}
        self.mask_threshold=float(config.get("validation",{}).get("mask_threshold",.5))

class StageBFeedbackModel(nn.Module):
    objective="ground3d_object_feedback_shared_qwen_stage_b_v4"
    def __init__(self,language,config,*,decoder,text_adapter,visual_adapter,objective_config):
        super().__init__()
        cfg=config["feedback"]
        if cfg.get("stage")!="b" or cfg.get("protocol")!=PROTOCOL or cfg.get("method")!="qwen_feedback":
            raise ValueError("Stage B requires the shared-Qwen object-feedback reader")
        if cfg["evidence_mode"] not in {"empty","text","visual"}:raise ValueError("Invalid Stage B evidence")
        self.config=config;self.method="qwen_feedback";self.mode=cfg["evidence_mode"]
        self.language=language.requires_grad_(False)
        for name,p in self.language.named_parameters():
            p.requires_grad_(name.startswith("language.") and "lora_" in name and "visual" not in name)
        if not any(p.requires_grad for p in self.language.parameters()):raise ValueError("Existing language LoRA is required")
        self.object_projection=nn.Sequential(nn.LayerNorm(cfg["input_dim"]),nn.Linear(cfg["input_dim"],cfg["language_dim"]))
        self.decoder=decoder.requires_grad_(True)
        self.text_adapter=text_adapter.requires_grad_(True)
        self.visual_adapter=visual_adapter.requires_grad_(True)
        self.segmentation=SegmentationObjective(objective_config,cfg)
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        self.language.train(mode);self.language.visual.eval()
        if hasattr(self.language,"interactor"):self.language.interactor.eval()
        return self

    def _run(self,records,*,teacher):
        if not records:raise ValueError("Empty Stage B batch")
        cfg=self.config["feedback"]
        if teacher:
            answers=[r["supervision"]["teacher_answer"] for r in records]
            context,lengths,lm_loss=object_text_context(self.language,records,self.object_projection,self.mode,
                answers=answers,synthetic=[False]*len(records),task="segmentation",compute_lm_loss=True)
            failed=[False]*len(records)
        else:
            answers=[];failed=[]
            for record in records:
                value=generate_object_answer(self.language,record,self.object_projection,self.mode,
                    max_new_tokens=cfg["max_new_tokens"])
                parsed=parse_answer(value["answer"])
                answers.append(value["answer"])
                failed.append(bool(value["format_failure"] or not parsed.valid or parsed.obj is None))
            context,lengths=object_text_context(self.language,records,self.object_projection,self.mode,
                answers=answers,synthetic=failed,task="segmentation")
            lm_loss=context.question_hidden[0].new_zeros(())
        device=self.visual_adapter.weight.device
        features=[self.visual_adapter(r["superpoint_features"].detach().to(
                    device=device,dtype=self.visual_adapter.weight.dtype)) for r in records]
        xyz=[r["superpoint_xyz"].detach().to(device=device,dtype=torch.float32) for r in records]
        text=[self.text_adapter(torch.cat([q,a]).to(self.text_adapter.weight.dtype))
              for q,a in zip(context.question_hidden,context.answer_hidden)]
        decoded=self.decoder(features,text,xyz,context.token_roles)
        entities=predicted_entity_scores(decoded,context)
        return {"masks":[decoded["pred_masks"][i,:,:len(r["superpoint_features"])] for i,r in enumerate(records)],
            "scores":[s[:,list(roles).index("OBJ")] for s,roles in zip(entities,context.entity_roles)],
            "first_masks":[r["masks"].to(device) for r in records],
            "first_scores":[r["first_scores"].to(device) for r in records],
            "answers":answers,"first_answers":[r["answer"] for r in records],
            "failed":failed,"first_failed":[r["failed"] for r in records],"prefix_lengths":lengths,
            "context":context,"decoder_output":decoded,"lm_loss":lm_loss}

    @torch.no_grad()
    def predict(self,records):
        return self._run(records,teacher=False)

    def forward(self,records):
        result=self._run(records,teacher=True);cfg=self.config["feedback"];objective=self.segmentation
        supervision=[r["supervision"] for r in records]
        _,positive,valid=objective._positive_maps(result["context"],[s["answer_entity_indices"] for s in supervision])
        decoded=result["decoder_output"];targets=[s["gt_superpoint_masks"] for s in supervision]
        primary,span,bce,dice,_,scores=objective._layer_loss(decoded,targets,positive,valid)
        auxiliary=torch.stack([objective._layer_loss(x,targets,positive,valid)[0] for x in decoded["aux_outputs"]]).sum()
        ranking,_=objective._ranking(scores,decoded["pred_masks"],
            [s["gt_point_masks"] for s in supervision],[r["mapping"] for r in records])
        loss=(primary+objective.loss_cfg["aux_weight"]*auxiliary+
              cfg["ranking_weight"]*ranking+cfg["answer_lm_weight"]*result["lm_loss"])
        if self.training:loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        result["loss"]=loss
        result["losses"]={k:v.detach() for k,v in {
            "span":span,"mask_bce":bce,"dice":dice,"auxiliary":auxiliary,
            "ranking":ranking,"answer_lm":result["lm_loss"],"total":loss}.items()}
        return result

    def optimizer_parameter_groups(self):
        cfg=self.config["optimizer"];groups=[];seen=set()
        specifications=[
            ("qwen_lora","language.",cfg["qwen_lora_lr"]),
            ("evidence","object_projection.",cfg["new_lr"]),
            ("adapters",("text_adapter.","visual_adapter."),cfg["adapter_lr"]),
            ("decoder","decoder.",cfg["decoder_lr"])]
        for name,prefix,lr in specifications:
            parameters=[p for n,p in self.named_parameters() if p.requires_grad and n.startswith(prefix)]
            if parameters:
                if any(id(p) in seen for p in parameters):raise RuntimeError("Duplicate Stage B optimizer parameter")
                seen.update(map(id,parameters));groups.append({"name":name,"params":parameters,
                    "lr":float(lr),"weight_decay":float(cfg["weight_decay"])})
        if seen!={id(p) for p in self.parameters() if p.requires_grad}:raise RuntimeError("Stage B optimizer boundary differs")
        return groups

    trainable_state_dict=FeedbackModel.trainable_state_dict

    def architecture_signature(self):
        return {"objective":self.objective,"feedback":dict(self.config["feedback"]),
            "prompt_protocol":SEGMENTATION_PROMPT_PROTOCOL,"decoder_version":self.decoder.version,
            "loss":self.segmentation.loss_cfg,"matcher":self.segmentation.matcher_cfg,
            "ranking":self.segmentation.ranking_cfg,"mask_threshold":self.segmentation.mask_threshold,
            "first_stage":"immutable_original_checkpoint_full_scene_cache"}
