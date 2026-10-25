"""Stage A: one Qwen, existing language LoRA, and immutable cached candidates."""
import torch
from torch import nn
from .object_prompt import PROMPT_PROTOCOL
from .object_context import object_text_context
from .text_scores import fixed_candidate_text_scores
from .selector import OrdinarySelector
from .objectives import ranking_loss

PROTOCOL="object_evidence_shared_qwen_textscore_v3"
OBJECTIVE="ground3d_object_feedback_shared_qwen_textscore_v3"

class FeedbackModel(nn.Module):
    objective=OBJECTIVE
    def __init__(self,language,config,*,text_adapter=None,decoder=None):
        super().__init__()
        self.config=config;cfg=config["feedback"]
        if cfg["stage"]!="a":raise ValueError("Stage B is not supported by object-feedback v3")
        if cfg.get("protocol")!=PROTOCOL:raise ValueError("Legacy all-query feedback protocol is not supported")
        self.method=cfg.get("method","qwen_feedback");self.mode=cfg["evidence_mode"]
        if self.method not in {"qwen_feedback","selector"} or self.mode not in {"empty","text","visual"}:
            raise ValueError("Invalid Stage A method/control")
        # Sole owner: no frozen first-stage model or copied language model is retained.
        self.language=language.requires_grad_(False)
        if self.method=="qwen_feedback":
            for name,p in self.language.named_parameters():
                p.requires_grad_(name.startswith("language.") and "lora_" in name and "visual" not in name)
            if not any(p.requires_grad for p in self.language.parameters()):
                raise ValueError("Existing Qwen language LoRA parameters are required")
            self.object_projection=nn.Sequential(nn.LayerNorm(cfg["input_dim"]),
                                                nn.Linear(cfg["input_dim"],cfg["language_dim"]))
            if text_adapter is None or decoder is None:
                raise ValueError("Original frozen text adapter and decoder text modules are required")
            self.text_adapter=text_adapter.requires_grad_(False)
            self.text_connector=decoder.text_connector.requires_grad_(False)
            self.role_embedding=decoder.role_embedding.requires_grad_(False)
            self.text_temperature=float(decoder.text_temperature)
            self.max_text_tokens=int(decoder.max_text_tokens)
        else:
            self.selector=OrdinarySelector(cfg["input_dim"],cfg["language_dim"],cfg["hidden_dim"],cfg["num_queries"])
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        # GroundObjAncLM.train keeps its native visual tower in eval.
        self.language.train(mode if self.method=="qwen_feedback" else False)
        self.language.visual.eval()
        if hasattr(self.language,"interactor"):self.language.interactor.eval()
        if hasattr(self,"text_adapter"):
            self.text_adapter.eval();self.text_connector.eval();self.role_embedding.eval()
        return self

    def predict(self,records):
        if not records:raise ValueError("Empty cached batch")
        if self.method=="qwen_feedback":
            context,lengths=object_text_context(self.language,records,self.object_projection,self.mode)
            scores=[]
            for i,record in enumerate(records):
                raw=torch.cat([context.question_hidden[i],context.answer_hidden[i]])
                if len(raw)>self.max_text_tokens:raise ValueError("Question+Answer exceeds original text limit")
                adapted=self.text_adapter(raw.to(self.text_adapter.weight.dtype))
                text=self.text_connector(adapted)+self.role_embedding(context.token_roles[i].to(adapted.device))
                if float(record["text_temperature"])!=self.text_temperature:
                    raise ValueError("Cached text scoring temperature changed")
                entity=fixed_candidate_text_scores(record["scoring_keys"],text,context.phrase_spans[i],
                                                   temperature=self.text_temperature)
                scores.append(entity[:,list(context.entity_roles[i]).index("OBJ")])
            scores=torch.stack(scores)
        else:
            scores=self.selector(records);lengths=[0]*len(records)
        return {"masks":[r["masks"].to(scores.device) for r in records],"scores":list(scores.unbind()),
            "first_masks":[r["masks"].to(scores.device) for r in records],"first_scores":[r["first_scores"].to(scores.device) for r in records],
            "answers":[r["answer"] for r in records],"first_answers":[r["answer"] for r in records],
            "failed":[r["failed"] for r in records],"first_failed":[r["failed"] for r in records],
            "prefix_lengths":lengths}

    def forward(self,records):
        result=self.predict(records);cfg=self.config["feedback"]
        rank=ranking_loss(result["scores"],result["masks"],[r["mapping"] for r in records],
                          [r["truth"] for r in records],cfg["ranking_temperature"])
        loss=cfg["ranking_weight"]*rank
        # Controls/invalid regions can leave the projection unused on one rank.
        # Include an exact zero dependency so DDP can synchronize every trainable.
        if self.training:loss=loss+sum(p.reshape(-1)[0]*0 for p in self.parameters() if p.requires_grad)
        result["loss"]=loss
        result["losses"]={"ranking":rank.detach(),"total":loss.detach()}
        return result

    def optimizer_parameter_groups(self):
        cfg=self.config["optimizer"];groups=[];seen=set()
        for name,select,lr in [
            ("qwen_lora",lambda n:n.startswith("language.") and "lora_" in n,cfg["qwen_lora_lr"]),
            ("evidence",lambda n:not n.startswith("language."),cfg["new_lr"])]:
            params=[p for n,p in self.named_parameters() if p.requires_grad and select(n)]
            if params:
                if any(id(p) in seen for p in params):raise RuntimeError("Duplicate optimizer parameter")
                seen.update(map(id,params))
                groups.append({"name":name,"params":params,"lr":float(lr),"weight_decay":float(cfg["weight_decay"])})
        if seen!={id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError("Optimizer parameter boundary mismatch")
        return groups

    def trainable_state_dict(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def architecture_signature(self):
        return {"objective":self.objective,"feedback":dict(self.config["feedback"]),
                "prompt_protocol":PROMPT_PROTOCOL,"score_protocol":"cached_keys_original_text_match_v3","first_stage":"immutable_original_checkpoint_cache"}
