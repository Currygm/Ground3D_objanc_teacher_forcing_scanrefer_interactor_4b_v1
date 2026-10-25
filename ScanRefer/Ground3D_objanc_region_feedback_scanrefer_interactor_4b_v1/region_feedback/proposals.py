"""Frozen first-stage inference accepts no annotation fields."""
from dataclasses import dataclass
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from ground3d_migration.unconstrained import generate_one_unconstrained,merge_prompt_visual
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer

INFERENCE_KEYS=("point_cloud","point_to_superpoint","superpoint_coord","question","ground_geometry")

@dataclass
class Proposal:
    features: torch.Tensor
    xyz: torch.Tensor
    masks: torch.Tensor
    entity_scores: torch.Tensor
    entity_roles: tuple
    answer: str
    visual: object
    point_to_superpoint: torch.Tensor
    failed: bool
    obj_index: int
    scoring_keys: torch.Tensor
    updated_text_features: torch.Tensor
    phrase_spans: torch.Tensor
    text_temperature: float

def predicted_entity_scores(decoder,context):
    output=[]
    for b,spans in enumerate(context.phrase_spans):
        spans=spans.to(decoder["pred_logits"].device)
        if not len(spans) or bool((spans.sum(-1)==0).any()):
            raise ValueError("Every predicted entity needs its own phrase span")
        logits=decoder["pred_logits"][b,:,:spans.shape[1]].float().sigmoid()
        output.append(logits@spans.float().T/spans.sum(-1)[None].clamp_min(1))
    return output

class FrozenProposalStage(nn.Module):
    def __init__(self,model,generation_protocol="unconstrained"):
        super().__init__()
        if generation_protocol not in {"constrained","unconstrained"}:
            raise ValueError("Unknown generation protocol")
        self.model=model.requires_grad_(False).eval()
        self.generation_protocol=generation_protocol
        # Replay actual unconstrained text even if it contains another point token.
        self.model.answer_lm._merge=MethodType(merge_prompt_visual,self.model.answer_lm)
    def train(self,mode=True):
        super().train(False)
        return self
    @torch.no_grad()
    def predict(self,batch):
        observed={k:batch[k] for k in INFERENCE_KEYS if k in batch}
        if any(k not in observed for k in INFERENCE_KEYS[:4]):
            raise ValueError("Missing inference inputs")
        self.model.eval()
        rows=self.model._encode_points(observed)
        if len(rows)!=len(observed["question"]):
            raise ValueError("Visual encoding batch length mismatch")
        lm=self.model.answer_lm
        answers=[];failed=[]
        for question,row in zip(observed["question"],rows):
            generated=(generate_one_unconstrained(lm,question,visual=row.compressed)
                       if self.generation_protocol=="unconstrained"
                       else lm.generate_one(question,visual=row.compressed))
            text=generated["answer"];parsed=parse_answer(text)
            answers.append(text)
            failed.append(bool(generated["format_failure"] or not parsed.valid or parsed.obj is None))
        context=lm.encode_full_context(observed["point_cloud"],observed["question"],answers,
            visuals=[r.compressed for r in rows],synthetic_obj=failed)
        text=[self.model.text_adapter(torch.cat([q,a]).to(self.model.text_adapter.weight.dtype))
              for q,a in zip(context.question_hidden,context.answer_hidden)]
        decoder=self.model.decoder(
            [self.model.visual_adapter(r.superpoint_hidden.to(self.model.visual_adapter.weight.dtype)) for r in rows],
            text,[r.superpoint_xyz for r in rows],context.token_roles)
        scores=predicted_entity_scores(decoder,context)
        result=[]
        for i,row in enumerate(rows):
            roles=tuple(context.entity_roles[i])
            if roles.count("OBJ")!=1:raise ValueError("Predicted context requires one OBJ")
            result.append(Proposal(row.superpoint_hidden.detach(),row.superpoint_xyz.detach(),
                decoder["pred_masks"][i,:,:len(row.superpoint_hidden)].detach(),
                scores[i].detach(),roles,answers[i],row.compressed,
                observed["point_to_superpoint"][i],failed[i],roles.index("OBJ"),
                F.normalize(self.model.decoder.text_query_projection(
                    self.model.decoder.output_norm(decoder["updated_queries"][i])),dim=-1).detach(),
                decoder["updated_text_features"][i,:len(text[i])].detach(),
                context.phrase_spans[i].detach(),float(self.model.decoder.text_temperature)))
        return result
