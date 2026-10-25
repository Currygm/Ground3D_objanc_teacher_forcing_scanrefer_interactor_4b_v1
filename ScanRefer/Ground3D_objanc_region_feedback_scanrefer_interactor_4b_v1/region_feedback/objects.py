"""One visual descriptor per actually predicted entity, never per decoder query."""
from dataclasses import dataclass
import torch
from .regions import pool_regions
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer

@dataclass
class ObjectEvidence:
    features: torch.Tensor
    roles: tuple
    phrases: tuple
    source_query_ids: torch.Tensor
    valid_mask: torch.Tensor

@torch.no_grad()
def select_object_evidence(proposal):
    parsed=parse_answer(proposal.answer)
    if proposal.failed or not parsed.valid or parsed.obj is None:
        return ObjectEvidence(proposal.features.new_empty((0,proposal.features.shape[-1])),
            (),(),torch.empty(0,device=proposal.features.device,dtype=torch.long),
            torch.empty(0,device=proposal.features.device,dtype=torch.bool))
    roles=tuple(e.role for e in parsed.entities)
    if roles!=proposal.entity_roles or proposal.entity_scores.shape!=(len(proposal.masks),len(roles)):
        raise ValueError("Predicted entity binding mismatch")
    selected=proposal.entity_scores.argmax(0)
    pooled=pool_regions(proposal.features,proposal.xyz,proposal.masks[selected],
                        proposal.point_to_superpoint)
    return ObjectEvidence(pooled.features,roles,tuple(e.phrase for e in parsed.entities),
                          selected,pooled.valid_mask)
