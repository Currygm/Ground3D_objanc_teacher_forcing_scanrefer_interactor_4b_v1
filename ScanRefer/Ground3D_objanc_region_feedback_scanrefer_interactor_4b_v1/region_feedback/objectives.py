"""Loss targets are computed only after GT-free prediction."""
import torch
from torch.nn import functional as F

def candidate_ious(masks,mapping,truth,threshold=.5):
    mapping=mapping.to(masks.device,dtype=torch.long)
    pred=masks.float().sigmoid()[:,mapping]>=threshold
    truth=truth.to(masks.device,dtype=torch.bool)
    if truth.ndim!=1 or truth.shape[0]!=pred.shape[1]:
        raise ValueError("Raw target does not align with candidate masks")
    return (pred & truth[None]).sum(-1).float()/(pred | truth[None]).sum(-1).clamp_min(1)

def ranking_loss(scores,masks,mappings,truths,temperature=.1):
    if len(scores)!=len(masks) or len(scores)!=len(truths):
        raise ValueError("Ranking batch length mismatch")
    terms=[]
    for s,m,mp,t in zip(scores,masks,mappings,truths):
        with torch.no_grad():quality=candidate_ious(m,mp,t)
        # No informative ordering exists if every predicted mask has zero overlap.
        if quality.max()<=0:terms.append(s.sum()*0.)
        else:terms.append(F.cross_entropy(s.float()[None]/float(temperature),quality.argmax()[None]))
    return torch.stack(terms).mean()
