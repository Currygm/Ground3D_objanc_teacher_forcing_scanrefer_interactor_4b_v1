"""Original query/text cosine logits and mean-sigmoid phrase scores."""
import torch
from torch.nn import functional as F

def fixed_candidate_text_scores(scoring_keys,text_features,phrase_spans,*,temperature):
    if temperature<=0 or scoring_keys.ndim!=2 or text_features.ndim!=2:
        raise ValueError("Invalid fixed-query text scoring inputs")
    if scoring_keys.shape[-1]!=text_features.shape[-1] or phrase_spans.shape[-1]!=len(text_features):
        raise ValueError("Cached query/text/phrase dimensions do not align")
    spans=phrase_spans.to(device=text_features.device,dtype=torch.bool)
    if not len(spans) or bool((spans.sum(-1)==0).any()):
        raise ValueError("Every predicted entity needs its own phrase span")
    keys=scoring_keys.detach().to(device=text_features.device,dtype=text_features.dtype)
    logits=torch.einsum("qd,td->qt",keys,F.normalize(text_features,dim=-1))/float(temperature)
    return logits.float().sigmoid()@spans.float().T/spans.sum(-1)[None]
