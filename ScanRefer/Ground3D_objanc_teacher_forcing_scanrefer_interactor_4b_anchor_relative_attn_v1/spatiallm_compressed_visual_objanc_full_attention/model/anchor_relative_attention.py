"""Prediction-only anchor conditioning; no GT, new heads or new loss terms."""
import hashlib
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def project(attn, x, which, bias=True):
    d=attn.embed_dim
    b=attn.in_proj_bias[which*d:(which+1)*d] if bias and attn.in_proj_bias is not None else None
    return F.linear(x,attn.in_proj_weight[which*d:(which+1)*d],b)


def shuffled_centres(mu, valid, sample_id, seed=42):
    ids=valid.nonzero().flatten()
    if len(ids)<2:return mu
    digest=hashlib.sha256(f'{seed}:{sample_id}'.encode()).digest()
    g=torch.Generator().manual_seed(int.from_bytes(digest[:8],'little')%(2**63-1))
    order=ids[torch.randperm(len(ids),generator=g).to(ids.device)]
    # A random cycle is a derangement for every n>=2, without retry loops.
    out=mu.clone();out[order]=mu[order.roll(1)]
    return out


def relative_inputs(attn, query, logits, masks, xyz, counts, spans, roles,
                    eps=1e-6, shuffle_id=None):
    if len(spans)!=len(roles) or spans.shape[-1]!=logits.shape[-1]:
        raise ValueError('Entity spans do not align with decoder text')
    for v in (query,logits,masks,xyz,counts):
        if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite anchor inputs')
    if (counts<0).any():raise ValueError('Negative superpoint counts')
    obj=[s for s,r in zip(spans,roles) if r=='OBJ' and s.any()]
    anc=[s for s,r in zip(spans,roles) if r=='ANC' and s.any()]
    if not obj or not anc or counts.sum()<=0:return None
    # FP32 statistics, even under surrounding BF16 autocast.
    p=masks.detach().float().sigmoid();counts=counts.float();xyz=xyz.float()
    weighted=p*counts
    mass=weighted.sum(-1)
    valid=mass/counts.sum()>eps
    if not valid.any():return None
    mu=weighted@xyz/mass.clamp_min(eps)[:,None]
    if shuffle_id is not None:mu=shuffled_centres(mu,valid,shuffle_id)
    scores=torch.stack([logits[:,s].float().sigmoid().mean(-1) for s in anc],-1)
    scores=scores*valid[:,None]
    keep=scores.sum(0)>eps
    if not keep.any():return None
    scores=scores[:,keep];w=scores/scores.sum(0,keepdim=True).clamp_min(eps)
    qa=w.T@query.float()
    h=attn.num_heads;d=attn.head_dim
    q=project(attn,query.float(),0).reshape(-1,h,d)
    k=project(attn,qa,1).reshape(-1,h,d)
    pair=torch.einsum('jhd,ahd->jha',q,k).mean(1)/math.sqrt(d)
    pi=torch.cat([pair.new_zeros(len(query),1),pair],-1).softmax(-1)[:,1:]
    u=pi@qa
    c=pi@w.T
    rho=logits[:,obj[0]].float().sigmoid().mean(-1)
    return u,c,rho,mu,valid


def geometry_delta(attn, pe, xyz, mu, c, q_heads, candidate_chunk, visual_chunk):
    """Return [H,Q,N] logits; never materialize [Q,K,N,C]."""
    h=attn.num_heads;dh=attn.head_dim
    def point_block(x, centres, weights, q):
        base=pe(x)
        result=q.new_zeros(h,q.shape[1],len(x))
        for start in range(0,len(centres),candidate_chunk):
            centres_block=centres[start:start+candidate_chunk]
            e=pe(x[None]-centres_block[:,None])-base[None]
            keys=project(attn,e,1,bias=False).reshape(len(centres_block),len(x),h,dh)
            # Contract candidates first: O(Q*N*C), not O(Q*K*N*C).
            k=torch.einsum('jk,kihd->jihd',weights[:,start:start+candidate_chunk],keys)
            result=result+torch.einsum('hjd,jihd->hji',q,k)/math.sqrt(dh)
        return result
    rows=[]
    for start in range(0,len(xyz),visual_chunk):
        x=xyz[start:start+visual_chunk]
        args=(x,mu,c,q_heads)
        if torch.is_grad_enabled():
            rows.append(checkpoint(point_block,*args,use_reentrant=False))
        else:rows.append(point_block(*args))
    return torch.cat(rows,-1)


def relative_cross_attention(attn, query_input, visual, visual_pos, blocked,
                             u,c,rho,mu,xyz,pe,gamma,eta,candidate_chunk=8,visual_chunk=1024,
                             semantic=True,geometry=True):
    """One sample; exact original projection/bias/softmax/value convention."""
    with torch.autocast(device_type=query_input.device.type,enabled=False):
        query=query_input.float()
        if semantic:query=query+gamma.float()*rho[:,None]*u
        q=project(attn,query,0).reshape(-1,attn.num_heads,attn.head_dim).transpose(0,1)
        k=project(attn,visual.float()+visual_pos.float(),1).reshape(-1,attn.num_heads,attn.head_dim).transpose(0,1)
        v=project(attn,visual.float(),2).reshape(-1,attn.num_heads,attn.head_dim).transpose(0,1)
        logits=q@k.transpose(-1,-2)/math.sqrt(attn.head_dim)
        if geometry:
            delta=geometry_delta(attn,pe,xyz.float(),mu,c,q,candidate_chunk,visual_chunk)
            logits=logits+eta.float()*rho[None,:,None]*delta
        logits=logits.masked_fill(blocked[None],float('-inf'))
        probability=logits.softmax(-1)
        if not torch.isfinite(probability).all():raise FloatingPointError('Nonfinite anchor attention')
        probability=F.dropout(probability,p=attn.dropout,training=attn.training)
        value=(probability@v).transpose(0,1).reshape(len(query),attn.embed_dim)
        return F.linear(value,attn.out_proj.weight,attn.out_proj.bias).to(query_input.dtype)
