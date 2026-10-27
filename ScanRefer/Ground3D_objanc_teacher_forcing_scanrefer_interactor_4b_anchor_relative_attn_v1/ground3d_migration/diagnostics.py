
import torch

def tie_aware_rank(scores,ious):
    values=torch.as_tensor(ious,device=scores.device)
    order=torch.argsort(scores,descending=True,stable=True)
    best=values.eq(values.max())
    rank=int(torch.where(best[order])[0][0])+1
    return {'best_query_rank':rank,'best_query_reciprocal_rank':1./rank,
        **{'best_query_top'+str(k)+'_rate':float(rank<=k) for k in [1,3,5,10]}}
