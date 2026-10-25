"""Full-denominator candidate quality and ID/tie-aware selection diagnostics."""
import torch
from .objectives import candidate_ious

def candidate_metrics(scores,masks,mapping,truth):
    quality=candidate_ious(masks,mapping,truth)
    if scores.shape!=quality.shape or not torch.isfinite(scores).all():
        raise ValueError("Invalid candidate scores")
    order=scores.float().argsort(descending=True,stable=True)
    selected=int(order[0]);best=int(quality.argmax())
    q=float(quality[best]);s=float(quality[selected])
    result={"selected_iou":s,"oracle_iou":q,"acc25":float(s>=.25),"acc50":float(s>=.5),
        "oracle_acc25":float(q>=.25),"oracle_acc50":float(q>=.5),
        "best_id_hit":float(selected==best),"best_tie_hit":float(abs(s-q)<=1e-6),
        "covered50":float(q>=.5),"covered50_correct":float(q>=.5 and s>=.5),
        "covered50_best_tie":float(q>=.5 and abs(s-q)<=1e-6),
        "regret":q-s,"selected_query":selected,"best_query":best,
        "best_rank":int((order==best).nonzero()[0])+1}
    for k in [1,3,5,10]:
        top=float(quality[order[:min(k,len(order))]].max())
        result["top"+str(k)+"_oracle_iou"]=top
        result["top"+str(k)+"_best_tie"]=float(abs(top-q)<=1e-6)
    return result

def zero_metrics(record):
    return {k:(51 if k=="best_rank" else 0.) for k in record}

def summarize(records):
    if not records:raise ValueError("No evaluation records")
    summary={"samples":len(records),"failure_rate":sum(bool(r["failed"]) for r in records)/len(records)}
    for key in ["main","strict","first"]:
        names=set(records[0][key])-{"selected_query","best_query"}
        values={name:sum(r[key][name] for r in records)/len(records) for name in names}
        covered=sum(r[key]["covered50"] for r in records)
        values["selection_acc50_given_covered"]=(sum(r[key]["covered50_correct"] for r in records)/covered if covered else None)
        values["best_tie_given_covered50"]=(sum(r[key]["covered50_best_tie"] for r in records)/covered if covered else None)
        summary[key]=values
    return summary
