"""Generated-only inference: GT is accessed only after the prediction is fixed."""
from collections import Counter
import torch
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from spatiallm_compressed_visual_objanc_full_attention.model.curriculum import normalize_phrase
from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import mean_sigmoid_scores


def failure(answer,reason,*,format_valid=False,budget_forced=False):
    return {"valid":False,"answer":answer,"reason":reason,"format_valid":format_valid,
            "budget_forced":budget_forced,"selected_query":None}


@torch.no_grad()
def infer_generated_no_fallback(model,batch):
    if len(batch["question"])!=1:
        raise ValueError("No-fallback evaluation uses batch size one")
    points=model._encode_points(batch)
    prefixes=[row.compressed for row in points]
    try:
        generated=model.answer_lm.generate_one(batch["question"][0],visual=prefixes[0])
    except torch.OutOfMemoryError:
        raise
    except RuntimeError as exc:
        if not any(message in str(exc) for message in ("No visible phrase token","No legal generated token")):
            raise
        return failure("","generation_failure")
    answer=generated["answer"]
    parsed=parse_answer(answer)
    budget_forced=bool(generated.get("budget_forced",False))
    if generated.get("format_failure",False) or not parsed.valid:
        return failure(answer,"format_failure",budget_forced=budget_forced)
    obj_indices=[i for i,entity in enumerate(parsed.entities) if entity.role=="OBJ"]
    if len(obj_indices)!=1:
        return failure(answer,"missing_or_multiple_obj",format_valid=True,budget_forced=budget_forced)
    # Only generated text reaches the causal replay. There is no call to the
    # inherited GT-phrase matching, teacher LM, Hungarian matcher or fallback.
    try:
        context=model.answer_lm.encode_full_context(batch["point_cloud"],batch["question"],[answer],
            visuals=prefixes)
    except ValueError as exc:
        if "zero Answer tokens" not in str(exc):raise
        return failure(answer,"empty_obj_token_span",format_valid=True,budget_forced=budget_forced)
    text_rows,positive,valid=model._positive_maps(context,[obj_indices])
    text_rows=[model.text_adapter(row.to(model.text_adapter.weight.dtype)) for row in text_rows]
    output=model.decoder([model.visual_adapter(row.superpoint_hidden.to(model.visual_adapter.weight.dtype)) for row in points],text_rows,
        [row.superpoint_xyz for row in points],context.token_roles)
    logits=output["pred_logits"]
    positive=positive[:,:,:logits.shape[-1]].to(logits.device)
    if not positive.any():
        return failure(answer,"obj_span_truncated",format_valid=True,budget_forced=budget_forced)
    scores=mean_sigmoid_scores(logits,positive,valid.to(logits.device))[0,:,0]
    selected=int(scores.argmax())
    return {"valid":True,"answer":answer,"reason":None,"format_valid":True,"budget_forced":budget_forced,
        "selected_query":selected,"query_scores":scores,"superpoint_logits":output["pred_masks"][0],
        "generated_obj_phrase":parsed.entities[obj_indices[0]].phrase}


def zero_metrics():
    return {"obj/selected_iou":0.,"obj/selected_acc@0.25":0.,"obj/selected_acc@0.50":0.,
        "obj/oracle_iou":0.,"obj/oracle_acc@0.25":0.,"obj/oracle_acc@0.50":0.,
        **{f"obj/top{k}_oracle_iou":0. for k in (1,3,5,10)}}


@torch.no_grad()
def evaluate_no_fallback(model,loader,threshold,examples_limit,*,use_amp):
    rows=[]
    for index,batch in enumerate(loader,1):
        if index==1 or index%100==0:
            print(f"generated_no_fallback: {index}/{len(loader)}",flush=True)
        device_type=next(model.parameters()).device.type
        with torch.autocast(device_type=device_type,dtype=torch.bfloat16,enabled=use_amp and device_type=="cuda"):
            prediction=infer_generated_no_fallback(model,batch)
        # All GT access is below inference, after query selection is complete.
        parsed=parse_answer(prediction["answer"])
        teacher=parse_answer(batch["teacher_answer"][0])
        predicted_obj=next((entity for entity in parsed.entities if entity.role=="OBJ"),None)
        target_obj=next(entity for entity in teacher.entities if entity.role=="OBJ")
        metrics=zero_metrics()
        if prediction["valid"]:
            masks=prediction["superpoint_logits"]
            mapping=batch["point_to_superpoint"][0].to(masks.device,torch.long)
            truth=batch["gt_point_masks"][0][0].to(masks.device,torch.bool)
            ious=[]
            for logits in masks:
                binary=logits[mapping].sigmoid()>=threshold
                ious.append(float(((binary&truth).sum().float()/(binary|truth).sum().clamp_min(1)).item()))
            selected=ious[prediction["selected_query"]]
            oracle=max(ious)
            order=torch.argsort(prediction["query_scores"],descending=True).tolist()
            metrics.update({"obj/selected_iou":selected,"obj/selected_acc@0.25":float(selected>=.25),
                "obj/selected_acc@0.50":float(selected>=.5),"obj/oracle_iou":oracle,
                "obj/oracle_acc@0.25":float(oracle>=.25),"obj/oracle_acc@0.50":float(oracle>=.5),
                **{f"obj/top{k}_oracle_iou":max(ious[q] for q in order[:k]) for k in (1,3,5,10)}})
        rows.append({"evaluation_index":int(batch["evaluation_index"][0]),"sample_id":batch["sample_id"][0],
            "question":batch["question"][0],"generated_answer":prediction["answer"],
            "selected_query":prediction["selected_query"],"valid":prediction["valid"],
            "format_valid":prediction["format_valid"],"budget_forced":prediction["budget_forced"],
            "failure_reason":prediction["reason"],"metrics":metrics,
            "obj_phrase_correct":bool(predicted_obj is not None and normalize_phrase(predicted_obj.phrase)==normalize_phrase(target_obj.phrase))})
    return {"rows":rows}


def merge_no_fallback_payloads(payloads,examples_limit):
    rows=sorted([row for payload in payloads for row in payload["rows"]],key=lambda row:row["evaluation_index"])
    if len({row["evaluation_index"] for row in rows})!=len(rows):
        raise ValueError("Duplicate samples in no-fallback distributed evaluation")
    denominator=max(len(rows),1)
    summary={key:sum(row["metrics"][key] for row in rows)/denominator for key in zero_metrics()}
    reasons=Counter(row["failure_reason"] for row in rows if not row["valid"])
    summary.update({"samples":len(rows),"valid_predictions":sum(row["valid"] for row in rows),
        "failure_rate":sum(not row["valid"] for row in rows)/denominator,"failure_reasons":dict(reasons),
        "answer/fallback_rate":0.,"answer/format_valid_rate":sum(row["format_valid"] for row in rows)/denominator,
        "answer/obj_phrase_accuracy":sum(row["obj_phrase_correct"] for row in rows)/denominator,
        "answer/budget_forced_rate":sum(row["budget_forced"] for row in rows)/denominator,
        "policy":"generated OBJ only; no GT conditioning or phrase-based rejection; invalid predictions score zero over the full split"})
    examples=[{key:value for key,value in row.items() if key!="evaluation_index"} for row in rows[:examples_limit]]
    return summary,examples
