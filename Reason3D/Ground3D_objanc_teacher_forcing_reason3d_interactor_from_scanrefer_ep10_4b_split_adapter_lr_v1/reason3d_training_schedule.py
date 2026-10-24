"""Answer-only language adaptation followed by the original joint boundary."""
import torch
from spatiallm_compressed_visual_objanc_full_attention.model.answer_stage_3b import is_answer_parameter


def set_training_epoch(stage,epoch):
    phase="answer_only" if epoch<int(stage.config["training"]["answer_warmup_epochs"]) else "joint"
    for name,param in stage.named_parameters():
        param.requires_grad_(name in stage._checkpoint_names and (phase=="joint" or is_answer_parameter(name)))
        if not param.requires_grad:param.grad=None
    stage._training_phase=phase
    stage.train()
    if phase=="answer_only":
        stage.answer_lm.visual.eval();stage.decoder.eval();stage.text_adapter.eval();stage.visual_adapter.eval()
    return phase


def answer_only_forward(stage,batch):
    rows=stage._encode_points(batch)
    lm=stage.answer_lm(batch["point_cloud"],batch["question"],batch["teacher_answer"],visuals=[row.compressed for row in rows])
    answer,obj=lm["loss"],lm["losses"]["answer_obj_ce"]
    loss=stage.answer_lm_weight*answer+stage.obj_ce_weight*obj
    loss=stage._connect_trainable_graph(loss)
    return {"loss":loss,"losses":{"total":loss.detach(),"answer_lm":answer.detach(),"answer_obj_ce":obj.detach(),
        "answer_contribution":stage.answer_lm_weight*answer.detach(),"obj_ce_contribution":stage.obj_ce_weight*obj.detach()},
        "metrics":{"answer_token_accuracy":lm["metrics"]["token_accuracy"],"answer_obj_token_accuracy":lm["metrics"]["obj_token_accuracy"],
        "answer_obj_token_count":lm["metrics"]["obj_token_count"]}}
