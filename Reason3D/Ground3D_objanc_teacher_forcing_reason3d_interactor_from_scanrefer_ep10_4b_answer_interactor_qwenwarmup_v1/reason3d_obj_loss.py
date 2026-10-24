"""OBJ content-token CE on logits already used for full Answer CE."""
import torch
import torch.nn.functional as F


def obj_prediction_mask(maps,parsed_rows,shifted_target):
    mask=torch.zeros_like(shifted_target,dtype=torch.bool)
    for batch_index,((_,answer_indices,_,spans),parsed) in enumerate(zip(maps,parsed_rows)):
        objects=[i for i,entity in enumerate(parsed.entities) if entity.role=="OBJ"]
        if len(objects)!=1:raise ValueError("OBJ supervision requires exactly one OBJ")
        # Hidden state p-1 predicts answer token at position p.
        positions=answer_indices[spans[objects[0]].to(answer_indices.device)]-1
        if not len(positions) or int(positions.min())<0 or int(positions.max())>=mask.shape[1]:
            raise ValueError("Invalid OBJ content prediction positions")
        mask[batch_index,positions]=True
    if not mask.any() or not shifted_target[mask].ne(-100).all():
        raise ValueError("OBJ CE mask must select supervised content tokens")
    return mask


def obj_content_loss(logits,targets,mask):
    if logits.ndim!=2 or targets.ndim!=1 or mask.shape!=targets.shape or not mask.any():
        raise ValueError("Invalid flattened OBJ CE inputs")
    chosen=logits[mask].float();truth=targets[mask]
    loss=F.cross_entropy(chosen,truth)
    return loss,{"obj_token_accuracy":chosen.argmax(-1).eq(truth).float().mean(),
                 "obj_token_count":mask.sum().detach()}
