"""The sole allowed parent-to-experiment loss change is aux_weight 0.5 -> 0.1."""
from copy import deepcopy

def validate_loss_config(loss,source_loss):
    if source_loss.get("aux_weight")!=.5:raise ValueError("Expected pinned baseline auxiliary weight 0.5")
    expected={**source_loss,"aux_weight":.1}
    if loss!=expected:raise ValueError("Only auxiliary loss weight 0.5 -> 0.1 is permitted")

def parent_signature_for_loading(signature):
    result=deepcopy(signature)
    if result.get("loss",{}).get("aux_weight")!=.1:
        raise ValueError("Expected native auxiliary-0.1 experiment signature")
    result["loss"]["aux_weight"]=.5
    return result
