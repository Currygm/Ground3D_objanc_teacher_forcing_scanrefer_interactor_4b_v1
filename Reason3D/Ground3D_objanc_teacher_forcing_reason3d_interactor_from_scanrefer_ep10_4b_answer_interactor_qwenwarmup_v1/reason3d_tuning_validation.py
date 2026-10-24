"""Allow existing scalar hyperparameters while retaining source/protocol checks."""
import copy
import math
from numbers import Real

SCALARS = {
    "loss": {"span_weight", "mask_bce_weight", "dice_weight", "aux_weight"},
    "matcher": {"span_weight", "mask_bce_weight", "dice_weight"},
    "ranking": {"weight", "temperature"},
}
RATES = ("qwen_lora_lr", "interactor_lr", "structured_token_lr",
         "visual_lr", "adapter_lr", "decoder_lr")

def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Expected finite numeric hyperparameter: "+name)
    if value < 0 or (positive and value == 0):
        raise ValueError("Invalid hyperparameter range: "+name)
    return float(value)

def validate_sections(actual, reference):
    for section, allowed in SCALARS.items():
        values, pinned = actual[section], reference[section]
        if set(values) != set(pinned):
            raise ValueError("Changed hyperparameter schema: "+section)
        for key, value in values.items():
            if key in allowed:
                number(value, section+"."+key,
                       positive=(section=="ranking" and key=="temperature"))
            elif value != pinned[key]:
                raise ValueError("Changed fixed protocol field: "+section+"."+key)
        if section in ("loss", "matcher"):
            primary = ("span_weight", "mask_bce_weight", "dice_weight")
            if not any(float(values[key]) > 0 for key in primary):
                raise ValueError("At least one primary loss/matcher weight must be positive")

def validate_training_scalars(training):
    for key in RATES:
        number(training[key], "training."+key, positive=True)
    for key in ("answer_lm_weight", "obj_ce_weight", "weight_decay"):
        number(training[key], "training."+key)
    if training["answer_lm_weight"] + training["obj_ce_weight"] <= 0:
        raise ValueError("Answer stage needs positive language supervision")
    number(training["max_grad_norm"], "training.max_grad_norm", positive=True)
    minimum = number(training["cosine_min_lr_ratio"],
                     "training.cosine_min_lr_ratio", positive=True)
    if minimum > 1:
        raise ValueError("Cosine minimum ratio must be <= 1")

def normalize_source_training_signature(actual, reference):
    """Normalize only approved scalar differences for parent-weight migration.

    This is not used for this experiment's resume or evaluation signature:
    those must still match the actual checkpoint configuration exactly.
    """
    normalized = copy.deepcopy(actual)
    # Structural/protocol differences remain visible to the caller's full equality check.
    for section, allowed in SCALARS.items():
        for key in allowed:
            if key in normalized.get(section, {}) and key in reference.get(section, {}):
                number(normalized[section][key], section+"."+key,
                       positive=(section=="ranking" and key=="temperature"))
                normalized[section][key] = copy.deepcopy(reference[section][key])
    return normalized
