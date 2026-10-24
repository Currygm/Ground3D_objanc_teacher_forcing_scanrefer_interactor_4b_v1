"""Instance-local raw decoding. Preserve native 2560 pre-adapter zero OBJ."""
from types import MethodType
from ground3d_migration.unconstrained import generate_one_unconstrained,merge_prompt_visual

GENERATION_PROTOCOL={
    "format_constraints":False,"greedy":True,"special_token_suppression":False,
    "undefined_id_suppression":False,"forced_tag_closure":False,"forced_eos":False,
    "max_new_tokens":128,"answer_repair":False,"teacher_fallback":False,
    "format_failure_policy":"zero_obj_decoder",
    "zero_obj_boundary":"llm_hidden_2560_pre_adapter",
    "strict_raw_failure_policy":"zero_score_over_full_sample_denominator",
}

def install_unconstrained(model,sink=None):
    model.answer_lm.generate_one=MethodType(generate_one_unconstrained,model.answer_lm)
    model.answer_lm._merge=MethodType(merge_prompt_visual,model.answer_lm)
    model.answer_lm._raw_generation_sink=sink
