
import hashlib,json

def language_signature(language,config):
    return {'objective':'ground3d_objanc_v1','language_hidden_size':2560,'language_layers':36,
        'source_manifests':language.source_manifests,'qwen_lora':dict(config['model']['lora']),
        'tokenizer_sha256':hashlib.sha256(json.dumps(language.tokenizer_snapshot(),sort_keys=True).encode()).hexdigest() if hasattr(language,'tokenizer_snapshot') else None,
        'token_ids':dict(language.structured_token_ids),'token_migration':{'<p>':'<obj>','</p>':'</obj>','<SEG>':'<anc>','</anc>':'copy </obj>'},
        'answer_protocol':{k:config['model'][k] for k in ['system_prompt','user_prefix','structured_tokens','generation']},
        'point_encoder':dict(config['point_encoder']),'teacher_forward':'single_causal_ce_and_full_text_hidden',
        'generated_forward':'cached_generation_then_full_causal_replay','geometry':'ground3d_0.02m_voxel_mean_then_raw_point_superpoint_mean'}
