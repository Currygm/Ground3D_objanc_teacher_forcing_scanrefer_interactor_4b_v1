import json,sys
from pathlib import Path
import torch
from torch import nn
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration,AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
from peft import LoraConfig,get_peft_model
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from ground3d_migration.visual import GroundVisualEncoder
from ground3d_migration.core import migrate_tokenizer
layout=json.loads((P/'docs/official_layout_preflight.json').read_text())['tensors']
config=Qwen3VLConfig.from_pretrained(P/'models/qwen3_vl_4b',local_files_only=True)
with torch.device('meta'):
    qwen=Qwen3VLForConditionalGeneration(config)
    targets=[n for n,m in qwen.named_modules() if any(n.endswith(k) for k in ['q_proj','k_proj','v_proj','o_proj','attn.qkv','attn.proj'])]
    language=get_peft_model(qwen,LoraConfig(task_type='CAUSAL_LM',r=32,lora_alpha=64,lora_dropout=.1,bias='none',target_modules=targets))
    # Bypass tensor copying in this metadata-only preflight; exact keys/shapes are compared below.
    import ground3d_migration.visual as route
    old=route.strict_prefix_load;route.strict_prefix_load=lambda *args:{}
    try:visual=GroundVisualEncoder({})
    finally:route.strict_prefix_load=old
    visual.point2token=nn.Sequential(Qwen3VLTextRMSNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,2560))
report={'preflight_only':True,'full_hash_verified':False,'components':[]}
for module,prefix in [(language,'decoder.reason_model.'),(visual.input_conv,'input_conv.'),(visual.unet,'unet.'),(visual.output_layer,'output_layer.'),(visual.x_mask,'decoder.x_mask.'),(visual.point2token,'decoder.point2token.')]:
    expected=module.state_dict();actual={k[len(prefix):]:v for k,v in layout.items() if k.startswith(prefix)}
    if set(actual)!=set(expected):raise ValueError(prefix+' keys differ: missing '+str(sorted(set(expected)-set(actual)))+' extra '+str(sorted(set(actual)-set(expected))))
    mismatch=[k for k in expected if list(expected[k].shape)!=actual[k]['shape']]
    if mismatch:raise ValueError(prefix+' shapes differ '+str(mismatch))
    report['components'].append({'prefix':prefix,'tensors':len(actual),'elements':sum(v.numel() for v in expected.values())})
original=AutoTokenizer.from_pretrained(P/'models/qwen3_vl_4b',local_files_only=True)
original.add_special_tokens({'additional_special_tokens':['<p>','</p>','<SEG>','<name>','<sentence>','<|point|>']})
tokenizer=migrate_tokenizer(original);directory=P/'models/objanc_tokenizer_preflight';tokenizer.save_pretrained(directory)
loaded=AutoTokenizer.from_pretrained(directory,local_files_only=True)
assert tokenizer.get_vocab()==loaded.get_vocab()
report['task_token_ids']={t:tokenizer.encode(t,add_special_tokens=False) for t in ['<obj>','</obj>','<anc>','</anc>']}
assert all(len(v)==1 for v in report['task_token_ids'].values())
report['lora_targets']=len(targets);report['language_lora_targets']=sum('visual' not in n for n in targets)
(P/'docs/official_shapes_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
