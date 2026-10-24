"""CPU-only full official verification; never starts training or GPU forward."""
import json,os,sys,time,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import torch
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
from ground3d_migration.language import GroundObjAncLM
from ground3d_migration.core import verify_download
started=time.time();config=load_segmentation_config(P/'configs/scanrefer_teacher_forcing_univlg9.yaml')
model=GroundObjAncLM(config)
report={'device':'cpu','full_hash_verified':True,'base_loading':model.base_loading_report,
    'official_components':model.loading_report+model.visual.loading_report,'excluded_keys':model.excluded_official_keys,
    'task_token_ids':model.structured_token_ids,'source_manifests':model.source_manifests,
    'interactor':model.interactor.signature(),'native_point2token_present':hasattr(model.visual,'point2token'),
    'elapsed_sec':time.time()-started,'gpu_initialized':torch.cuda.is_initialized()}
assert not report['gpu_initialized']
for t,i in model.structured_token_ids.items():assert model.tokenizer.encode(t,add_special_tokens=False)==[i]
assert torch.equal(model.structured.input_rows[3],model.structured.input_rows[1])
assert torch.equal(model.structured.output_rows[3],model.structured.output_rows[1])
assert not hasattr(model.visual,'point2token')
assert all(not p.requires_grad for p in model.visual.parameters())
assert all(not p.requires_grad for n,p in model.language.named_parameters() if 'lora_' not in n or 'visual' in n)
model.train();assert not model.visual.training
model.tokenizer.save_pretrained(P/'models/objanc_tokenizer')
(P/'docs/full_weight_initialization.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {'source_manifests','excluded_keys'}},indent=2))
