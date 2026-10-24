import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
source=P.parent/'SpatialLM_objanc_teacher_forcing_scanrefer_3b_v1'
manifest=json.loads((P/'docs/source_manifest.json').read_text())
changed=[name for name,digest in manifest.items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest]
if changed:raise RuntimeError('Original source changed '+str(changed))
original_python='/home/huangxiang/anaconda3/envs/gmspatialLMseg/bin/python'
version=subprocess.check_output([original_python,'-c','import importlib.metadata as m;print(m.version("transformers"))'],text=True).strip()
if version!='4.46.1':raise RuntimeError('Original Transformers environment version changed')
report={'source_files_verified':len(manifest),'changed':changed,'original_transformers':version}
(P/'docs/source_preservation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
