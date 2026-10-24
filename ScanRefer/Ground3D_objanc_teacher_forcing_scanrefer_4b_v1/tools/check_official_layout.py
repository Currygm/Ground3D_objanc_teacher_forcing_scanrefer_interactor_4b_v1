# Metadata preflight only; this sparse archive is never an initialization checkpoint.
import json,sys,collections
from pathlib import Path
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
sizes=json.loads((P/'cache/checkpoint_metadata_sizes.json').read_text());archive=P/'cache/checkpoint_metadata_only.pth'
with archive.open('wb') as f:
    f.truncate(sizes['size']);f.seek(0);f.write((P/'cache/checkpoint_header.bin').read_bytes())
    f.seek(sizes['tail_offset']);f.write((P/'cache/checkpoint_tail.bin').read_bytes())
from ground3d_migration.checkpoint import load_official_checkpoint
with FakeTensorMode():source=load_official_checkpoint(archive)
state=source.get('state_dict',source);state={k.removeprefix('module.'):v for k,v in state.items()}
report={'preflight_only':True,'full_hash_verified':False,'top_level_groups':dict(collections.Counter(k.split('.')[0] for k in state)),
        'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()}}
(P/'docs/official_layout_preflight.json').write_text(json.dumps(report,indent=2))
print('groups',report['top_level_groups'])
for k,v in state.items():
    if 'embed_tokens.weight' in k or 'lm_head.weight' in k or k.startswith('decoder.point2token.') or k.startswith('decoder.x_mask.'):
        print(k,tuple(v.shape),v.dtype)
