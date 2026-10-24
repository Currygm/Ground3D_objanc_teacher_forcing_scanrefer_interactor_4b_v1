
"""Small, independently testable migration contracts."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TOKEN_RENAMES={'<p>':'<obj>','</p>':'</obj>','<SEG>':'<anc>'}

def visual_indices(xyz, *, training, cap=2000):
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not torch.isfinite(xyz).all():
        raise ValueError('FPS requires finite [S,3] centers')
    n=len(xyz)
    if n==0 or cap<1:raise ValueError('Empty scene or invalid cap')
    if not training or n<=cap:return torch.arange(n,device=xyz.device)
    chosen=torch.empty(cap,device=xyz.device,dtype=torch.long)
    distances=torch.full((n,),float('inf'),device=xyz.device)
    used=torch.zeros(n,device=xyz.device,dtype=torch.bool)
    current=torch.randint(n,(),device=xyz.device)
    points=xyz.float()
    for i in range(cap):
        chosen[i]=current;used[current]=True
        distances=torch.minimum(distances,(points-points[current]).square().sum(-1))
        distances.masked_fill_(used,-1.)
        current=distances.argmax()
    return chosen

def scatter_mean(values, inverse, count):
    out=values.new_zeros(count,values.shape[-1])
    out.index_add_(0,inverse,values)
    counts=torch.bincount(inverse,minlength=count).clamp_min(1).to(values.dtype)
    return out/counts[:,None]

def prepare_geometry(xyz,rgb,superpoints,*,voxel_size=.02):
    xyz=xyz.float();rgb=rgb.float();superpoints=superpoints.long().reshape(-1)
    if xyz.shape!=rgb.shape or xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)!=len(superpoints):
        raise ValueError('Raw point geometry is misaligned')
    if not torch.isfinite(xyz).all() or not torch.isfinite(rgb).all():raise ValueError('Nonfinite geometry')
    shifted=xyz-xyz.amin(0)
    coords,inverse=torch.unique(torch.floor(shifted/voxel_size).long(),dim=0,sorted=True,return_inverse=True)
    values,raw_to=torch.unique(superpoints,sorted=True,return_inverse=True)
    features=torch.cat([rgb,xyz-xyz.mean(0)],dim=1)
    return {'voxel_coord':coords,'voxel_features':scatter_mean(features,inverse,len(coords)),
        'inverse_map':inverse,'point_to_superpoint':raw_to,'superpoint_values':values,
        'superpoint_coord':scatter_mean(shifted,raw_to,len(values))}

class MigratedTokenRows(nn.Module):
    """Independent input/output rows, with arbitrary original IDs and no base updates."""
    def __init__(self,token_ids,input_weight,output_weight,*,closing_source_id):
        super().__init__()
        if len(token_ids)!=4 or len(set(token_ids))!=4:raise ValueError('Four distinct task IDs required')
        self.token_ids=tuple(map(int,token_ids))
        self.base_vocab_size=output_weight.shape[0]
        self.vocab_size=max(self.base_vocab_size,max(self.token_ids)+1)
        in_rows=[];out_rows=[]
        for row,token in enumerate(self.token_ids):
            source=closing_source_id if row==3 else token
            in_rows.append(input_weight[source].detach().clone())
            out_rows.append(output_weight[source].detach().clone())
        self.input_rows=nn.Parameter(torch.stack(in_rows).float())
        self.output_rows=nn.Parameter(torch.stack(out_rows).float())

    def embed(self,ids,base_embedding):
        task=torch.zeros_like(ids,dtype=torch.bool)
        for token in self.token_ids:task|=ids.eq(token)
        if bool(((ids>=base_embedding.num_embeddings)&~task).any()):raise ValueError('Unknown vocabulary ID')
        result=base_embedding(ids.masked_fill(task,0))
        for row,token in enumerate(self.token_ids):
            result=torch.where(ids.eq(token)[...,None],self.input_rows[row].to(result.dtype),result)
        return result

    def logits(self,hidden,base_head):
        base=base_head(hidden)
        if base.shape[-1]<self.vocab_size:
            base=F.pad(base,(0,self.vocab_size-base.shape[-1]))
        extra=F.linear(hidden.float(),self.output_rows).to(base.dtype)
        return base.index_copy(-1,torch.tensor(self.token_ids,device=base.device),extra)

def migrate_tokenizer(tokenizer):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    original={old:tokenizer.convert_tokens_to_ids(old) for old in TOKEN_RENAMES}
    vocabulary=tokenizer.get_vocab()
    if any(old not in vocabulary for old in TOKEN_RENAMES):raise ValueError('Original grounding tokens missing')
    if any(new in vocabulary for new in [*TOKEN_RENAMES.values(),'</anc>']):raise ValueError('Task token name collision')
    payload=json.loads(tokenizer.backend_tokenizer.to_str())
    for row in payload['added_tokens']:
        row['content']=TOKEN_RENAMES.get(row['content'],row['content'])
    # HF registration can replace additional_special_tokens while retaining backend flags.
    special=[row['content'] for row in payload['added_tokens'] if row.get('special')]
    migrated=PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(payload)),
        bos_token=tokenizer.bos_token,eos_token=tokenizer.eos_token,pad_token=tokenizer.pad_token,
        unk_token=tokenizer.unk_token,padding_side='right',chat_template=tokenizer.chat_template,
        additional_special_tokens=special)
    migrated.add_special_tokens({'additional_special_tokens':[*special,'</anc>']})
    for old,new in TOKEN_RENAMES.items():
        if migrated.convert_tokens_to_ids(new)!=original[old]:raise ValueError('Task ID drift')
    return migrated

def validate_signature(expected,actual):
    if expected!=actual:raise ValueError('Checkpoint objective/architecture/provenance differs')

def verify_download(root):
    root=Path(root).resolve(strict=True)
    manifest=json.loads((root/'download_manifest.json').read_text(encoding='utf-8'))
    for name,record in manifest['files'].items():
        path=(root/name).resolve(strict=True)
        if not path.is_relative_to(root):raise ValueError('Manifest path escape')
        if path.stat().st_size!=record['size']:raise ValueError('Incomplete model file '+name)
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=record['sha256']:raise ValueError('Model hash mismatch '+name)
        official=record.get('official_lfs_sha256')
        if official and official!=record['sha256']:raise ValueError('Official hash mismatch '+name)
    return manifest

def strict_prefix_load(module,state,prefix):
    values={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    expected=module.state_dict()
    if set(values)!=set(expected):
        raise ValueError('Official prefix keys differ '+prefix+' missing='+str(sorted(set(expected)-set(values)))+' extra='+str(sorted(set(values)-set(expected))))
    for name,value in values.items():
        if value.shape!=expected[name].shape or not torch.isfinite(value).all():raise ValueError('Invalid official tensor '+prefix+name)
    module.load_state_dict(values,strict=True)
    return {'prefix':prefix,'tensors':len(values),'parameters':sum(v.numel() for v in values.values())}
