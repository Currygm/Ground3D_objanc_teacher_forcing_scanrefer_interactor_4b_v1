"""LL3DA visual-only latent visual prefix, without visual prompts."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import torch
from torch import nn
from transformers import InstructBlipQFormerConfig, InstructBlipQFormerModel

QFORMER_CONFIG = dict(vocab_size=30522, hidden_size=768, num_hidden_layers=6,
    num_attention_heads=12, intermediate_size=3072, encoder_hidden_size=256,
    cross_attention_frequency=2, hidden_act='gelu', hidden_dropout_prob=.1,
    attention_probs_dropout_prob=.1, max_position_embeddings=512,
    initializer_range=.02, layer_norm_eps=1e-12, pad_token_id=0,
    position_embedding_type='absolute')

@dataclass
class LatentVisualPrefix:
    hidden: torch.Tensor
    memory_count: int
    truncated: bool = False

def sync_clock(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
    return time.perf_counter()

class GroundInteractor(nn.Module):
    def __init__(self, config, *, load_embeddings=True):
        super().__init__()
        self.config = dict(config)
        if int(config['num_queries']) != 32: raise ValueError('LL3DA requires 32 latent queries')
        self.encoder_to_qformer_projection = nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.qformer = InstructBlipQFormerModel(InstructBlipQFormerConfig(**QFORMER_CONFIG))
        self.latent_query = nn.Embedding(32,768)
        nn.init.normal_(self.latent_query.weight,mean=0.,std=.02)
        self.qformer_to_language_projection = nn.Linear(768,int(config['language_hidden_size']))
        self.asset_manifest = self._load_embeddings() if load_embeddings else {'test_random_initialization':True}
        self.last_seconds = 0.
        self.qformer.embeddings.word_embeddings.requires_grad_(False)
        self.qformer.embeddings.position_embeddings.requires_grad_(False)
        for layer in self.qformer.encoder.layer:
            layer.intermediate.requires_grad_(False)
            layer.output.requires_grad_(False)
        self.frozen_text_parameters=sorted(n for n,p in self.named_parameters() if not p.requires_grad)

    def _load_embeddings(self):
        from runtime_helpers import sha256
        root=Path(self.config['asset_dir'])
        manifest=json.loads((root/'asset_manifest.json').read_text())
        from prepare_interactor_assets import REPO,REVISION,MODEL_SHA,FILES,LL3DA_REVISION
        if (manifest.get('repo'),manifest.get('revision'),manifest.get('official_ll3da_revision'))!=(REPO,REVISION,LL3DA_REVISION):
            raise ValueError('Interactor initialization provenance differs')
        if set(manifest['files'])!=set(FILES) or manifest['files'].get('model.safetensors')!=MODEL_SHA:
            raise ValueError('Interactor initialization manifest incomplete or untrusted')
        for name,digest in manifest['files'].items():
            if Path(name).name!=name or sha256(root/name)!=digest: raise ValueError('Interactor asset hash mismatch')
        from safetensors.torch import load_file
        path=root/'model.safetensors'
        state=load_file(str(path)) if path.exists() else torch.load(root/'pytorch_model.bin',map_location='cpu',weights_only=True)
        destination=self.qformer.state_dict(); loaded=[]; excluded=[]
        with torch.no_grad():
            for name,value in state.items():
                key=name.removeprefix('qformer.').replace('.gamma','.weight').replace('.beta','.bias')
                if name.startswith(('bert.encoder.','bert.pooler.','cls.')) or name=='embeddings.token_type_embeddings.weight':
                    excluded.append(name);continue
                if not key.startswith('embeddings.') or key not in destination or value.shape!=destination[key].shape:
                    raise ValueError(f'Unexpected LL3DA embedding weight: {name}')
                destination[key].copy_(value); loaded.append(key)
        required={n for n in destination if n.startswith('embeddings.')}
        if set(loaded)!=required: raise ValueError(f'Incomplete embedding initialization: {required-set(loaded)}')
        return {**manifest,'loaded_keys':sorted(loaded),'excluded_keys':sorted(excluded),'qformer_config':dict(QFORMER_CONFIG)}

    def signature(self):
        return {'implementation':'ground3d_full_superpoint_query_only_v1','queries':32,
            'interactor_text_input':False,'frozen_text_parameters':self.frozen_text_parameters,
            'qformer':dict(QFORMER_CONFIG),'config':self.config,'assets':self.asset_manifest}

    def forward(self, memories):
        if not memories: raise ValueError('Interactor batch is empty')
        device=self.latent_query.weight.device; dtype=next(self.encoder_to_qformer_projection.parameters()).dtype
        start=sync_clock(device)
        lengths=[len(x) for x in memories]
        if min(lengths)<=0 or any(x.ndim!=2 or x.shape[1]!=256 for x in memories):
            raise ValueError('Interactor requires nonempty full superpoint [S,256]')
        memory=torch.nn.utils.rnn.pad_sequence([x.to(device=device,dtype=dtype) for x in memories],batch_first=True)
        memory_mask=torch.arange(memory.shape[1],device=device)[None]<torch.tensor(lengths,device=device)[:,None]
        queries=self.latent_query.weight[None].expand(len(memories),-1,-1)
        output=self.qformer(input_ids=None,query_embeds=queries,
            attention_mask=torch.ones((len(memories),32),device=device,dtype=torch.long),
            encoder_hidden_states=self.encoder_to_qformer_projection(memory),
            encoder_attention_mask=memory_mask,return_dict=True)
        prefix=self.qformer_to_language_projection(output.last_hidden_state[:,:32])
        self.last_seconds=sync_clock(device)-start
        return list(prefix.unbind(0))
