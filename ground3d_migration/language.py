
"""Ground3D language base with one causal teacher pass and full-answer context."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from spatiallm_compressed_visual_objanc_full_attention.model.llm import (
    CompressedVisualObjAncLM,FullAttentionContext,ROLE_QUESTION,ROLE_OBJ,ROLE_OTHER)
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import ObjAncGrammar,Phase,parse_answer
from .core import migrate_tokenizer,MigratedTokenRows,verify_download,strict_prefix_load
from .visual import GroundVisualEncoder
from .checkpoint import load_official_checkpoint

class GroundObjAncLM(CompressedVisualObjAncLM):
    objective='ground3d_objanc_answer_lm_v1'
    def __init__(self,config):
        nn.Module.__init__(self)
        from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
        from peft import LoraConfig,get_peft_model
        self.config=config;cfg=config['model']
        self.source_manifests={}
        specs=json.loads(Path(__file__).with_name('official_sources.json').read_text())
        for name,key,folder in [('qwen','model_name_or_path','qwen3_vl_4b'),('ground3d','ground3d_weights_dir','ground3d_3d')]:
            manifest=verify_download(cfg[key]);spec=specs[folder]
            if manifest['repo_id']!=spec['repo_id'] or manifest['revision']!=spec['revision'] or set(manifest['files'])!=set(spec['files']):raise ValueError('Unrecognized official repository/revision/files '+name)
            for filename,expected in spec['files'].items():
                record=manifest['files'][filename]
                if record['size']!=expected['size']:raise ValueError('Official file size differs '+filename)
                if expected['official_lfs_sha256']:
                    if record['sha256']!=expected['official_lfs_sha256']:raise ValueError('Pinned official LFS hash differs '+filename)
                else:
                    import hashlib
                    data=(Path(cfg[key])/filename).read_bytes()
                    actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                    if actual!=expected['official_git_blob_sha1']:raise ValueError('Pinned official Git blob differs '+filename)
            self.source_manifests[name]=manifest
        source=load_official_checkpoint(Path(cfg['ground3d_weights_dir'])/'pytorch_model.pth')
        state=source.get('state_dict',source)
        if not isinstance(state,dict):raise ValueError('Official checkpoint lacks state dict')
        state={k.removeprefix('module.'):v for k,v in state.items()}
        self.original_tokenizer=AutoTokenizer.from_pretrained(cfg['model_name_or_path'],local_files_only=True,use_fast=True,padding_side='right')
        self.original_tokenizer.add_special_tokens({'additional_special_tokens':['<p>','</p>','<SEG>','<name>','<sentence>','<|point|>']})
        qwen,base_loading=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_name_or_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        bad={k:v for k,v in base_loading.items() if k in {'missing_keys','unexpected_keys','mismatched_keys','error_msgs'} and v}
        if bad:raise ValueError('Incomplete official Qwen base '+str(bad))
        self.base_loading_report=base_loading
        if (qwen.config.text_config.hidden_size,qwen.config.text_config.num_hidden_layers)!=(2560,36):raise ValueError('Expected Qwen3-VL-4B')
        if len(self.original_tokenizer)>qwen.get_input_embeddings().num_embeddings:qwen.resize_token_embeddings(len(self.original_tokenizer))
        qwen.requires_grad_(False)
        targets=[n for n,m in qwen.named_modules() if any(n.endswith(k) for k in cfg['lora']['target_modules'])]
        self.language=get_peft_model(qwen,LoraConfig(task_type='CAUSAL_LM',r=int(cfg['lora']['r']),lora_alpha=int(cfg['lora']['alpha']),lora_dropout=float(cfg['lora']['dropout']),bias=cfg['lora']['bias'],target_modules=targets))
        self.lora_module_count=sum('visual' not in n for n in targets)
        self.loading_report=[strict_prefix_load(self.language,state,'decoder.reason_model.')]
        self.visual=GroundVisualEncoder(state)
        from .interactor import GroundInteractor
        self.interactor=GroundInteractor(config['interactor'])
        used=['input_conv.','unet.','output_layer.','decoder.x_mask.','decoder.reason_model.']
        permitted=set(json.loads(Path(__file__).with_name('official_exclusions.json').read_text()))
        excluded={k for k in state if not any(k.startswith(prefix) for prefix in used)}
        if excluded!=permitted:raise ValueError('Official exclusion keys differ: extra='+str(sorted(excluded-permitted))+' missing='+str(sorted(permitted-excluded)))
        self.excluded_official_keys=sorted(excluded)
        self.tokenizer=migrate_tokenizer(self.original_tokenizer)
        tokens=cfg['structured_tokens'];self.structured_token_ids={t:self.tokenizer.convert_tokens_to_ids(t) for t in tokens}
        self.structured=MigratedTokenRows([self.structured_token_ids[t] for t in tokens],qwen.get_input_embeddings().weight,qwen.lm_head.weight,closing_source_id=self.structured_token_ids['</obj>'])
        self.base_vocab_size=qwen.get_input_embeddings().num_embeddings
        self.point_token_id=self.tokenizer.convert_tokens_to_ids('<|point|>')
        self.eos_token_id=int(self.tokenizer.eos_token_id)
        self.system_prompt=cfg['system_prompt'];self.point_prompt='<|point|>';self.user_prefix=cfg['user_prefix'];self.generation_config=cfg['generation']
        for n,p in self.language.named_parameters():p.requires_grad_('lora_' in n and 'visual' not in n)
        self.visual.requires_grad_(False)
        if cfg.get('gradient_checkpointing',True):qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        qwen.config.use_cache=False
        del state,source

    def tokenizer_snapshot(self):
        return {'backend':self.tokenizer.backend_tokenizer.to_str(),'special_tokens':self.tokenizer.special_tokens_map,
            'chat_template':self.tokenizer.chat_template,'padding_side':self.tokenizer.padding_side}

    def restore_tokenizer_snapshot(self,snapshot):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        if snapshot!=self.tokenizer_snapshot():raise ValueError('Checkpoint tokenizer protocol differs')
        restored=PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(snapshot['backend']),
            chat_template=snapshot['chat_template'],padding_side=snapshot['padding_side'],**snapshot['special_tokens'])
        if restored.get_vocab()!=self.tokenizer.get_vocab():raise ValueError('Checkpoint tokenizer ID drift')
        self.tokenizer=restored

    @property
    def base(self):
        qwen=self.language.get_base_model()
        return SimpleNamespace(config=qwen.config.text_config,lm_head=qwen.lm_head)

    def train(self,mode=True):
        nn.Module.train(self,mode)
        self.language.get_base_model().model.visual.eval()
        if not any(p.requires_grad for p in self.visual.parameters()):self.visual.eval()
        return self

    def _render_prompt(self,question):
        return self.tokenizer.apply_chat_template([{'role':'system','content':self.system_prompt},
            {'role':'user','content':self.point_prompt+'\n'+self.user_prefix+question}],tokenize=False,add_generation_prompt=True)

    def _merge(self,ids,visual):
        device=next(self.language.parameters()).device;ids=ids.to(device)
        marker=self._single_position(ids,self.point_token_id)
        embeds=self.structured.embed(ids,self.language.get_base_model().get_input_embeddings())
        points=visual.hidden.to(device=device,dtype=embeds.dtype)
        merged=torch.cat([embeds[:marker],points,embeds[marker+1:]])
        return merged,torch.arange(len(merged),device=device),marker,len(points)-1

    def _text_forward(self,embeds,valid,positions,*,past=None,use_cache=False):
        if not hasattr(self,'call_counts'):self.call_counts={}
        key='generation_prefill' if use_cache and past is None else ('generation_incremental' if use_cache else 'causal_forward')
        self.call_counts[key]=self.call_counts.get(key,0)+1
        return self.language.get_base_model().model.language_model(inputs_embeds=embeds,attention_mask=valid,
            position_ids=positions[None].expand(3,-1,-1),past_key_values=past,use_cache=use_cache,return_dict=True)

    def _causal(self,questions,answers,visuals,*,synthetic_obj=None,compute_loss=True):
        if not(len(questions)==len(answers)==len(visuals)):raise ValueError('Causal batch lengths differ')
        synthetic_obj=[False]*len(answers) if synthetic_obj is None else list(synthetic_obj)
        if len(synthetic_obj)!=len(answers) or (compute_loss and any(synthetic_obj)):raise ValueError('Synthetic OBJ is evaluation-only')
        rows=[];labels=[];maps=[];parsed_rows=[]
        for question,answer,visual,synthetic in zip(questions,answers,visuals,synthetic_obj):
            parsed=parse_answer(answer)
            if not parsed.valid and not synthetic:raise ValueError('Invalid OBJ/ANC answer '+str(parsed.errors))
            prompt_ids,qidx=self._prompt_tokens(question)
            encoded=self.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
            aids=encoded['input_ids'][0];all_a=torch.cat([aids,aids.new_tensor([self.eos_token_id])])
            ids=torch.cat([prompt_ids,all_a]);merged,pos,marker,shift=self._merge(ids,visual)
            label=torch.full((len(merged),),-100,device=merged.device,dtype=torch.long)
            start=len(prompt_ids)+shift;label[start:]=all_a.to(merged.device)
            if synthetic:
                roles=torch.full((len(aids),),ROLE_OTHER,dtype=torch.long)
                spans=torch.zeros((0,len(aids)),dtype=torch.bool)
            else:
                roles,spans=self._answer_roles(answer,encoded['offset_mapping'][0],parsed)
            mapped_q=qidx+(qidx>marker).long()*shift
            maps.append((mapped_q.to(merged.device),torch.arange(len(aids),device=merged.device)+start,roles,spans))
            rows.append(merged);labels.append(label);parsed_rows.append(parsed)
        embeds=pad_sequence(rows,batch_first=True)
        valid=pad_sequence([torch.ones(len(x),device=embeds.device,dtype=torch.long) for x in rows],batch_first=True)
        positions=pad_sequence([torch.arange(len(x),device=embeds.device) for x in rows],batch_first=True)
        output=self._text_forward(embeds,valid,positions)
        if compute_loss:
            target=pad_sequence(labels,batch_first=True,padding_value=-100)[:,1:];selected=target.ne(-100)
            logits=self.structured.logits(output.last_hidden_state[:,:-1][selected],self.base.lm_head)
            loss=F.cross_entropy(logits.float(),target[selected]);accuracy=logits.argmax(-1).eq(target[selected]).float().mean()
        else:
            loss=output.last_hidden_state.new_zeros(());accuracy=loss.detach()
        qh=[];ah=[];rs=[];ss=[];ers=[]
        for b,(qi,ai,role,span) in enumerate(maps):
            q=output.last_hidden_state[b,qi];a=output.last_hidden_state[b,ai]
            if synthetic_obj[b]:
                a=torch.cat([a,a.new_zeros(1,a.shape[-1])],dim=0)
                role=torch.cat([role,role.new_tensor([ROLE_OBJ])])
                span=torch.zeros((1,len(q)+len(a)),dtype=torch.bool);span[0,-1]=True
                entity_roles=('OBJ',)
            else:
                span=torch.cat([torch.zeros(len(span),len(q),dtype=torch.bool),span],dim=1)
                entity_roles=tuple(e.role for e in parsed_rows[b].entities)
            qh.append(q);ah.append(a)
            rs.append(torch.cat([torch.full((len(q),),ROLE_QUESTION,dtype=torch.long),role]))
            ss.append(span);ers.append(entity_roles)
        return {'loss':loss,'metrics':{'token_accuracy':accuracy},'context':FullAttentionContext(qh,ah,rs,ss,ers,parsed_rows)}

    def encode_batch_points(self,batch):
        import time
        device=next(self.interactor.parameters(),torch.empty(0)).device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        rows=[self.visual(cloud,mapping,xyz,**({'geometry':batch['ground_geometry'][i]} if 'ground_geometry' in batch else {})) for i,(cloud,mapping,xyz) in enumerate(zip(batch['point_cloud'],batch['point_to_superpoint'],batch['superpoint_coord']))]
        if device.type=='cuda':torch.cuda.synchronize(device)
        encoded=time.perf_counter()
        prefixes=self.interactor([row.superpoint_hidden for row in rows])
        from .visual import GroundVisualTokens
        for row,prefix in zip(rows,prefixes):row.compressed=GroundVisualTokens(prefix,None)
        self.visual_seconds=encoded-started
        self.interactor_seconds=time.perf_counter()-encoded
        return rows

    def forward(self,point_clouds,questions,teacher_answers,*,visuals=None):
        if visuals is None:raise ValueError('Ground3D requires full raw-superpoint batch encoding')
        return self._causal(questions,teacher_answers,visuals)

    def encode_full_context(self,point_clouds,questions,answers,*,visuals=None,synthetic_obj=None):
        if visuals is None:raise ValueError('Missing Ground3D visual features')
        return self._causal(questions,answers,visuals,synthetic_obj=synthetic_obj,compute_loss=False)['context']

    @torch.no_grad()
    def generate_one(self,question,point_cloud=None,*,visual=None):
        if visual is None:raise ValueError('Generation requires encoded Ground3D visuals')
        prompt_ids,_=self._prompt_tokens(question)
        embeds,pos,_,_=self._merge(prompt_ids,visual)
        valid=torch.ones(1,len(embeds),device=embeds.device,dtype=torch.long)
        out=self._text_forward(embeds[None],valid,pos[None],use_cache=True)
        hidden=out.last_hidden_state[:,-1];past=out.past_key_values
        grammar=ObjAncGrammar(self.structured_token_ids,self.eos_token_id,int(self.generation_config['max_phrase_tokens']))
        generated=[];forced=False;budget=int(self.generation_config['max_new_tokens'])
        defined=torch.zeros(self.structured.vocab_size,device=embeds.device,dtype=torch.bool)
        defined[torch.tensor(sorted(set(self.tokenizer.get_vocab().values())),device=embeds.device)]=True
        forbidden=set(self.tokenizer.all_special_ids)-set(self.structured_token_ids.values())-{self.eos_token_id}
        for step in range(budget):
            logits=self.structured.logits(hidden,self.base.lm_head)[0]
            phrase=self.tokenizer.decode(grammar.phrase_ids or [],skip_special_tokens=True).strip()
            allowed=grammar.allowed(len(logits),logits.device,phrase_has_text=bool(phrase));allowed&=defined[:len(logits)];before=allowed.clone()
            allowed=grammar.force_for_budget(allowed,budget-step,phrase_has_text=bool(phrase));forced|=not torch.equal(before,allowed)
            if forbidden:allowed[torch.tensor(sorted(forbidden),device=logits.device)]=False
            masked=logits.masked_fill(~allowed,-torch.inf)
            if not torch.isfinite(masked).any():raise RuntimeError('No legal generated token')
            token=int(masked.argmax())
            if grammar.phase in {Phase.IN_OBJ,Phase.IN_ANC} and not phrase:
                while not self.tokenizer.decode([token],skip_special_tokens=True).strip():
                    masked[token]=-torch.inf
                    if not torch.isfinite(masked).any():raise RuntimeError('No visible phrase token')
                    token=int(masked.argmax())
            generated.append(token);grammar.update(token,phrase_has_text=bool(phrase))
            if grammar.phase is Phase.FINISHED:break
            tid=torch.tensor([[token]],device=embeds.device)
            valid=torch.cat([valid,valid.new_ones(1,1)],dim=1)
            next_pos=torch.tensor([[len(embeds)+step]],device=embeds.device)
            out=self._text_forward(self.structured.embed(tid,self.language.get_base_model().get_input_embeddings()),valid,next_pos,past=past,use_cache=True)
            hidden=out.last_hidden_state[:,-1];past=out.past_key_values
        answer=self.tokenizer.decode([t for t in generated if t!=self.eos_token_id],skip_special_tokens=False,clean_up_tokenization_spaces=False).strip()
        parsed=parse_answer(answer)
        return {'answer':answer,'parsed':parsed,'budget_forced':forced,'format_failure':not parsed.valid or grammar.phase is not Phase.FINISHED,'errors':list(parsed.errors),'generated_token_ids':generated}
