import torch
from torch import nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast,Qwen3VLConfig,Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model
from ground3d_migration.language import GroundObjAncLM
from ground3d_migration.core import migrate_tokenizer,MigratedTokenRows
from ground3d_migration.visual import GroundVisualTokens

def tiny_lm():
    raw=Tokenizer(WordLevel({'[UNK]':0,'[EOS]':1,'where':2,'is':3,'red':4,'chair':5,'near':6,'table':7,'and':8,'door':9,'It':10},unk_token='[UNK]'))
    raw.pre_tokenizer=Whitespace()
    original=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',eos_token='[EOS]',pad_token='[UNK]')
    original.add_special_tokens({'additional_special_tokens':['<p>','</p>','<SEG>','<name>','<sentence>','<|point|>']})
    original.chat_template="{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    tokenizer=migrate_tokenizer(original)
    config=Qwen3VLConfig(text_config={'vocab_size':len(original),'hidden_size':64,'intermediate_size':128,'num_hidden_layers':2,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':16,'rope_scaling':{'rope_type':'default','mrope_section':[2,3,3],'mrope_interleaved':True}},vision_config={'depth':1,'hidden_size':64,'intermediate_size':128,'num_heads':4,'out_hidden_size':64,'deepstack_visual_indexes':[]})
    config._attn_implementation='sdpa'
    qwen=Qwen3VLForConditionalGeneration(config).requires_grad_(False)
    language=get_peft_model(qwen,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj']))
    for n,p in language.named_parameters():p.requires_grad_('lora_' in n and 'visual' not in n)
    lm=GroundObjAncLM.__new__(GroundObjAncLM);nn.Module.__init__(lm)
    lm.language=language;lm.tokenizer=tokenizer
    lm.structured_token_ids={t:tokenizer.convert_tokens_to_ids(t) for t in ['<obj>','</obj>','<anc>','</anc>']}
    lm.structured=MigratedTokenRows(list(lm.structured_token_ids.values()),qwen.get_input_embeddings().weight,qwen.lm_head.weight,closing_source_id=lm.structured_token_ids['</obj>'])
    lm.point_token_id=tokenizer.convert_tokens_to_ids('<|point|>');lm.eos_token_id=1
    lm.point_prompt='<|point|>';lm.system_prompt='ground';lm.user_prefix='';lm.generation_config={'max_phrase_tokens':32,'max_new_tokens':128}
    lm.visual=nn.Identity();lm.eval();return lm

def test_real_qwen_causal_padding_offsets_full_spans_and_gradients():
    torch.manual_seed(8);lm=tiny_lm()
    questions=['where is red chair near table','where is chair']
    answers=['It is <obj>red chair</obj> near <anc>table</anc> and <anc>door</anc>','<obj>chair</obj>']
    visuals=[GroundVisualTokens(torch.randn(n,64,requires_grad=True),torch.zeros(n,3)) for n in [5,3]]
    before={n:p.detach().clone() for n,p in lm.named_parameters() if not p.requires_grad}
    calls=[];hook=lm.language.get_base_model().model.language_model.register_forward_hook(lambda *x:calls.append(1))
    output=lm._causal(questions,answers,visuals);hook.remove()
    assert len(calls)==1
    context=output['context']
    assert context.entity_roles[0]==('OBJ','ANC','ANC')
    for b in range(2):
        prompt,qidx=lm._prompt_tokens(questions[b]);aids=lm.tokenizer.encode(answers[b],add_special_tokens=False)
        embeds,pos,marker,shift=lm._merge(torch.cat([prompt,torch.tensor(aids+[1])]),visuals[b])
        manual=lm._text_forward(embeds[None],torch.ones(1,len(embeds),dtype=torch.long),pos[None]).last_hidden_state[0]
        assert torch.allclose(context.question_hidden[b],manual[qidx+(qidx>marker)*shift],atol=1e-5)
        assert torch.allclose(context.answer_hidden[b],manual[len(prompt)+shift:len(prompt)+shift+len(aids)],atol=1e-5)
        assert context.phrase_spans[b].shape[1]==len(context.question_hidden[b])+len(aids)
        assert all(context.phrase_spans[b].sum(1)>0)
    output['loss'].backward()
    assert all(v.hidden.grad is not None and v.hidden.grad.abs().sum()>0 for v in visuals)
    assert lm.structured.input_rows.grad.abs().sum()>0
    assert lm.structured.output_rows.grad.abs().sum()>0
    assert sum(p.grad.abs().sum() for n,p in lm.named_parameters() if 'lora_' in n and p.grad is not None)>0
    torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],lr=1e-3,weight_decay=.1).step()
    assert all(torch.equal(dict(lm.named_parameters())[n],v) for n,v in before.items())

def test_cached_generation_matches_reference_causal_replay_and_replays_once():
    torch.manual_seed(9);lm=tiny_lm();question='where is chair'
    visual=GroundVisualTokens(torch.randn(4,64),torch.zeros(4,3))
    answer='<obj>red chair</obj> near <anc>table</anc>'
    expected=lm.tokenizer.encode(answer,add_special_tokens=False)+[lm.eos_token_id]
    original_logits=lm.structured.logits;counter=[0];captured=[];forward=lm._text_forward
    def force(hidden,head):
        scores=torch.full((hidden.shape[0],lm.structured.vocab_size),-1000.)
        scores[:,expected[counter[0]]]=1000.;counter[0]+=1;return scores
    def record(*args,**kwargs):
        output=forward(*args,**kwargs);captured.append(output.last_hidden_state[0,-1].detach());return output
    lm.structured.logits=force;lm._text_forward=record
    generated=lm.generate_one(question,visual=visual)
    assert generated['generated_token_ids']==expected and not generated['format_failure']
    lm.structured.logits=original_logits;lm._text_forward=forward
    prompt,_=lm._prompt_tokens(question)
    embeds,pos,_,shift=lm._merge(torch.cat([prompt,torch.tensor(expected)]),visual)
    replay=lm._text_forward(embeds[None],torch.ones(1,len(embeds),dtype=torch.long),pos[None]).last_hidden_state[0]
    for step,hidden in enumerate(captured):assert torch.allclose(hidden,replay[len(prompt)+shift-1+step],atol=1e-5)
    calls=[];hook=lm.language.get_base_model().model.language_model.register_forward_hook(lambda *x:calls.append(1))
    context=lm.encode_full_context([], [question],[generated['answer']],visuals=[visual]);hook.remove()
    assert len(calls)==1 and context.entity_roles[0]==('OBJ','ANC')

def test_tokenizer_snapshot_roundtrip_rejects_task_protocol_drift():
    import copy,pytest
    lm=tiny_lm();snapshot=lm.tokenizer_snapshot()
    lm.restore_tokenizer_snapshot(snapshot)
    assert lm.tokenizer_snapshot()==snapshot
    bad=copy.deepcopy(snapshot);bad['chat_template']='wrong'
    with pytest.raises(ValueError):lm.restore_tokenizer_snapshot(bad)

def test_generation_masks_padded_ids_outside_and_inside_visible_phrase():
    lm=tiny_lm();qwen=lm.language.get_base_model()
    # Native Qwen head is larger than tokenizer vocabulary.
    qwen.resize_token_embeddings(32,mean_resizing=False)
    lm.structured.base_vocab_size=32;lm.structured.vocab_size=32
    answer='<obj>red chair</obj>'
    expected=lm.tokenizer.encode(answer,add_special_tokens=False)+[lm.eos_token_id]
    counter=[0]
    def biased_logits(hidden,head):
        scores=torch.full((1,32),-1000.)
        # Invalid padded ID is always attractive, including before OBJ and after red.
        scores[:,31]=1000.;scores[:,expected[min(counter[0],len(expected)-1)]]=999.
        counter[0]+=1;return scores
    lm.structured.logits=biased_logits
    generated=lm.generate_one('where is chair',visual=GroundVisualTokens(torch.randn(3,64),torch.zeros(3,3)))
    assert generated['generated_token_ids']==expected
    assert not generated['format_failure']
