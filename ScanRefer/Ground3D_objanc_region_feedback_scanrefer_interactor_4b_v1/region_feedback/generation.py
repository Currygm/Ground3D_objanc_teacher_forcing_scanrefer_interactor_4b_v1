"""Second-pass raw greedy generation from English object evidence, without GT."""
import torch
from .object_prompt import render_object_prompt
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer

@torch.no_grad()
def generate_object_answer(language,record,projection,mode,*,max_new_tokens):
    budget=int(max_new_tokens)
    if budget<1:raise ValueError("Second-pass generation budget must be positive")
    embeddings,_=render_object_prompt(language,record,projection,mode,task="segmentation")
    valid=torch.ones((1,len(embeddings)),device=embeddings.device,dtype=torch.long)
    positions=torch.arange(len(embeddings),device=embeddings.device)[None]
    out=language._text_forward(embeddings[None],valid,positions,use_cache=True)
    hidden=out.last_hidden_state[:,-1];past=out.past_key_values
    generated=[];stopped=False
    for step in range(budget):
        logits=language.structured.logits(hidden,language.language.get_base_model().lm_head)[0]
        if not torch.isfinite(logits).any() or torch.isnan(logits).any():raise RuntimeError("Invalid second-pass logits")
        token=int(logits.argmax());generated.append(token)
        if token==language.eos_token_id:stopped=True;break
        if step+1==budget:break
        ids=torch.tensor([[token]],device=embeddings.device,dtype=torch.long)
        valid=torch.cat([valid,valid.new_ones((1,1))],dim=1)
        positions=torch.tensor([[len(embeddings)+step]],device=embeddings.device,dtype=torch.long)
        out=language._text_forward(language.structured.embed(ids,language.language.get_base_model().get_input_embeddings()),
                                   valid,positions,past=past,use_cache=True)
        hidden=out.last_hidden_state[:,-1];past=out.past_key_values
    answer_ids=generated[:-1] if stopped else generated
    answer=language.tokenizer.decode(answer_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    parsed=parse_answer(answer);vocabulary=set(language.tokenizer.get_vocab().values())
    undefined=sorted({i for i in generated if i not in vocabulary})
    return {"answer":answer,"format_failure":not parsed.valid or parsed.obj is None or bool(undefined),
        "generated_token_ids":generated,"undefined_token_ids":undefined,
        "stop_reason":"eos" if stopped else "max_new_tokens","format_constraints":False}
