"""Replay actual predictions or training completions under predicted object evidence."""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from .object_prompt import render_object_prompt
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer
from spatiallm_compressed_visual_objanc_full_attention.model.llm import (
    FullAttentionContext,ROLE_QUESTION,ROLE_OBJ,ROLE_OTHER)

def object_text_context(language,records,projection,mode,*,answers=None,synthetic=None,
                        task="ranking",compute_lm_loss=False):
    if not records or (answers is not None and len(answers)!=len(records)) or (synthetic is not None and len(synthetic)!=len(records)):
        raise ValueError("Feedback text batch lengths differ")
    rows=[];maps=[];parsed_rows=[];lengths=[];labels=[]
    device=next(language.language.parameters()).device
    for b,record in enumerate(records):
        prompt,info=render_object_prompt(language,record,projection,mode,task=task)
        answer=record["answer"] if answers is None else answers[b]
        parsed=parse_answer(answer)
        encoded=language.tokenizer(answer,add_special_tokens=False,return_offsets_mapping=True,return_tensors="pt")
        ids=encoded["input_ids"][0].to(device)
        completion=torch.cat([ids,ids.new_tensor([language.eos_token_id])])
        embedded=language.structured.embed(completion,language.language.get_base_model().get_input_embeddings())
        if mode=="empty" and task=="ranking":
            # Entity text must be absent in BOTH the evidence prefix and replay.
            # Preserve token count, phrase positions and EOS, not lexical content.
            embedded=torch.cat([torch.zeros_like(embedded[:-1]),embedded[-1:]],dim=0)
        rows.append(torch.cat([prompt,embedded.to(prompt)],dim=0));lengths.append(len(prompt))
        target=torch.full((len(prompt)+len(completion),),-100,device=device,dtype=torch.long)
        if compute_lm_loss:target[len(prompt):]=completion
        labels.append(target)
        is_synthetic=(bool(record["failed"] or not parsed.valid or parsed.obj is None)
                      if synthetic is None else bool(synthetic[b]))
        if not is_synthetic and (not parsed.valid or parsed.obj is None):raise ValueError("Invalid second-pass teacher answer")
        if compute_lm_loss and is_synthetic:raise ValueError("Synthetic OBJ cannot be a language training target")
        if is_synthetic:
            roles=torch.full((len(ids),),ROLE_OTHER,dtype=torch.long)
            spans=torch.zeros((0,len(ids)),dtype=torch.bool)
        else:roles,spans=language._answer_roles(answer,encoded["offset_mapping"][0],parsed)
        maps.append((info["question_indices"],torch.arange(len(ids),device=device)+len(prompt),
                     roles,spans,is_synthetic))
        parsed_rows.append(parsed)
    embeds=pad_sequence(rows,batch_first=True)
    valid=pad_sequence([torch.ones(len(row),device=device,dtype=torch.long) for row in rows],batch_first=True)
    positions=pad_sequence([torch.arange(len(row),device=device) for row in rows],batch_first=True)
    hidden=language._text_forward(embeds,valid,positions).last_hidden_state
    question_hidden=[];answer_hidden=[];role_rows=[];span_rows=[];entity_roles=[]
    for b,(qi,ai,roles,spans,synthetic) in enumerate(maps):
        q=hidden[b,qi];a=hidden[b,ai]
        if synthetic:
            a=torch.cat([a,a.new_zeros(1,a.shape[-1])])
            roles=torch.cat([roles,roles.new_tensor([ROLE_OBJ])])
            spans=torch.zeros((1,len(q)+len(a)),dtype=torch.bool);spans[0,-1]=True
            er=("OBJ",)
        else:
            spans=torch.cat([torch.zeros((len(spans),len(q)),dtype=torch.bool),spans],dim=1)
            er=tuple(e.role for e in parsed_rows[b].entities)
        question_hidden.append(q);answer_hidden.append(a);span_rows.append(spans);entity_roles.append(er)
        role_rows.append(torch.cat([torch.full((len(q),),ROLE_QUESTION,dtype=torch.long),roles]))
    context=FullAttentionContext(question_hidden,answer_hidden,role_rows,span_rows,entity_roles,parsed_rows)
    if compute_lm_loss:
        target=pad_sequence(labels,batch_first=True,padding_value=-100)[:,1:]
        selected=target.ne(-100)
        logits=language.structured.logits(hidden[:,:-1][selected],language.language.get_base_model().lm_head)
        loss=F.cross_entropy(logits.float(),target[selected])
        return context,lengths,loss
    return context,lengths
