"""Evaluation-only raw greedy decoding for GroundObjAncLM.

No grammar, special-token suppression, visible-token filter, phrase budget or
forced EOS. EOS stopping and the configured total generation budget remain.
Padded/undefined vocabulary rows are NOT masked; they are recorded and treated
as an unrepresentable answer, routed to the existing zero-OBJ evaluator.
"""
import torch
from spatiallm_compressed_visual_objanc_full_attention.model.answer_format import parse_answer


@torch.no_grad()
def generate_one_unconstrained(self, question, point_cloud=None, *, visual=None):
    if visual is None:
        raise ValueError("Generation requires encoded Ground3D visuals")
    budget=int(self.generation_config["max_new_tokens"])
    if budget<1:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids,_=self._prompt_tokens(question)
    embeds,pos,_,_=self._merge(prompt_ids,visual)
    valid=torch.ones(1,len(embeds),device=embeds.device,dtype=torch.long)
    out=self._text_forward(embeds[None],valid,pos[None],use_cache=True)
    hidden,past=out.last_hidden_state[:,-1],out.past_key_values
    generated=[]
    stopped_on_eos=False
    for step in range(budget):
        logits=self.structured.logits(hidden,self.base.lm_head)[0]
        if not bool(torch.isfinite(logits).any()) or bool(torch.isnan(logits).any()):
            raise RuntimeError("Invalid unconstrained generation logits")
        # Intentionally select directly from the complete model output.
        token=int(logits.argmax())
        generated.append(token)
        if token==self.eos_token_id:
            stopped_on_eos=True
            break
        if step+1==budget:
            break
        tid=torch.tensor([[token]],device=embeds.device,dtype=torch.long)
        valid=torch.cat([valid,valid.new_ones(1,1)],dim=1)
        next_pos=torch.tensor([[len(embeds)+step]],device=embeds.device)
        out=self._text_forward(
            self.structured.embed(tid,self.language.get_base_model().get_input_embeddings()),
            valid,next_pos,past=past,use_cache=True)
        hidden,past=out.last_hidden_state[:,-1],out.past_key_values
    # EOS is a stopping marker, not part of the answer text. Nothing else is removed.
    answer_ids=generated[:-1] if stopped_on_eos else generated
    answer=self.tokenizer.decode(answer_ids,skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
    parsed=parse_answer(answer)
    vocabulary=set(self.tokenizer.get_vocab().values())
    undefined=sorted({t for t in generated if t not in vocabulary})
    errors=list(parsed.errors)
    if undefined:
        errors.append("undefined_token_id")
    result={
        "answer":answer,"parsed":parsed,"generated_token_ids":generated,
        "raw_text_including_eos":self.tokenizer.decode(
            generated,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        "answer_token_ids":list(answer_ids),"undefined_token_ids":undefined,
        "budget_forced":False,"format_failure":not parsed.valid or bool(undefined),
        "errors":errors,"stop_reason":"eos" if stopped_on_eos else "max_new_tokens",
        "generated_tokens":len(generated),"max_new_tokens":budget,
        "format_constraints":False,"special_token_suppression":False,
    }
    sink=getattr(self,"_raw_generation_sink",None)
    if sink is not None:
        sink({k:v for k,v in result.items() if k!="parsed"})
    return result


def merge_prompt_visual(self, ids, visual):
    """Replace the first prompt point marker, preserving generated markers as text.

    The native chat prompt places its visual marker before the user question and
    assistant answer. Unconstrained answers may contain the same special token.
    Counting markers over the entire replay would incorrectly reject them.
    """
    device=next(self.language.parameters()).device
    ids=ids.to(device)
    markers=torch.where(ids.eq(self.point_token_id))[0]
    if markers.numel()==0:
        raise ValueError("Missing prompt point marker")
    marker=int(markers[0])
    embeds=self.structured.embed(ids,self.language.get_base_model().get_input_embeddings())
    points=visual.hidden.to(device=device,dtype=embeds.dtype)
    merged=torch.cat([embeds[:marker],points,embeds[marker+1:]])
    return merged,torch.arange(len(merged),device=device),marker,len(points)-1
