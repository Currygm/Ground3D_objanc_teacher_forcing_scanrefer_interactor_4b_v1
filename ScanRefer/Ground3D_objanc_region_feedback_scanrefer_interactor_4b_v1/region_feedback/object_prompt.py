"""English object-evidence prompt, with embeddings inserted at tracked boundaries."""
import torch
from torch.nn.utils.rnn import pad_sequence

SYSTEM_PROMPT=("You are performing a second-pass 3D grounding assessment. Use the global scene context "
"and the visual evidence for the predicted objects to identify the instance requested by the original query. "
"The first-pass prediction is a hypothesis and may be incorrect. Check the target description and its spatial "
"relations to the predicted anchors. Do not assume that the first-pass target is correct.")
PROMPT_PROTOCOL="english_object_evidence_textscore_v3"

def render_object_prompt(language,record,projection,mode):
    if mode not in {"empty","text","visual"}:raise ValueError("Unknown object evidence mode")
    device=next(language.language.parameters()).device
    dtype=language.language.get_base_model().get_input_embeddings().weight.dtype
    evidence=record["evidence"]
    chunks=[];insertions=[];masked=[];length=0
    def text(s,*,hide=False):
        nonlocal length
        chunks.append(s)
        if hide:masked.append((length,length+len(s)))
        length+=len(s)
    def visual(value):
        insertions.append((length,value.to(device=device,dtype=dtype)))
    text("Global scene context:\n")
    visual(record["scene"].detach())
    text("\nFirst-pass prediction:\n")
    text(record["answer"],hide=mode=="empty")
    text("\n\n")
    anchors=0;slots=0
    if record["failed"]:
        text("The first-pass output could not be parsed into valid object entities.\n")
    for i,(role,phrase) in enumerate(zip(evidence["roles"],evidence["phrases"])):
        if role=="ANC":
            anchors+=1;label=f"Predicted anchor {anchors}: "
        else:label="Predicted target: "
        text(label)
        text(phrase,hide=mode=="empty")
        text("\nVisual evidence:\n")
        if bool(evidence["valid_mask"][i]):
            feature=evidence["features"][i].detach().to(device=device,dtype=projection[-1].weight.dtype)
            token=projection(feature)[None]
            if mode!="visual":token=token*0
            visual(token);slots+=1
            text("\n")
        else:text("Visual evidence is unavailable for this predicted object.\n")
        text("\n")
    question_left=length+len("Original query:\n")
    text("Original query:\n"+record["question"]+
         "\n\nAssess which instance best satisfies the original query using the supplied evidence.\n")
    user="".join(chunks)
    rendered=language.tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}],
        tokenize=False,add_generation_prompt=True)
    start=rendered.rfind(user)
    if start<0:raise ValueError("Chat template lost evidence content")
    encoded=language.tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True,return_tensors="pt")
    ids=encoded["input_ids"][0].to(device)
    offsets=encoded["offset_mapping"][0]
    embeddings=language.structured.embed(ids,language.language.get_base_model().get_input_embeddings())
    if masked:
        hide=torch.zeros(len(ids),dtype=torch.bool,device=device)
        for left,right in masked:
            hide|=((offsets[:,1]>start+left)&(offsets[:,0]<start+right)).to(device)
        embeddings=torch.where(hide[:,None],torch.zeros_like(embeddings),embeddings)
    # No textual placeholder replacement: each continuous block has an explicit
    # character boundary that is mapped into the rendered template token stream.
    qi=torch.where((offsets[:,1]>start+question_left)&
                   (offsets[:,0]<start+question_left+len(record["question"])))[0].to(device)
    if not len(qi):raise ValueError("Original query mapped to no tokens")
    mapped_qi=qi.clone()
    blocks=[];cursor=0
    for boundary,value in insertions:
        candidates=torch.where(offsets[:,0]>=start+boundary)[0]
        index=int(candidates[0]) if len(candidates) else len(ids)
        if index<cursor:raise ValueError("Non-monotonic evidence boundary")
        mapped_qi=mapped_qi+(qi>=index).long()*len(value)
        blocks.extend([embeddings[cursor:index],value]);cursor=index
    blocks.append(embeddings[cursor:])
    merged=torch.cat(blocks)
    return merged,{"visual_slots":slots,"user_text":user,"prompt_protocol":PROMPT_PROTOCOL,"question_indices":mapped_qi}

def object_prompt_hidden(language,records,projection,mode):
    rendered=[render_object_prompt(language,r,projection,mode) for r in records]
    rows=[x[0] for x in rendered]
    embeds=pad_sequence(rows,batch_first=True)
    valid=pad_sequence([torch.ones(len(x),device=x.device,dtype=torch.long) for x in rows],batch_first=True)
    positions=pad_sequence([torch.arange(len(x),device=x.device) for x in rows],batch_first=True)
    output=language._text_forward(embeds,valid,positions).last_hidden_state
    indices=torch.tensor([len(x)-1 for x in rows],device=embeds.device)
    return output[torch.arange(len(rows),device=embeds.device),indices],[len(x) for x in rows]
