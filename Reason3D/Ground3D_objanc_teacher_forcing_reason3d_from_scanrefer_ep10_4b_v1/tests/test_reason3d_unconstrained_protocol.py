"""CPU regressions for unconstrained generation; no model weights or CUDA required."""
from types import SimpleNamespace, MethodType
import importlib
import torch
from torch import nn
import pytest

class TinyTokenizer:
    eos_token_id = 7
    all_special_ids = [0, 1, 2, 3, 4, 7, 8]
    words = {0:"<obj>",1:"</obj>",2:"<anc>",3:"</anc>",4:"<vision>",5:"chair",6:" ",7:"<eos>",8:"<pad>"}
    def get_vocab(self): return {v:k for k,v in self.words.items()}
    def decode(self, ids, **kwargs): return "".join(self.words.get(int(i), "") for i in ids)

class TinyLM:
    def __init__(self, tokens, budget=128):
        self.tokens=tokens; self.tokenizer=TinyTokenizer(); self.eos_token_id=7
        self.generation_config={"max_new_tokens":budget, "max_phrase_tokens":32, "constrained":True}
        self.structured_token_ids={"<obj>":0,"</obj>":1,"<anc>":2,"</anc>":3}
        self.next_index=0; self.recorded=[]
        self._raw_generation_sink=self.recorded.append
        self.base=SimpleNamespace(lm_head=None)
        self.language=SimpleNamespace(get_base_model=lambda:SimpleNamespace(get_input_embeddings=lambda:nn.Embedding(10,2)))
        self.structured=SimpleNamespace(logits=self.logits,embed=lambda ids,base:torch.zeros(*ids.shape,2))
    def logits(self, hidden, head):
        scores=torch.full((1,10),-20.)
        scores[0,self.tokens[min(self.next_index,len(self.tokens)-1)]]=20
        self.next_index+=1
        return scores
    def _prompt_tokens(self, question):return torch.tensor([5]),torch.tensor([0])
    def _merge(self, ids, visual):return torch.zeros(2,2),torch.arange(2),0,0
    def _text_forward(self, embeds, valid, positions, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.zeros(1,embeds.shape[1],2),past_key_values=())

def implementation():
    try: module=importlib.import_module("ground3d_migration.unconstrained")
    except ModuleNotFoundError: pytest.fail("Missing independent unconstrained generation implementation")
    return module.generate_one_unconstrained

def test_no_masks_no_forced_labels_and_exact_raw_ids():
    lm=TinyLM([4,1,6,7])
    result=implementation()(lm,"question",visual=object())
    assert result["generated_token_ids"]==[4,1,6,7]
    assert result["answer"]=="<vision></obj> "
    assert result["format_failure"] is True
    assert result["stop_reason"]=="eos"
    assert result["budget_forced"] is False
    assert lm.recorded[0]["generated_token_ids"]==[4,1,6,7]

def test_empty_eos_is_preserved_without_obj_insertion():
    lm=TinyLM([7])
    result=implementation()(lm,"question",visual=object())
    assert result["answer"]==""
    assert result["generated_token_ids"]==[7]
    assert result["format_failure"]

def test_phrase_longer_than_32_not_forced_closed_and_budget_kept_raw():
    lm=TinyLM([0]+[5]*34,budget=35)
    result=implementation()(lm,"question",visual=object())
    assert result["generated_token_ids"]==[0]+[5]*34
    assert result["answer"]=="<obj>"+"chair"*34
    assert result["stop_reason"]=="max_new_tokens"
    assert result["format_failure"]

def test_valid_wrong_obj_is_not_replaced_with_teacher():
    lm=TinyLM([0,5,1,7])
    result=implementation()(lm,"question",visual=object())
    assert result["answer"]=="<obj>chair</obj>"
    assert result["format_failure"] is False

def test_undefined_id_is_not_silently_hidden_as_valid():
    lm=TinyLM([0,5,9,1,7])
    result=implementation()(lm,"question",visual=object())
    assert result["generated_token_ids"]==[0,5,9,1,7]
    assert result["format_failure"]
    assert result["undefined_token_ids"]==[9]

def test_real_model_chooses_existing_zero_obj_and_keeps_raw_text():
    from reason3d_unconstrained_model import ObjAncSegmentationModel
    lm=TinyLM([7]); lm.generate_one=MethodType(implementation(),lm)
    model=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(model)
    model.answer_lm=lm
    batch={"question":["q"],"teacher_answer":["<obj>table</obj>"]}
    answers,rows,indices,reasons,attempted,ratio=model._choose_answers(batch,[SimpleNamespace(compressed=object())],"generated")
    assert answers==[""] and attempted==[""]
    assert reasons==["zero_obj:format_failure"] and rows==[[0]] and indices==[[0]]
    assert ratio==0.

def test_failed_raw_stays_in_full_denominator():
    from evaluate_reason3d_unconstrained import MetricBook,add_failed_raw_obj,merge_mode_payloads
    book=MetricBook(); add_failed_raw_obj(book)
    payload={"metrics":book.state_dict(),"answer_counts":{"samples":1,"format_valid":0},
        "fallback":{"zero_obj:format_failure":1},"examples":[],
        "raw_metrics":book.state_dict(),"raw_counts":{"attempted":1,"format_unevaluable":1},
        "zero_obj_metrics":book.state_dict()}
    summary,_=merge_mode_payloads([payload],20)
    assert summary["samples"]==1
    assert summary["raw_generated_no_fallback"]["denominators"]["obj/selected_iou"]==1
    assert summary["raw_generated_no_fallback"]["obj/selected_iou"]==0.
    assert summary["zero_obj_rate"]==1.


def test_replay_keeps_generated_point_token_as_text():
    from ground3d_migration import unconstrained
    merge=getattr(unconstrained,"merge_prompt_visual",None)
    assert callable(merge), "Replay must not confuse generated point token with prompt insertion"
    embedding=nn.Embedding.from_pretrained(torch.arange(20,dtype=torch.float32).reshape(10,2))
    language=nn.Module()
    language.embedding=embedding
    language.get_base_model=lambda:SimpleNamespace(get_input_embeddings=lambda:embedding)
    lm=SimpleNamespace(language=language,point_token_id=4,
        structured=SimpleNamespace(embed=lambda ids,base:base(ids)))
    ids=torch.tensor([5,4,5,4,6])
    visual=SimpleNamespace(hidden=torch.tensor([[100.,101.],[102.,103.]]))
    merged,positions,marker,shift=merge(lm,ids,visual)
    assert merged.tolist()==[[10.,11.],[100.,101.],[102.,103.],[10.,11.],[8.,9.],[12.,13.]]
    assert positions.tolist()==[0,1,2,3,4,5]
    assert marker==1 and shift==1

def test_subsets_keep_failures_and_weighted_overall_without_mutating_examples():
    from evaluate_reason3d_unconstrained import MetricBook, add_failed_raw_obj, merge_mode_payloads, ExactDistributedEvalSampler
    from reason3d_eval_subsets import merge_source_payloads
    def payload(count,index):
        book=MetricBook()
        for _ in range(count):add_failed_raw_obj(book)
        return {"metrics":book.state_dict(),"answer_counts":{"samples":count,"format_valid":0},
            "fallback":{"zero_obj:format_failure":count},
            "examples":[{"evaluation_index":index,"sample_id":str(index)}],
            "raw_metrics":book.state_dict(),"raw_counts":{"attempted":count,"format_unevaluable":count},
            "zero_obj_metrics":book.state_dict()}
    gathered=[{"generated":{"scannet":payload(1,0),"mp3d":payload(2,308)}}]
    overall,examples,subsets=merge_source_payloads(gathered,["generated"],{"scannet":1,"mp3d":2},merge_mode_payloads,None,20)
    assert overall["generated"]["samples"]==3
    assert overall["generated"]["raw_generated_no_fallback"]["denominators"]["obj/selected_iou"]==3
    assert subsets["scannet"]["generated_answer"]["samples"]==1
    assert subsets["mp3d"]["generated_answer"]["samples"]==2
    assert "evaluation_index" in gathered[0]["generated"]["scannet"]["examples"][0]
    for n in (308,837):
        shards=[list(ExactDistributedEvalSampler(range(n),rank,4)) for rank in range(4)]
        flat=[i for shard in shards for i in shard]
        assert len(flat)==len(set(flat))==n and sorted(flat)==list(range(n))
