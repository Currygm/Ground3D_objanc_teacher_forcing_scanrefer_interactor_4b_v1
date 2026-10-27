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
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_model import ObjAncSegmentationModel
    lm=TinyLM([7]); lm.generate_one=MethodType(implementation(),lm)
    model=ObjAncSegmentationModel.__new__(ObjAncSegmentationModel);nn.Module.__init__(model)
    model.answer_lm=lm
    batch={"question":["q"],"teacher_answer":["<obj>table</obj>"]}
    answers,rows,indices,reasons,attempted,ratio=model._choose_answers(batch,[SimpleNamespace(compressed=object())],"generated")
    assert answers==[""] and attempted==[""]
    assert reasons==["zero_obj:format_failure"] and rows==[[0]] and indices==[[0]]
    assert ratio==0.

def test_failed_raw_stays_in_full_denominator():
    from evaluate_segmentation import MetricBook,add_failed_raw_obj,merge_mode_payloads
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

def test_entrypoint_installs_generation_on_model_instance_only(tmp_path):
    try: from evaluate_segmentation_unconstrained import install_generation,finalize_summary
    except ModuleNotFoundError:pytest.fail("Missing independent unconstrained evaluator")
    lm=TinyLM([7])
    model=SimpleNamespace(answer_lm=lm)
    sink=[]
    install_generation(model,sink.append)
    lm.generate_one("q",visual=object())
    assert sink[0]["generated_token_ids"]==[7]
    import json
    (tmp_path/"summary.json").write_text(json.dumps({"validation_samples":1}))
    finalize_summary(tmp_path,world_size=1)
    protocol=json.loads((tmp_path/"summary.json").read_text())["generation_protocol"]
    assert protocol["format_constraints"] is False
    assert protocol["special_token_suppression"] is False
    assert protocol["format_failure_policy"]=="existing_zero_obj_decoder"

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
