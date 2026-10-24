from types import SimpleNamespace
import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from spatiallm_compressed_visual_objanc_full_attention.model import llm

def test_3b_shape_validation_uses_configuration():
    validate = getattr(llm, 'validate_language_shape', None)
    assert callable(validate), 'missing configuration-based 3B shape validation'
    validate(SimpleNamespace(hidden_size=2048, num_hidden_layers=36),
             {'expected_hidden_size':2048, 'expected_num_hidden_layers':36})
    with pytest.raises(ValueError):
        validate(SimpleNamespace(hidden_size=896, num_hidden_layers=24),
                 {'expected_hidden_size':2048, 'expected_num_hidden_layers':36})

def test_lora_covers_all_36_layers_and_preserves_dense_weights():
    attach = getattr(llm, 'attach_language_lora', None)
    assert callable(attach), 'missing language-only LoRA attachment'
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=48,
         num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2))
    # A point-backbone leaf with the same name must NOT receive a language adapter.
    base.point_backbone = nn.Module(); base.point_backbone.q_proj = nn.Linear(32,32)
    before = base.model.layers[0].self_attn.q_proj.weight.detach().clone()
    model, count = attach(base, {'r':16,'alpha':32,'dropout':.05,'bias':'none',
           'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']})
    assert count == 252
    assert not any('point_backbone' in n and 'lora_' in n for n,_ in model.named_parameters())
    assert torch.equal(before, model.get_base_model().model.layers[0].self_attn.q_proj.base_layer.weight)
    model(torch.tensor([[1,2,3]]), labels=torch.tensor([[1,2,3]])).loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)

def test_2048_structured_tokens_are_tied_and_trainable():
    rows=llm.StructuredTokenRows(2048,[40,41,42,43],40)
    dense=nn.Embedding(40,2048).requires_grad_(False)
    head=nn.Linear(2048,40,bias=False).requires_grad_(False)
    embedded=rows.embed(torch.tensor([[1,40,42]]),dense)
    logits=rows.logits(embedded,head)
    assert logits.shape==(1,3,44)
    logits[...,40:].sum().backward()
    assert rows.rows.weight.grad is not None
    assert rows.output_bias.grad is not None
    assert dense.weight.grad is None

def test_segmentation_loss_also_updates_language_lora():
    from spatiallm_compressed_visual_objanc_full_attention.model.segmentation_decoder import ObjAncFullCycleDecoder
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=40,hidden_size=32,intermediate_size=48,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2))
    language,_=llm.attach_language_lora(base,{'r':16,'alpha':32,'dropout':0.,'bias':'none',
         'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']})
    hidden=language(torch.tensor([[1,2,3]]),output_hidden_states=True).hidden_states[-1][0]
    decoder=ObjAncFullCycleDecoder(visual_dim=8,text_dim=32,hidden_dim=16,num_queries=50,
            num_layers=9,num_heads=4,feedforward_dim=32)
    output=decoder([torch.randn(5,8)],[hidden],[torch.randn(5,3)],[torch.tensor([0,1,1])])
    output['pred_masks'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in language.named_parameters() if 'lora_B' in n)
    assert all(p.grad is None for n,p in language.named_parameters() if 'lora_' not in n)
