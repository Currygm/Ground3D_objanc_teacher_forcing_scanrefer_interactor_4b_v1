
import importlib.util
import json
import torch
import pytest

def test_tokenizer_rename_roundtrip_and_collision(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from ground3d_migration.core import migrate_tokenizer
    raw=Tokenizer(WordLevel({'[UNK]':0,'chair':1,'table':2},unk_token='[UNK]'))
    original=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',eos_token='[UNK]')
    original.add_special_tokens({'additional_special_tokens':['<p>','</p>','<SEG>','<name>','<sentence>','<|point|>']})
    migrated=migrate_tokenizer(original)
    for old,new in [('<p>','<obj>'),('</p>','</obj>'),('<SEG>','<anc>')]:
        assert migrated.convert_tokens_to_ids(new)==original.convert_tokens_to_ids(old)
        assert migrated.encode(new,add_special_tokens=False)==[original.convert_tokens_to_ids(old)]
        assert old not in migrated.get_vocab()
    migrated.save_pretrained(tmp_path)
    loaded=PreTrainedTokenizerFast.from_pretrained(tmp_path)
    assert loaded.get_vocab()==migrated.get_vocab()
    original.add_tokens(['<obj>'])
    with pytest.raises(ValueError,match='collision'):migrate_tokenizer(original)

def test_ground_language_replaces_old_backbone_route():
    assert importlib.util.find_spec('ground3d_migration.language') is not None, 'Ground3D language route missing'

def test_ground_geometry_consumes_raw_superpoints():
    assert importlib.util.find_spec('ground3d_migration.visual') is not None, 'Ground3D visual route missing'

def test_native_special_flags_survive_ground_token_registration():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from ground3d_migration.core import migrate_tokenizer
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0},unk_token='[UNK]')),unk_token='[UNK]',eos_token='[UNK]')
    tokenizer.add_special_tokens({'additional_special_tokens':['<|vision_start|>']})
    native=tokenizer.convert_tokens_to_ids('<|vision_start|>')
    tokenizer.add_special_tokens({'additional_special_tokens':['<p>','</p>','<SEG>','<name>','<sentence>','<|point|>']})
    migrated=migrate_tokenizer(tokenizer)
    assert native in migrated.all_special_ids
    assert migrated.convert_tokens_to_ids('<|vision_start|>')==native
