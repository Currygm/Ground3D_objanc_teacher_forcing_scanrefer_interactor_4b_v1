import copy,json,math
from pathlib import Path
import pytest,yaml
import reason3d_runtime as runtime

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml"
EXPECTED=json.loads((ROOT/"docs/source_checkpoint.json").read_text())["architecture_signature"]

def config():
    return yaml.safe_load(BASE.read_text())

def load(c,tmp_path):
    f=tmp_path/"candidate.yaml"
    f.write_text(yaml.safe_dump(c),encoding="utf-8")
    return runtime.load_segmentation_config(f)

def test_config_accepts_scalar_tuning_without_changing_protocol(tmp_path):
    c=config()
    c["training"].update(qwen_lora_lr=2e-4,interactor_lr=3e-4,structured_token_lr=2e-5,
        answer_lm_weight=5.,obj_ce_weight=7.,adapter_lr=5e-5,decoder_lr=5e-5)
    c["loss"]["aux_weight"]=.1
    c["ranking"]["weight"]=1.5
    actual=load(c,tmp_path)
    assert actual["loss"]["aux_weight"]==.1
    assert actual["training"]["qwen_lora_lr"]==2e-4
    assert actual["ranking"]["weight"]==1.5

@pytest.mark.parametrize("section,key,value",[
    ("training","qwen_lora_lr",-1.),("training","interactor_lr",float("nan")),
    ("training","decoder_lr",0.),("training","obj_ce_weight",float("inf")),
    ("training","answer_lm_weight",-1.),("training","cosine_min_lr_ratio",1.1),
    ("loss","aux_weight",-1.),("ranking","temperature",0.),
    ("matcher","dice_weight",-1.),("loss","span_weight",True)])
def test_rejects_invalid_scalars(tmp_path,section,key,value):
    c=config();c[section][key]=value
    with pytest.raises(ValueError):load(c,tmp_path)

@pytest.mark.parametrize("section,key,value",[
    ("decoder","num_queries",51),("point_encoder","output_dim",999),
    ("interactor","num_queries",64),("training","joint_epochs",15),
    ("model","generation",{"constrained":True}),("ranking","scope","obj_only"),
    ("loss","focal_gamma",3.),("matcher","span_pooling","max")])
def test_structure_and_fixed_protocol_still_rejected(tmp_path,section,key,value):
    c=config();c[section][key]=value
    with pytest.raises(ValueError):load(c,tmp_path)

def test_source_normalization_changes_only_approved_scalar_fields():
    actual=copy.deepcopy(EXPECTED)
    actual["loss"]["aux_weight"]=.1;actual["ranking"]["weight"]=1.5
    original=copy.deepcopy(actual)
    normalized=runtime.normalize_source_training_signature(actual,EXPECTED)
    assert normalized==EXPECTED
    assert actual==original
    assert normalized is not actual
    for section,key,value in [("decoder","num_queries",51),("loss","focal_gamma",3.),
                              ("ranking","scope","obj_only")]:
        bad=copy.deepcopy(actual);bad[section][key]=value
        assert runtime.normalize_source_training_signature(bad,EXPECTED)!=EXPECTED

def test_source_invalid_scalar_is_rejected():
    c=copy.deepcopy(EXPECTED);c["loss"]["aux_weight"]=float("nan")
    with pytest.raises(ValueError):runtime.normalize_source_training_signature(c,EXPECTED)

def test_untouched_baseline_config_and_signature_remain_compatible(tmp_path):
    c=load(config(),tmp_path)
    assert c["training"]["qwen_lora_lr"]==1e-4
    assert runtime.normalize_source_training_signature(EXPECTED,EXPECTED)==EXPECTED
