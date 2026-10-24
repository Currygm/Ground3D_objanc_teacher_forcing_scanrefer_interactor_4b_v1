"""Configuration loading and strict path-boundary validation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

OBJECTIVE = "spatiallm_compressed_visual_objanc_answer_lm_v1"
ARCHITECTURE_SIGNATURE = {
    "objective": OBJECTIVE,
    "visual_route": "sonata_enc4_native_stride_projector",
    "attention_train": "causal_visual_question_teacher_answer",
    "attention_future_api": "full_visual_question_answer",
    "structured_answer": "single_obj_zero_or_more_anc_v1",
    "decoder": None,
    "fourier_encode": False,
}

def _required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing required configuration value: {key}")
    return value

def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value

def ensure_within(path: str | Path, root: str | Path, *, must_exist: bool) -> Path:
    root_path = Path(root).expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if must_exist:
        resolved = candidate.resolve(strict=True)
    else:
        missing: list[str] = []
        cursor = candidate
        while not cursor.exists():
            missing.append(cursor.name)
            if cursor.parent == cursor:
                raise ValueError(f"Cannot resolve a safe ancestor for {candidate}")
            cursor = cursor.parent
        resolved = cursor.resolve(strict=True).joinpath(*reversed(missing))
    if resolved != root_path and root_path not in resolved.parents:
        raise ValueError(f"Path escapes authorized root {root_path}: {resolved}")
    return resolved

def remap_renamed_data_path(path: Path) -> Path:
    """Resolve the historical Reason3D directory name without writing to it."""
    if path.exists() or not path.is_absolute():
        return path
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part == "Reason3D":
            candidate = Path(*(parts[:index] + ["Reason3D_base"] + parts[index + 1:]))
            if candidate.exists():
                return candidate
    return path


def ensure_within_any(path: str | Path, roots: list[str | Path], *, must_exist: bool) -> Path:
    errors: list[str] = []
    for root in roots:
        try:
            return ensure_within(path, root, must_exist=must_exist)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    raise ValueError(
        f"Path is outside all authorized roots: {path}; "
        + (errors[-1] if errors else "no roots")
    )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    project = _mapping(config, "project")
    code_root = Path(_required(project, "authorized_code_root")).resolve(strict=True)
    artifact_root = Path(_required(project, "artifact_root")).resolve(strict=True)
    expected_artifact = Path("/mnt/conda/huangxiang/guiming/SpatialLM_compressed_visual_objanc_full_attention")
    if artifact_root != expected_artifact:
        raise ValueError(f"artifact_root must be exactly {expected_artifact}")
    project["authorized_code_root"] = str(code_root)
    project["artifact_root"] = str(artifact_root)
    for key in ("output_dir", "evaluation_dir", "cache_dir"):
        project[key] = str(ensure_within(_required(project, key), artifact_root, must_exist=False))
    model = _mapping(config, "model")
    model["spatiallm_repo"] = str(ensure_within(_required(model, "spatiallm_repo"), "/home/yibo/SpatialLM-pretrain", must_exist=True))
    model["model_name_or_path"] = str(ensure_within(_required(model, "model_name_or_path"), "/home/yibo/SpatialLM-pretrain", must_exist=True))
    if model.get("enable_fourier_encode") is not False:
        raise ValueError("This experiment requires model.enable_fourier_encode=false")
    if model.get("attention_mode") != "causal":
        raise ValueError("Answer pretraining must use standard causal attention")
    if list(_required(model, "structured_tokens")) != ["<obj>", "</obj>", "<anc>", "</anc>"]:
        raise ValueError("structured_tokens must be the four lowercase OBJ/ANC tags")
    data = _mapping(config, "data")
    data["scene_root"] = str(ensure_within(_required(data, "scene_root"), code_root, must_exist=True))
    annotation_roots = [code_root, artifact_root]
    for source in _required(data, "train_sources"):
        source["annotations"] = str(
            ensure_within_any(source["annotations"], annotation_roots, must_exist=True)
        )
    data["val_annotations"] = str(
        ensure_within_any(
            _required(data, "val_annotations"), annotation_roots, must_exist=True
        )
    )
    normal = _mapping(data, "normal")
    normal["cache_dir"] = str(ensure_within(_required(normal, "cache_dir"), code_root, must_exist=True))
    if data.get("mixture_epoch_size") != 60000:
        raise ValueError("mixture_epoch_size must remain 60000")
    names = [str(source["name"]) for source in data["train_sources"]]
    weights = [float(source["sampling_weight"]) for source in data["train_sources"]]
    if names != ["scanrefer", "nr3d", "sr3d", "multi3drefer"]:
        raise ValueError("Unexpected training source order")
    if any(abs(a - b) > 1e-9 for a, b in zip(weights, [0.45, 0.25, 0.15, 0.15])):
        raise ValueError("Training source weights must be 45/25/15/15")
    training = _mapping(config, "training")
    if _required(training, "objective") != OBJECTIVE:
        raise ValueError(f"Unexpected objective: {training.get('objective')}")
    if int(_required(training, "epochs")) != 3:
        raise ValueError("Answer pretraining is configured for three epochs")
    return config


def load_segmentation_config(path):
    with Path(path).open('r',encoding='utf-8') as handle:config=yaml.safe_load(handle)
    project=config['project'];root=Path(project['artifact_root']).resolve(strict=True)
    expected=Path('/home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_4b_v1')
    if root!=expected or not root.is_relative_to(Path(project['authorized_code_root']).resolve(strict=True)):raise ValueError('Wrong Ground3D artifact root')
    for key in ['output_dir','evaluation_dir','cache_dir']:project[key]=str(ensure_within(project[key],root,must_exist=False))
    model=config['model']
    for key in ['model_name_or_path','ground3d_weights_dir']:model[key]=str(ensure_within(model[key],root,must_exist=False))
    if config['initialization']!={'mode':'official_ground3d_3d'}:raise ValueError('Wrong initialization')
    if (model['expected_hidden_size'],model['expected_num_hidden_layers'])!=(2560,36):raise ValueError('Wrong Qwen dimensions')
    if model['structured_tokens']!=['<obj>','</obj>','<anc>','</anc>']:raise ValueError('Wrong answer protocol')
    if (model['lora']['r'],model['lora']['alpha'])!=(32,64):raise ValueError('3D checkpoint LoRA mismatch')
    if model['lora']['bias']!='none' or model['lora']['target_modules']!=['q_proj','k_proj','v_proj','o_proj','attn.qkv','attn.proj']:raise ValueError('Official Qwen LoRA target mismatch')
    if model['dtype']!='bfloat16' or model['attention_mode']!='causal' or model['point_prompt']!='<|point|>':raise ValueError('Official Ground3D execution protocol mismatch')
    if config['decoder']['num_heads']!=8:raise ValueError('UniVLG9 requires eight heads')
    if config['point_encoder']!={'route':'official_ground3d_spconv_superpoint','voxel_size':.02,'output_dim':256,'training_cap':2000,'evaluation_sampling':'all_superpoints'}:raise ValueError('Visual route mismatch')
    data=config['data']
    for source in data['train_sources']:ensure_within(source['annotations'],project['authorized_code_root'],must_exist=True)
    for key in ['val_annotations','scene_root']:ensure_within(data[key],project['authorized_code_root'],must_exist=True)
    if data['sampling_mode']!='full_dataset_no_replacement' or data['expected_train_samples']!=36569:raise ValueError('Wrong dataset schedule')
    dec=config['decoder']
    if (dec['num_queries'],dec['num_layers'],dec['hidden_dim'],dec['text_dim'],dec['visual_dim'])!=(50,9,256,2048,512):raise ValueError('Wrong UniVLG9 architecture')
    training=config['training']
    if training['objective']!='ground3d_objanc_teacher_forcing_scanrefer_4b_v1' or training['epochs']!=10 or training['answer_pretrain_epochs']!=3:raise ValueError('Wrong two-stage objective')
    if training['answer_mode']!='teacher' or training['allow_generated_answers']:raise ValueError('Teacher-only training required')
    return config
