from pathlib import Path
import pytest
import yaml
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config

ROOT = Path(__file__).resolve().parents[1]

def test_fresh_3b_config_needs_no_05b_checkpoint():
    config = load_segmentation_config(ROOT / 'configs/scanrefer_teacher_forcing_univlg9.yaml')
    assert config['initialization']['mode'] == 'official_ground3d_3d'
    assert config['decoder']['text_dim'] == 2048
    assert config['training']['epochs'] == 10
    assert [s['name'] for s in config['data']['train_sources']] == ['scanrefer']
    assert 'answer_checkpoint' not in config['initialization']
    assert 'source_segmentation_checkpoint' not in config['initialization']

def test_config_rejects_text_dimension_mismatch(tmp_path):
    config = yaml.safe_load((ROOT / 'configs/scanrefer_teacher_forcing_univlg9.yaml').read_text())
    config['decoder']['text_dim'] = 896
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_segmentation_config(path)

