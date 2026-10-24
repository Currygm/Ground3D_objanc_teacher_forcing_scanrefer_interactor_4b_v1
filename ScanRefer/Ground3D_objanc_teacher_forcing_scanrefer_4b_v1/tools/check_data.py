import json,sys
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
from spatiallm_compressed_visual_objanc_full_attention.data.segmentation_dataset import build_segmentation_dataset
config=load_segmentation_config(PROJECT/'configs/scanrefer_teacher_forcing_univlg9.yaml')
report={}
for split,expected in [('train',36569),('val',9483)]:
    data=build_segmentation_dataset(config,split);assert len(data)==expected
    row=data[0];g=row['ground_geometry']
    n=len(row['point_cloud']);s=len(row['superpoint_coord'])
    assert len(row['point_to_superpoint'])==n
    assert g['voxel_features'].shape==(len(g['voxel_coord']),6)
    assert row['gt_point_masks'].shape[-1]==n
    assert row['gt_superpoint_masks'].shape[-1]==s
    assert torch.equal(g['point_to_superpoint'],row['point_to_superpoint'])
    assert torch.allclose(g['superpoint_coord'],row['superpoint_coord'])
    report[split]={'samples':len(data),'scene':row['scene_id'],'raw_points':n,'voxels':len(g['voxel_coord']),'superpoints':s,'entities':len(row['grounding_object_ids'])}
(PROJECT/'docs/data_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
