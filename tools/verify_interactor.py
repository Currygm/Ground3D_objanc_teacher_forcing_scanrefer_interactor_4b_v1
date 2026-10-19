import os,sys,json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import torch
from spatiallm_compressed_visual_objanc_full_attention.config import load_segmentation_config
from ground3d_migration.interactor import GroundInteractor
torch.set_num_threads(1)
c=load_segmentation_config(P/"configs/scanrefer_teacher_forcing_univlg9.yaml")
m=GroundInteractor(c["interactor"]).eval()
with torch.no_grad():
    x=torch.randn(2343,256);out=m([x])[0]
assert out.shape==(32,2560) and torch.isfinite(out).all()
assert not torch.cuda.is_initialized()
report={"device":"cpu","input_superpoints":2343,"output_shape":list(out.shape),"signature":m.signature(),
    "trainable_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad),
    "frozen_parameters":sum(p.numel() for p in m.parameters() if not p.requires_grad),"gpu_initialized":False}
(P/"docs/interactor_cpu_initialization.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="signature"},indent=2))
