"""Select a genuinely completed final checkpoint before automatic evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from reason3d_runtime import load_segmentation_config,OBJECTIVE
from train_reason3d import data_manifest


def select_final_checkpoint(config,output,completed_epochs=None):
    total=int(config["training"]["epochs"])
    target=total if completed_epochs is None else int(completed_epochs)
    answer=int(config["training"]["answer_warmup_epochs"])
    if not answer<target<=total:raise ValueError("Requested epoch is outside the Joint schedule")
    output=Path(output).resolve(strict=True)
    root=Path(config["project"]["artifact_root"]).resolve(strict=True)
    if not output.is_relative_to(root):raise ValueError("Output escapes project")
    paths=list(output.glob(f"epoch_{target:03d}_step_*.pt"))
    if len(paths)!=1:raise ValueError("Need exactly one completed final checkpoint before evaluation")
    path=paths[0].resolve(strict=True)
    if not path.is_relative_to(output):raise ValueError("Checkpoint escapes output")
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint.get("objective")!=OBJECTIVE or checkpoint.get("config")!=config:raise ValueError("Final checkpoint objective/config differs")
    if (checkpoint.get("completed_epochs"),checkpoint.get("answer_training_phase"),checkpoint.get("stage_epoch"))!=(target,"joint",target-answer):raise ValueError("Final joint training is incomplete")
    if checkpoint.get("data_sha256")!=data_manifest(config):raise ValueError("Final dataset hash differs")
    return path


if __name__=="__main__":print(select_final_checkpoint(load_segmentation_config(sys.argv[1]),sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else None))
