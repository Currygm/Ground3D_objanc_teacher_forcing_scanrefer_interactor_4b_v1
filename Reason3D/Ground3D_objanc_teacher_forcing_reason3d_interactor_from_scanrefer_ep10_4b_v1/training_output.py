"""Protect real training progress while permitting a failed empty initialization."""
import json
from pathlib import Path

def validate_fresh_output(output,config):
    output=Path(output).resolve()
    if not output.exists():return "fresh"
    checkpoints=list(output.glob("*.pt"))+list(output.glob("*.pt.tmp"))
    if checkpoints:
        raise FileExistsError(f"Checkpoint exists in {output}; use --resume or a fresh OUTPUT_DIR")
    metrics=output/"metrics.jsonl"
    saved=output/"config.json"
    for path in (metrics,saved):
        if not path.resolve().is_relative_to(output):
            raise ValueError("Training metadata symlink escapes output directory")
    if metrics.exists() and (not metrics.is_file() or metrics.stat().st_size>0):
        raise FileExistsError(f"Nonempty metrics.jsonl in {output}; use --resume or a fresh OUTPUT_DIR")
    if saved.exists():
        try:previous=json.loads(saved.read_text(encoding="utf-8"))
        except (OSError,ValueError) as exc:
            raise FileExistsError("Existing config.json is unreadable; use a fresh OUTPUT_DIR") from exc
        if previous!=config:
            raise FileExistsError("Existing config differs; use a fresh OUTPUT_DIR")
    return "empty_initialization" if metrics.exists() or saved.exists() else "fresh"
