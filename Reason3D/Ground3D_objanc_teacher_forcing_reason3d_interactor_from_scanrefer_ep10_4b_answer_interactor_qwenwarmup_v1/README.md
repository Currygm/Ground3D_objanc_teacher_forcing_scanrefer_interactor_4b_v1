# Reason3D Interactor Answer alignment + Qwen warmup

Copied from the original Reason3D Interactor project. Initialize original
ScanRefer Interactor epoch10, reset optimizer/scheduler/step/epoch. Source
models/data/embeddings/environment remain read-only.

Answer3: train original allowed Interactor parameters, Qwen LoRA and structured
tokens. Visual encoder frozen/eval; adapters and Decoder frozen/eval. Preserve
intrinsic Interactor frozen embedding/FFN parameters. Only frozen visual
encoding has no gradients; Interactor and LLM remain differentiable.
Joint10 restores original full task training boundary.

Qwen peakLR1e-4: linear warmup through the last Answer update, then Joint-local
cosine to1e-5. Other groups retain continuous13epoch cosine to10%: Interactor
1e-4, tokens1e-5, visual1e-5, adapters/Decoder1e-4 base. No optimizer reset.
Four GPUs/batch1/accumulation2:168 updates/epoch;504 warmup+1680 Joint updates.
First Qwen update~1.98e-7; last Answer update1e-4; final Joint update1e-5.

CE3/OBJCE3/Aux0.5 and other losses unchanged. Save global3/8/13; final expected
epoch_013_step_00002184.pt. Full1145 unconstrained teacher/generated evaluation,
failure zeroOBJ, strictRaw full denominator, ScanNet308/MP3D837 summaries.
Same pinned UUIDs for training/eval; preserve NCCL preflight and rank buffer/RNG.

Train + evaluate:
    GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh
Resume this experiment only:
    GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh --resume outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_008_step_00001344.pt
Standalone evaluation (fresh output):
    GPU_IDS=1,2,4,6 bash scripts/evaluate.sh outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_manual --modes teacher generated

Setup performs CPU checks only; no GPU task automatically started.
