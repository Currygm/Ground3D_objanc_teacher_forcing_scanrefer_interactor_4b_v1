# Reason3D Interactor with separate Adapter learning rates

Copied from the completed Answer/Interactor alignment + Qwen warmup project.
No new model, loss, dataset, scheduler, or evaluation implementation.
Original source project and data/model/environment assets remain read-only.

Initialize ScanRefer Interactor epoch10 step45720, SHA256
3b4a70ce794d9db32a1d3183b4d288420eed75c15188c489f9f9198659e6e0d3.
Reset optimizer/scheduler/epoch/step; do not resume another Reason3D experiment.

Answer3: Qwen LoRA + original allowed Interactor + structured tokens train.
Visual encoder/Adapter/Decoder frozen. Qwen linear warmup peaks1e-4 on the
last Answer update. Joint10: restore original task training boundary.
Joint Qwen cosine1e-4 to1e-5; other groups retain continuous13epoch cosine10%.

Base rates:
- Qwen1e-4, Interactor1e-4, structuredtokens1e-5, Jointvisual1e-5.
- text_adapter_lr1e-4, visual_adapter_lr5e-5, decoder_lr5e-5.
- adapter_lr1e-4 is only the legacy fallback; explicit paired fields select
  two independent optimizer groups. Old configs retain the combined group.
- CE3/OBJCE3, span6/BCE5/Dice2/Aux0.5/ranking0.5; matching2/15/6.

Native batch1/GPU, accumulation2, BF16, seed42, AdamW WD0.01, clip1.
Four GPUs:168updates/epoch, total2184. Save global3/8/13.
Final13 native evaluation: full1145, unconstrained teacher/generated, failed
zeroOBJ and strictRaw full denominator; ScanNet308/MP3D837 summaries.
GPU UUID pinning, allocator fix, NCCL preflight, rank buffers/RNG unchanged.

Train and automatically evaluate:
    GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh
Resume only this experiment:
    GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh --resume outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_008_step_00001344.pt
Independent evaluation:
    GPU_IDS=1,2,4,6 bash scripts/evaluate.sh outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_manual --modes teacher generated

Only CPU validation performed during setup. No GPU job automatically started.
