# Reason3D auxiliary-loss weight 0.1 ablation

Independent no-Interactor Ground3D project on SSH 224. Reinitialize from the
original ScanRefer epoch010_step45720 checkpoint, not a Reason3D checkpoint.
Pinned source SHA256: 03ee673ab966db40dcbb29b1991500326b47b34b158ed6d2ddb3e9104e740426.

Only experiment change: auxiliary loss weight 0.5 -> 0.1. Auxiliary losses are
still summed over all eight intermediate Decoder layers (not averaged).
Final segmentation, ranking, CE3/OBJCE3, matcher, batch, seed, AdamW and all
learning rates match the ORIGINAL baseline, not the Joint-LR experiment.
Answer3 + Joint10; continuous 13-epoch cosine to10%, no added warmup.
Language/token/visual LR 1e-5; adapters/Decoder LR 1e-4; batch1/GPU, accumulation2,
four GPUs1,2,4,6, BF16, seed42, weight decay0.01 and gradient clipping1.
Save global epochs3/8/13; evaluate final13 on the same pinned GPU UUIDs.

Parent signature validation permits precisely aux_weight0.5->0.1; all other
architecture, base-manifest, tokenizer, task-state and parent-hash checks remain.
New checkpoints retain native aux_weight0.1 signatures for strict resume/eval.
Only checkpoints from THIS experiment can be used with --resume.

```bash
cd /home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_reason3d_from_scanrefer_ep10_4b_aux01_v1
GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh
```

The automatic evaluator requires a complete epoch013 checkpoint. Evaluation is
raw greedy without formatting constraints or GT-answer fallback. Failed answers
use zero OBJ hidden at the 2560-dimensional pre-adapter boundary. Nested strictRaw
counts failures as zero over full denominators. Outputs include comprehensive
summary.json, by_dataset, summary_scannet.json (308), summary_mp3d.json (837)
and raw generation traces, for all1145 validation samples.
Expected result directory:
evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset

For resume, append --resume /absolute/path/to/THIS_PROJECT/epoch_008_step_00001344.pt
to train_then_eval.sh. For independent evaluation:
```bash
GPU_IDS=1,2,4,6 bash scripts/evaluate_unconstrained.sh \
  outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt \
  evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset
```

Official models/environment and native Reason3D data are read-only reused.
Original and Joint-LR projects are untouched. No formal GPU job is launched
during implementation and CPU verification.
