# Reason3D Joint learning-rate and unfreeze-warmup ablation

Independent no-Interactor Ground3D experiment on SSH 224.
Initialization: Ground3D_objanc_teacher_forcing_scanrefer_4b_v1/outputs/scanrefer_teacher_forcing_univlg9/epoch_010_step_00045720.pt.
Parent SHA256: 03ee673ab966db40dcbb29b1991500326b47b34b158ed6d2ddb3e9104e740426.

The Answer3 + Joint10 schedule, complete Reason3D train1339/val1145, frozen base,
task state, architecture, loss, optimizer, effective batch8, BF16, seed42 and
gradient clipping1 are unchanged. All source artifacts stay read-only.

Loss: 3 Answer CE + 3 OBJ-content CE; final segmentation 6 span + 5 BCE + 2 Dice,
plus 0.5 times the SUM of auxiliary-layer losses and 0.5 ranking. No loss ablation.

The first 504 optimizer updates (Answer3 at 168 updates/epoch on four GPUs)
have exactly the baseline learning rates and freeze boundaries. They use
the original 13-epoch cosine trajectory. Joint resets the LR trajectory only;
the existing AdamW language/token moments are retained.
Joint peak rates:
- Qwen LoRA and structured tokens: 5e-6, immediate stage-local cosine.
- Ground visual encoder / point2token: 3e-6.
- Text/visual adapters and Decoder: 3e-5.
The newly unfrozen visual/adapters/Decoder groups linearly warm up over their
first 100 updates (starting at peak/100), then cosine decay. All Joint groups
end at 10% of their peak. State and schedule resume exactly from saved optimizer
and LambdaLR state; do not resume a baseline experiment checkpoint into this run.

Save global epochs3/8/13 only (Answer final, Joint5, Joint10). The automatic
entrypoint requires a genuinely completed epoch13 checkpoint before evaluating.
Expected four-GPU final name: epoch_013_step_00002184.pt.

Run:
```bash
cd /home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_reason3d_from_scanrefer_ep10_4b_jointlr_warmup_v1
GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh
```

Resume THIS experiment:
```bash
GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh --resume /absolute/path/to/this_experiment_checkpoint.pt
```

Standalone evaluation:
```bash
GPU_IDS=1,2,4,6 bash scripts/evaluate_unconstrained.sh \
  outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt \
  evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset
```

GPU numeric IDs are resolved once to UUIDs and reused by training/evaluation.
All caches/logs/results/checkpoints are independent. Native frozen model/env
paths and original Reason3D annotations/scenes are reused read-only.
Evaluation has no grammar/forced closure/special-token suppression/GT fallback.
Failures use a zero OBJ vector at the 2560-dimensional pre-adapter boundary;
nested strictRaw scores failures zero over the full sample denominator.
Outputs include summary.json with by_dataset, summary_scannet.json (308),
summary_mp3d.json (837), and raw generated IDs/text per rank.
The original config generation flag stays unchanged for parent architecture
compatibility; the independent evaluator installs raw decoding on the instance.

Implementation does not automatically launch GPU training or evaluation.
