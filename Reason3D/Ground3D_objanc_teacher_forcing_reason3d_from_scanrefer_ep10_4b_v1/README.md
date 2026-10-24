# Ground3D OBJ/ANC Reason3D from ScanRefer epoch10

New independent Reason3D fine-tuning project. Source code, source checkpoints,
official base model files and the compatible source environment are not modified.

## Recipe

- Initial task state: Ground3D_objanc_teacher_forcing_scanrefer_4b_v1/outputs/
  scanrefer_teacher_forcing_univlg9/epoch_010_step_00045720.pt.
  SHA256: 03ee673ab966db40dcbb29b1991500326b47b34b158ed6d2ddb3e9104e740426.
  All 875 task tensors, including visual weights/BN buffers and adapters/decoder,
  are restored. Original verified official frozen base and tokenizer are retained.
- Original architecture: Qwen3-VL-4B, Ground3D SpConv visual encoder and point2token,
  full question+answer causal hidden states, adapters and UniVLG9 decoder.
- Reason3D full train1339 and val1145, with native scene superpoints. No new
  superpoint segmentation or normal cache. Geometry cache is specific to the
  original Ground3D 0.02m voxel/point-feature path and lives in this new project.
- Fresh AdamW, WD .01, clip1, BF16, language gradient checkpointing, batch1/card,
  accumulation2. Qwen LoRA/task rows/visual LR1e-5; adapters/decoder1e-4.
  One continuous 13-epoch cosine schedule to10%; no stage LR restart.
- Global epochs1-3: Answer-only language LoRA + structured token rows. Visual
  encoder/BN, adapters and decoder frozen. Save only completed epoch3.
- Global epochs4-13: original joint trainable boundary, retaining all transferred
  visual/adapters/decoder state. Save completed global8 and13 (joint5 and10).
- Both phases: full Answer CE weight3 + independent OBJ-content CE weight3.
  OBJ CE uses the same causal logits and excludes tags, ANC, template and EOS.
  Joint also retains span6/BCE5/Dice2, auxiliary .5 and ranking .5(all OBJ/ANC).
- Four default physical GPUs1,2,4,6:168 optimizer steps/epoch; checkpoints:
  epoch_003_step_00000504.pt, epoch_008_step_00001344.pt,
  epoch_013_step_00002184.pt. Relative names change if GPU world size changes.

## Run

```bash
PROJECT=/home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_reason3d_from_scanrefer_ep10_4b_v1
GPU_IDS=1,2,4,6 bash "$PROJECT/scripts/train_then_eval.sh"
# Resume from a checkpoint of THIS new experiment (same GPU count):
GPU_IDS=1,2,4,6 bash "$PROJECT/scripts/train_then_eval.sh" --resume \
  "$PROJECT/outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_008_step_00001344.pt"
# Training only:
GPU_IDS=1,2,4,6 bash "$PROJECT/scripts/train.sh"
# Evaluation only, checkpoint and output directory are positional:
GPU_IDS=1,2,4,6 bash "$PROJECT/scripts/evaluate.sh" \
  "$PROJECT/outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt" \
  "$PROJECT/evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_by_dataset" \
  --modes teacher generated generated_no_fallback
```

The one-shot launcher freezes GPU UUIDs once and uses those same cards for
training and evaluation. It refuses to evaluate if final13 is absent/incomplete.
Optional OUTPUT_DIR/EVAL_OUTPUT_DIR choose other directories within this project.
Default Python is the existing source project env/bin/python; packages are reused
read-only, not installed or changed. This project therefore depends on that
source environment and its verified official model files remaining available.

## Evaluation output

- summary.json: full1145, with by_dataset.scannet and by_dataset.mp3d.
- summary_scannet.json: ScanNet308, all requested modes.
- summary_mp3d.json: Matterport3D837, all requested modes.
- examples.jsonl: selected examples in global sample order.

Teacher and generated with reference fallback are diagnostic. For autonomous
performance use generated_no_fallback. That path reads no GT until inference
and query selection finish; invalid predictions score0 over the full subset.
Source metrics are merged from raw sums/counts, not averaged across datasets.
Exact shards have no padding; atomic rank files avoid inference-time collective
waits when scenes or rank durations differ. The per-run rank directory prevents
old shard reuse.

## Verification

CPU tests and real source handoff are recorded under docs/. No GPU training or
inference job is launched during project creation. A CPU handoff check validates
checkpoint compatibility and does not certify peak GPU memory on large scenes.
