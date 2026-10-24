# Unconstrained ScanRefer evaluation

Run the independent entry; do not switch the original evaluate.sh or edit the
checkpoint's generation configuration (it is part of the architecture signature).

```bash
cd /home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1
GPU_IDS=3,2 bash scripts/evaluate_unconstrained.sh \
  outputs/scanrefer_teacher_forcing_univlg9/epoch_005_step_00022860.pt \
  evaluations/scanrefer_joint005_unconstrained
```

Use an unused output directory for each run. The script accepts physical GPU
IDs or full GPU UUIDs, and preserves the native evaluator's optional arguments.
For another checkpoint, replace the first positional argument. All validation
samples are evaluated by default. This code does not start any job on its own.

Protocol:
- Same model, checkpoint checks, prompt, visual encoding, teacher reference,
  greedy strategy, total max_new_tokens=128 and native causal Decoder replay.
- Direct argmax over the whole model output at each step; no grammar masks,
  special-token suppression, visible-text forcing, phrase length cap, tag repair,
  forced closing tags, forced EOS or GT replacement.
- Stop on generated EOS or the total token budget. Remove only the terminal
  EOS from the answer passed to the existing parser/replay. Whitespace and
  all other special tokens are retained.
- Token IDs without a tokenizer representation are not masked out: exact IDs
  are archived and the sample is marked failed rather than silently repaired.
- Format failure uses the existing synthetic zero-OBJ Decoder as requested.
  The native strictRaw/no-fallback metric ALSO remains: such failures count as
  zero OBJ localization scores in the complete validation denominator.
- A well-formed answer at the total budget is evaluated as generated, without
  appending EOS or tags; stop_reason records the truncation separately.
- Only the prompt visual marker is expanded during replay. An unconstrained
  generated <|point|> remains an ordinary generated token, without replacement.
- Native parsing is retained, including its existing acceptance of ordinary
  text outside tags. This is not a stricter formatting protocol.

Outputs:
- summary.json: native teacher/generated/strictRaw/normal/zero-OBJ metrics plus
  generation_protocol specifying the unconstrained settings.
- examples.jsonl: native example records.
- raw_generations.rank_XXXXX.jsonl: every original generated answer, full token
  IDs including EOS if generated, raw_text_including_eos, sample ID/index,
  stop reason, parse errors, and undefined IDs. These files are exclusively
  created and never overwritten.
- .rank_results/: native exact, non-padding shard aggregation.

No changes to original model, checkpoint, training, evaluate_segmentation.py or
scripts/evaluate.sh. New executable code only runs through the new entry.

CPU regression command:
```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 env/bin/python -B -m pytest \
  -q -p no:cacheprovider --basetemp=cache/unconstrained_tests_cpu \
  tests/test_unconstrained_evaluation.py
bash -n scripts/evaluate_unconstrained.sh
```
