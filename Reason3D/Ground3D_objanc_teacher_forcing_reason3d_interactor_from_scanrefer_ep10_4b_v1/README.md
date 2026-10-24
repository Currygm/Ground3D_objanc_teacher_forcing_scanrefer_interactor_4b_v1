# Ground3D Interactor: ScanRefer epoch10 -> Reason3D

Independent adaptation. Parent and datasets/models are reused read-only.
Training: 3 Answer-only epochs then 10 joint epochs, Answer CE=3 and OBJ CE=3.
Answer-only updates Qwen LoRA and structured token rows; freezes visual encoder,
Interactor, adapters and UniVLG9. Joint restores all parent task parameters.
AdamW: language/token/visual 1e-5; Interactor/adapters/Decoder 1e-4; cosine over13
epochs to0.1; WD0.01; gradclip1; BF16; batch1/GPU, accumulation2; seed42.
Loss: 6span+5BCE+2Dice+0.5sum(aux)+0.5ranking+3AnswerCE+3OBJCE.
Retain all-superpoint visual route, 32-query Interactor and question+answer
own-position causal Decoder context.

Data: full Reason3D train1339, val1145, native scene superpoints; no cache generation.
Val partitions: ScanNet308 / Matterport3D837. Exact no-padding evaluation shards.
Save global epochs3,8,13 (=Answer3, Joint5, Joint10).
Four GPUs yield168 steps/epoch; final epoch_013_step_00002184.pt.
Resume restores task state, optimizer, scheduler, per-rank RNG and buffers.

Run on224:
    cd /home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_reason3d_interactor_from_scanrefer_ep10_4b_v1
    GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh
    GPU_IDS=1,2,4,6 bash scripts/train_then_eval.sh --resume /absolute/new/project/checkpoint.pt

Independent evaluation:
    GPU_IDS=1,2,4,6 bash scripts/evaluate.sh outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset

Automatic evaluation requires the completed final checkpoint.
GPU UUIDs are retained across train/evaluation; incomplete training does not
evaluate an earlier checkpoint as final.

Evaluation uses unconstrained raw greedy decoding: no grammar, special-token
mask, forced tag closure/EOS. Normal EOS stop/max_new_tokens128 remain.
On format failure retain raw failed answer and append zero2560 OBJ hidden BEFORE
text_adapter; decode without GT-answer fallback. The decoded zeroOBJ predictions
are scored and need not have zero IoU. raw_generated_no_fallback additionally
scores failures as zero over the full1145 denominator. Report both, normal/zeroOBJ
subsets, and raw generations. No phrase-mismatch rejection.

Outputs: summary.json with by_dataset, summary_scannet.json, summary_mp3d.json,
examples.jsonl, raw_generations.rank_XXXXX.jsonl, atomic rank metric files.
Default modes: teacher and generated; strictRaw nested in generated_answer.

CPU verification: docs/cpu_handoff.json. No GPU training/evaluation launched by implementation.

Restart after initialization failure: an empty metrics.jsonl and matching config.json
without any checkpoint are reusable. Nonempty metrics, checkpoints (including .pt.tmp),
or a different config still block accidental overwrites. No files are auto-deleted.


NCCL startup compatibility (2026-10-22):
Training now performs a disposable four-GPU communication probe before loading
weights: ALLGATHER, Answer/Joint DDP reconstruction/backward with exact averaged
gradient checks, 1M-element ALLREDUCE and broadcast.
Default NCCL_PROFILE=auto tests native transport, then socket-only if it fails.
Successful transport settings are sourced by the training shell; GPU UUID order
is unchanged. A failed preflight stops before model initialization.
NCCL_PROFILE=socket directly uses the validated conservative profile:
Socket network over lo, IB/P2P/SHM disabled, cuMem allocation disabled.
This may reduce communication performance; model/data/loss/optimizer are unchanged.
Probe workers have a45-second collective timeout and240-second overall limit
(including imports). Only the probe's own process group is stopped on failure.
Training NCCL operations time out after600 seconds, configurable using
NCCL_TRAIN_TIMEOUT_SECONDS. Gloo checkpoint/control behavior remains unchanged.

Check communications only (does not start training):
    GPU_IDS=1,2,4,6 NCCL_PROFILE=auto bash scripts/check_nccl.sh
Force the compatible path when retrying a failing node:
    GPU_IDS=1,2,4,6 NCCL_PROFILE=socket bash scripts/train_then_eval.sh

Both native and socket profiles were validated on224 GPUs1,2,4,6.
The old timeout was logged on228; its exact old transport failure was not
reproduced on that host. Do not treat224 validation as a test on228.
Evidence: docs/fixes/20261022_nccl/*/summary.json.
