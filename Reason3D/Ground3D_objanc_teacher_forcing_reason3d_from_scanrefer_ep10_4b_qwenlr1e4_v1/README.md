# Reason3D Joint vision LR1e-5 + Qwen LoRA LR1e-4

Direct code copy of visfrozen_qwenlr1e4_v1; only restore original visual training
in Joint and use a new project/config identity. Answer3+Joint15 from original
ScanRefer epoch10, not from a Reason3D checkpoint. Optimizer/scheduler/steps reset.

Answer: only Qwen LoRA and structured tokens train, baseLR1e-5. Joint:
visual baseLR1e-5, Qwen LoRA baseLR1e-4, tokens baseLR1e-5,
adapters/Decoder baseLR1e-4. Original continuous18epoch cosine to10%, no added
warmup or stage restart. Visual parameters and BN buffers frozen/eval in Answer,
trainable/train in Joint. Preserve five optimizer groups and Adam moments.
CE3/OBJCE3/Aux0.5; all other model/data/loss/protocol settings unchanged.

At fourGPUs save epochs3/8/13/18; evaluate Joint10 (epoch13 step2184) and Joint15 (epoch18 step3024) after training.
Auto-evaluation uses same pinned GPU UUIDs, complete1145 samples with
ScanNet308/MP3D837, unconstrained teacher/generated, failed zeroOBJ, strictRaw.

Training plus final evaluation:
    GPU_IDS=1,3,4,6 bash scripts/train_then_eval.sh
Resume only this experiment:
    GPU_IDS=1,3,4,6 bash scripts/train_then_eval.sh --resume outputs/reason3d_answer_lm3_objce3_warmup3_joint15/epoch_008_step_00001344.pt
Manual evaluation (fresh output):
    GPU_IDS=1,3,4,6 bash scripts/evaluate_unconstrained.sh outputs/reason3d_answer_lm3_objce3_warmup3_joint15/epoch_018_step_00003024.pt evaluations/reason3d_answer_lm3_objce3_warmup3_joint15/epoch018_unconstrained_manual --modes teacher generated

Models/data/source environment read-only; original experiment untouched.
No automatic GPU launch during setup.
