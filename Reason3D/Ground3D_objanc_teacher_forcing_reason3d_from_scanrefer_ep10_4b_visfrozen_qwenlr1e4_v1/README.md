# Reason3D frozen vision + Joint Qwen LoRA 1e-4

Copied from the original no-Interactor Ground3D Reason3D baseline. Original
model/data/environment paths are reused read-only. No training output or cache
was copied.

Answer3 + Joint10; CE3/OBJCE3; Aux0.5. Vision parameters have no gradients in
either stage and its running buffers stay in eval mode. Visual Adapter remains
trainable during Joint. The five original optimizer groups are retained; the
visual group has LR0 and never receives gradients or AdamW state.

The baseline 13-epoch cosine (minimum ratio0.1) continues across both stages.
Qwen LoRA uses base1e-5 in Answer, base1e-4 in Joint; structured tokens use1e-5;
adapters/Decoder use1e-4 in Joint. No new warmup or optimizer-state reset at
the phase boundary. At four GPUs:168 steps/epoch, Joint boundary504, final2184.
Qwen LR at the Joint boundary is approximately8.87e-5, ending at1e-5.

Save global epochs3/8/13. The launcher evaluates only a verified completed final
checkpoint on the same pinned GPU UUIDs, with teacher and unconstrained generated
answers. Failed generation uses zeroOBJ; strictRaw also reports failure as zero
over the full1145 samples. Outputs include overall/ScanNet308/MP3D837 summaries.

Training and automatic evaluation:
    GPU_IDS=1,3,4,6 bash scripts/train_then_eval.sh
Resume this experiment only:
    GPU_IDS=1,3,4,6 bash scripts/train_then_eval.sh --resume outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_008_step_00001344.pt
Independent evaluation:
    GPU_IDS=1,3,4,6 bash scripts/evaluate_unconstrained.sh outputs/reason3d_answer_lm3_objce3_warmup3_joint10/epoch_013_step_00002184.pt evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset --modes teacher generated

Default GPU_IDS remains1,2,4,6; set explicitly for intended physical cards.
Do not overwrite an existing nonempty evaluation directory.
