# Ground3D OBJ/ANC ScanRefer 4B

Independent migration project. Source experiments, annotations and scene/superpoint caches are read-only.

## Architecture

`raw XYZ+RGB -> 0.02m voxel mean -> pretrained SpConvUNet -> raw-point expansion -> superpoint mean -> pretrained x_mask [S,256]`.
The language branch uses center-only random-start FPS (training cap 2000; evaluation all), pretrained point2token, and Qwen3-VL-4B with official LoRA r32/alpha64. BatchNorm freeze is independent of training sampling.
The decoder branch always uses all pre-LLM features through trainable Linear(256,512). Full question and answer hidden states pass through trainable Linear(2560,2048) to the original UniVLG9 (50 queries, nine layers, hidden256, eight heads). No phrase averaging or two-entity limit.

## Answer and weights

Official repos: `amolharsh/Ground3D-LMM-4B-3D` and `Qwen/Qwen3-VL-4B-Instruct`. Complete download manifests pin repository revisions, SHA256 and official LFS hashes. Missing/incomplete/mismatched files reject initialization. Original mask parameters are excluded by an explicit key list; every retained component loads with exact keys/shapes and finite tensor checks.
Task IDs: `<obj>`151669, `</obj>`151670, `<anc>`151671, `</anc>`151675. First three retain original IDs and pretrained rows; `</anc>` copies `</obj>` even though its new ID is inside Qwen's padded vocabulary. Independent input/output overlays permit only four task rows to update; dense embeddings/head remain frozen. Tokenizer snapshots and protocol hashes are saved in task checkpoints.
Ground3D archives include MMEngine training metadata. The restricted archive reader allows only listed tensor, NumPy and inert metadata globals; it never imports MMEngine or permits arbitrary pickle globals. This is separate from component loading, which is strict.

## Training

Answer: three epochs, unscaled CE; language LoRA and four token rows LR1e-5; all pretrained visual modules frozen with frozen BatchNorm statistics. New adapters/decoder do not exist in this stage.
Joint: ten epochs; language/rows and pretrained visual modules LR1e-5; fresh adapters/decoder LR1e-4; original final/auxiliary segmentation and ranking losses plus 0.1 Answer CE. Optimizer/scheduler restart on handoff; within-stage resume restores their state and per-rank Python/NumPy/Torch/CUDA RNG. Visual buffers are saved as well as parameters.
BF16, gradient checkpointing, batch1/device, accumulation2. Train/val:36569/9483. Independent `cache/geometry` fingerprints only coordinates, RGB and superpoint IDs; no Sonata voxel or normal cache is consumed.
Training/teacher evaluation use one causal answer pass. Generated evaluation uses cached generation, then one causal pass for the chosen full answer (GT only for the reference fallback). Raw generated diagnostics have an independent pass and no teacher fallback. No SEG prediction-position semantics.

## Commands

GPU choice is explicit; no GPU validation or training has been started. Run actual model forward/backward and longest evaluation scene memory checks before formal training.

```bash
cd /home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_4b_v1
GPU_IDS=<explicit,devices> bash scripts/train_answer.sh
GPU_IDS=<explicit,devices> bash scripts/train_joint.sh --answer-checkpoint outputs/open_answer/latest.pt
GPU_IDS=<explicit,devices> bash scripts/evaluate.sh <joint-checkpoint.pt> <independent-output-directory>
# Resume either training stage with --resume <same-stage-checkpoint.pt>.
# CPU-only checks:
OMP_NUM_THREADS=1 env/bin/python -m pytest -q
env/bin/python tools/check_data.py
env/bin/python tools/verify_initialize.py
```

`verify_initialize.py` requires complete official files and does no GPU forward. `check_official_layout.py` / `check_layout_shapes.py` are explicitly metadata preflight, not full weight verification.

## Evaluation

Teacher, generated with reference fallback, and raw_generated_no_fallback: selected/matched/oracle IoU and Acc@0.25/0.50; selected-mask precision/recall/Dice; query match rate; best-query Top1/3/5/10 and mean rank; score Top-k threshold recall. Additional tie-aware best-query metrics accept any maximum-IoU tied query. Format validity, OBJ phrase accuracy, ANC precision/recall/F1 and fallback reasons are reported. Raw metrics divide by evaluable generated samples; attempted count/coverage and unevaluable formats are explicit. Visual token/superpoint mean/max and peak evaluation memory are recorded; training peak memory is recorded per epoch.

## Environment and validation status

`env/` is a standalone venv namespace. PyTorch2.4.1+CUDA12.4 and binary dependencies are explicit read-only symlinks to the existing installation; Transformers4.57.3, Tokenizers0.22.1 and PEFT0.18.1 are installed locally. Original environments are not modified. SpConv-cu1202.3.6/cumm-cu1200.4.11 are reused; CUDA compatibility requires the pending device-specific GPU check.
Version lock and dependency origins: `docs/environment-lock.txt`, `docs/readonly-dependencies.json`. Implementation decisions and test evidence: `docs/implementation-ledger.md` and `docs/*.log`.
Official structural/tokenizer preflight and real dataset alignment passed. Full weight initialization is complete only when `docs/full_weight_initialization.json` exists with `full_hash_verified:true`. Large-file download/transfer is still in progress until that report is produced. No accuracy improvement is assumed.

The F-drive transfer helper is `tools/windows_model_transfer.py`: official files only are copied, partial ranges are retained locally, and verified transfer automatically triggers CPU-only full initialization and regression tests on224. It does not launch GPU validation or training. PCCM inline, Torch extensions, CUDA and Triton caches are routed under the new project.
