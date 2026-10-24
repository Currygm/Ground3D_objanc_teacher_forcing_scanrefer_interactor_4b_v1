# Ground3D 4B: visual-only Interactor and zero-OBJ prediction

Independent experiment derived from Ground3D_objanc_teacher_forcing_scanrefer_4b_v1. Original code, official models, annotation files, geometry cache and environment are read-only. No old ScanRefer task checkpoint is loaded. No GPU validation or training has been launched.

## Architecture

Official SpConv U-Net -> raw-point expansion/superpoint pooling -> official x_mask [S,256].
All S superpoints enter a query-only six-layer InstructBlip Q-Former:32queries,768hidden,12heads,3072FFN,cross-attention every2layers,256memory width,dropout0.1.
Visual MLP:256->256 ReLU->256 ReLU; language projection:768->2560. Queries initialize Normal(0,0.02).
Original point2token is never constructed; its five official keys are explicitly excluded.
input_ids=None; no BERT tokenizer, text/answer/GT mask input. Unused word/position embeddings and text FFNs are frozen.
Pinned BERT embedding assets initialize the same shared embedding LayerNorm as preceding experiments; all asset files are hashed.
32prefixes replace the point marker, with continuous native1D positions repeated across Qwen's three position axes; no invented point centers.
Full pre-LLM F[S,256] -> Linear(256,512) -> UniVLG9; question+answer own-position causal hidden -> Linear(2560,2048) -> UniVLG9.
Decoder stays50queries/9layers/256hidden/8heads. Both Joint branches share full features without detach. No FPS.

## Training

Official Ground3D pure3D/Qwen3-VL-4B initialization includes official language LoRA(r32,alpha64); task vocabulary migration/row overlays follow source.
Answer3epochs: visual frontend frozen/eval; language LoRA and four task rows1e-5; new Interactor1e-4.
Joint20epochs: migrate completed Answer epoch3 language, rows and complete Interactor; visual frontend1e-5; language/rows1e-5; Interactor1e-4; fresh adapters/Mask Decoder1e-4.
Original final/auxiliary losses and all-valid OBJ+ANC ranking retained, plus0.1AnswerCE. Answer uses unscaledCE.
BF16 language/Interactor, FP32 sparse frontend, LLM gradient checkpointing, batch1/device, accumulation2, original weight decay/clipping/cosine minimum.
Training/Teacher use one causal pass. Generated uses one visual encode/Interactor, KV-cache generation, one actual-answer causal replay; no extra GTCE.
Save per-rank RNG, optimizer/scheduler, complete Interactor and visual states. Dedicated DataLoader RNG isolates iterator seeds from global model RNG.
Restore rejects old objectives/signatures, changed schedules/config, incomplete Interactor state or changed assets. Joint requires completed Answer epoch3.

## Prediction and metrics

Generated never replaces its answer with GT, including valid wrong OBJ phrases.
Invalid format, missing/multipleOBJ or illegal termination: replay actual text, append2560D zero to answer hidden, mark final token OBJ, all failed-answer tokens OTHER; question stays QUESTION.
The sole entity span points at the final zero; empty answer supported. No synthetic vocabulary token, extra LLM input or additional forward.
Biased text adapter and role embedding remain active, so downstream features need not stay zero.
This branch is eval-only. Infrastructure/CUDA/nonfinite/data errors remain errors. Text over384tokens raises instead of truncating OBJ.

Main Generated uses all9483 samples:
- generated_with_zero_obj: main result, noGT, explicit per-metric entity denominators.
- raw_generated_no_fallback: strict comparison; failedOBJ quality,Query hits andMRR are zero with full attempted denominator. Undefined failed rank sentinel51.
- normal_generated_subset / zero_obj_subset: separate quality,Query metrics and counts.
- original raw_coverage, zero_obj_rate, original format/answer quality, failure reasons, teacher_fallbacks=0.
All/OBJ/ANC preserve selected/matched/oracleIoU,Acc@.25/.50,precision/recall/Dice,Query match,bestQueryTop1/3/5/10,rank and scoreTopK threshold recall.
ANC metrics retain phrase-aligned entities with explicit denominators. GT never gates OBJ inference. Teacher remains separate.
Call counters distinguish causal replay, generation prefill/incremental, visual encode and Interactor; runtime reports allocated/reserved peaks.

## Commands

env/bin/python reuses source venv read-only; no dependency installation. Select devices after GPU forward/backward, longest-scene memory and short DDP acceptance.

~~~bash
cd /home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1
GPU_IDS=0,1,2,3 bash scripts/train_all.sh
GPU_IDS=0,1,2,3 bash scripts/train_joint.sh --answer-checkpoint outputs/open_answer/epoch_003_step_00013716.pt
GPU_IDS=0,1,2,3 bash scripts/train_joint.sh --resume outputs/scanrefer_teacher_forcing_univlg9/latest.pt
GPU_IDS=0,1 bash scripts/evaluate.sh outputs/scanrefer_teacher_forcing_univlg9/latest.pt evaluations/joint_latest_2gpu
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES= env/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= env/bin/python tools/verify_initialize.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= env/bin/python tools/check_data.py
~~~

Four-GPU epoch3 step13716 assumes36569samples,batch1,accumulation2. Use actual completed epoch3 filename if world size changes.
CPU/source-integrity evidence is in docs/. CPU tests do not establish GPU compatibility, memory fit, DDP correctness or accuracy gains.
