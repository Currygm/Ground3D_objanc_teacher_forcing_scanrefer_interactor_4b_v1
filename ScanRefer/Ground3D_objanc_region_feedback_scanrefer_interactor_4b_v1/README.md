# Ground3D Interactor: shared-Qwen object feedback, Stage A v3

The first pass supplies one visual descriptor per actually predicted OBJ/ANC,
rather than one token per decoder query. Stage A retains all 50 masks for scoring.
Both passes reuse the original Qwen; there is no copied student Qwen.

## Weight and evidence contract

1. Load the pinned original Interactor epoch015 checkpoint, freeze it, and cache
   first-pass train/val outputs in a deterministic per-sample coordinate frame.
2. Release that model. Initialize the second pass from the same original checkpoint.
   Train only its existing language LoRA and object projection; no new scoring head.
   Qwen base weights, task token rows and visual components remain frozen.
3. Training/evaluation read the original first-pass cache. Updated LoRA must never
   regenerate the supposedly fixed first-pass candidates.

For each generated entity, select its highest phrase-match-score mask, pool its
full superpoint features using mask probability times raw point count, and project
256 -> 2560 (LayerNorm + Linear). Exactly one visual token per valid predicted
entity is interleaved with its English role/phrase label. The original 32 scene
tokens, raw first answer and original question are retained. English instructions
explicitly treat the first prediction as a potentially incorrect hypothesis.
There are no geometry tokens, candidate-ID embeddings or all-query token inputs.
Repeated masks retain separate entity bindings. Missing regions are labeled in
English; invalid answers do not synthesize object evidence or use GT.

Stage A replays the actual predicted answer under the object evidence prefix and
uses the resulting question/answer token hidden states. Frozen original text_adapter,
text_connector and role embeddings map them into the original text matching space.
Each cached candidate key is the original final query after output_norm,
text_query_projection and normalization. Cosine logits / original temperature,
then mean sigmoid over OBJ phrase tokens, score all50 fixed candidates.
No new classifier is introduced. The original nine query/visual update layers are
not rerun: that would change the candidate representations and masks. This reuses
the original scoring formula and text projection parameters, not the complete
original decoder text-update computation. LoRA learns through this frozen interface.
The oracle and every mask are asserted unchanged at evaluation. Stage B and the
old visual_geometry config are disabled. Old checkpoints cannot load into v3.

## Environment

Run within this project on node224. Existing env/models are read-only shared
resources. No install is required. Always explicitly set GPU_IDS (e.g. 0 or 0,1).
Use the same device count and accumulation for matched training controls.
GPU runs below have not been executed as part of this code delivery.

## 1. Metadata preflight (CPU)

~~~bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= env/bin/python tools/check_feedback_contract.py
~~~

## 2. Generate immutable first-pass caches

Use full caches for formal training; the output root is pinned in configs.
Default per-split disk budget is 200 GiB. The tool reports a pilot size estimate
before writing its first new record; sizes vary, and actual usage is checked too.

~~~bash
GPU_IDS=0 bash scripts/cache_first_stage.sh --split train --max-cache-gib 200
GPU_IDS=0 bash scripts/cache_first_stage.sh --split val --max-cache-gib 200
~~~

If interrupted, rerun the same command with --resume. It validates source identity,
sample bindings and record checksums. Partial files must not be silently reused.
Caches store selected object descriptors, 32 scene tokens, masks, scores, mapping,
raw answers, fixed candidate scoring keys and separate target labels, not full [S,256] features for every sample.
Training uses that fixed preprocessing realization instead of epoch augmentations.
Per-sample shards and manifests are under cache/first_stage_epoch015_objects_v3/.

For limited first-pass acceptance, --split val --max-samples 16 creates a partial
cache. It is only usable with baseline --max-samples 16, cannot train Stage A,
and occupies the same split path; use a separate copied config with a distinct
cache_dir for this acceptance run. Never overwrite a completed formal cache.

## 3. Baseline on the same cached candidates

~~~bash
GPU_IDS=0 bash scripts/evaluate_feedback.sh --baseline --output-dir evaluations/a0_objects_v3
~~~

No language model is constructed by baseline cache evaluation.

## 4. Train Stage A

~~~bash
CONFIG=configs/feedback_a_visual.yaml GPU_IDS=0 bash scripts/train_feedback.sh
~~~

Default: 5 epochs, language LoRA LR 1e-5, new modules LR 1e-4, batch/rank 1,
accumulation 2, BF16. Only language LoRA is trainable, not the visual LoRA.
Checkpoints are under outputs/feedback_a_visual_v3/. Resume strictly:

~~~bash
CONFIG=configs/feedback_a_visual.yaml GPU_IDS=0 bash scripts/train_feedback.sh \
  --resume outputs/feedback_a_visual_v3/latest.pt
~~~

No automatic in-training validation is performed.

## 5. Evaluate each epoch

~~~bash
for ckpt in outputs/feedback_a_visual_v3/epoch_*.pt; do
  name=$(basename "$ckpt" .pt)
  CONFIG=configs/feedback_a_visual.yaml GPU_IDS=0 bash scripts/evaluate_feedback.sh \
    "$ckpt" --output-dir "evaluations/stage_a_visual_v3/$name"
done
~~~

Every evaluation directory must be new. summary.json contains first/main/strict
metrics; predictions.jsonl includes predicted entity phrases, roles, source query
IDs and evidence validity. First-pass format failures use the inherited zero-OBJ
baseline and strict zeroed metrics, with all sample denominators retained.
Final validation covers 9483 unique samples. Track selected_iou/acc50,
best_id_hit and tie-aware best_tie_hit, covered50 conditional selection,
Top-k oracle and regret. A best-hit gain alone is not proof of quality improvement.
Checkpoint feedback evaluation needs both original train/val manifests for provenance.

## Controls

Train/evaluate each with its own CONFIG and output directory:
- feedback_a_empty.yaml: first answer/entity phrase embeddings and object tokens zeroed, including assistant answer replay.
- feedback_a_text.yaml: entity text retained; object visual tokens zeroed.
- feedback_a_visual.yaml: text plus actual predicted-object visual evidence.
- feedback_a_selector.yaml: ordinary two-layer MLP, no second Qwen pass. Its inputs
  are pooled scene tokens, frozen question embedding mean, and valid object-feature
  mean. Same data/seed/update budget; parameters/FLOPs are not claimed equal.

## Tests and limitations

~~~bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= env/bin/python -m pytest -q
~~~

Tiny real Qwen/UniVLG9 tests exercise entity binding, LoRA/projection gradients, exact original score reproduction with cached first-pass updated text,
frozen parameters, prompt padding, immutable caches, oracle and checkpoint rules.
Full 4B/CUDA/SpConv, distributed execution, memory acceptance and empirical accuracy
require separate hardware runs. No formal training or cache-generation GPU job
was launched for this delivery.

For later uncached deployment, sequentially load original LoRA for the first pass,
then feedback LoRA for the second pass, in one model instance. The current entrypoints
use the offline-cache protocol; an adapter-switching deployment API is not implemented.

Final verification: 136 CPU tests passed; metadata/CLI/syntax checks passed.
Independent final review was attempted but could not complete due reviewer usage
limits. Author self-review found/fixed the empty-control replay text leak with a
RED/GREEN regression and reran the full suite. This is not independent approval.
See docs/verification.json for evidence and exact unvalidated hardware paths.
