# Kaggle Runbook — LiTFiC How2Sign Vid+Prev demo (2×T4)

Operational guide for Tasks 5–7 of `2026-07-13-litfic-how2sign-t4-demo.md`.
These steps run on **Kaggle** (need GPU + internet); they cannot run in the
local CPU dev env. Code Tasks 1–4 are already implemented, tested, and committed
on branch `how2sign-t4-demo`.

> **T4 reminder:** precision `16-mixed` (no bf16 tensor cores), attention
> `sdpa` (FlashAttention-2 unsupported on Turing). Both are already baked into
> `configs/experiment/how2sign-vid*.yaml`. **Do not install `flash-attn`.**

---

## Task 5 — Data acquisition + schema verification

### 5.1 Download public I3D features + tsv
Fetch the UPC How2Sign SLT release referenced by
`github.com/imatge-upc/slt_how2sign_wicv2023` (per-sentence I3D `.npy` +
`.tsv` per split, on the linked dataverse). Record URLs + total size here:

- Train tsv: `TODO_URL`  ·  Val tsv: `TODO_URL`  ·  Test tsv: `TODO_URL`
- Features archive(s): `TODO_URL` (size: `TODO` GB)

### 5.2 Inspect real schema (one cell)
```python
import csv, glob, os, numpy as np
tsv = sorted(glob.glob("**/*train*.tsv", recursive=True))[0]
with open(tsv, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t"); print("COLUMNS:", r.fieldnames)
    row = next(r); print("ROW0:", row)
npy = sorted(glob.glob("**/*.npy", recursive=True))[0]
a = np.load(npy); print("NPY", os.path.basename(npy), a.shape, a.dtype)
```
**Record:** column names, one clip id, and the feature `.shape`.
- Confirm feature **dim == 1024** (else update `feats_dim` + `mm_hidden_size` everywhere).
- Note avg/max `T`. If `T` is large (> ~150), set
  `model.net.mm_projector_config.projector_type: conv_K5_P2_KP2_L2` in both
  experiment configs (conv keeps dim=1024, compresses length).

### 5.3 Reconcile schema with the dataset code
The dataset defaults expect columns `id`, `translation`, `npy_path` and clip ids
like `<video>_<idx>-...`. If the real tsv differs:
- Pass `id_col` / `text_col` / `npy_col` via `data.dataset_config` (Hydra
  override), e.g. `data.dataset_config.text_col=tgt_text`.
- If clip ids don't encode `<video>_<idx>`, adjust `_ID_RE` in
  `src/data/components/how2sign.py`.
- Rename/symlink the split tsvs to `how2sign_train.tsv` / `how2sign_test.tsv`
  (the loader opens `how2sign_{setname}.tsv`; val uses the **test** tsv by design).

### 5.4 Package as a private Kaggle Dataset
Upload `feats/` + `tsv/` as dataset `how2sign-i3d` mounted at
`/kaggle/input/how2sign-i3d/` → matches `configs/paths/how2sign.yaml`
(`h2s_tsv_dir`, `h2s_feats_dir`). Adjust those two paths if the mount differs.

---

## Kaggle environment setup (run once per session)

```bash
cd /kaggle/working/LiTFiC          # your checkout of branch how2sign-t4-demo
pip install -q "lightning==2.3.0" torchmetrics "hydra-core==1.3.2" \
  "hydra-colorlog==1.2.0" omegaconf rich rootutils einops lmdb \
  "transformers==4.45.2" "peft==0.12.0" sentencepiece \
  "lightning-utilities==0.11.2" nltk pycocoevalcap lightning-bolts
export PROJECT_ROOT=$(pwd)
# Episode-index files → iterate the WHOLE val/test set (verified trick):
python - <<'PY'
import json
for p in ["val_start_indices.json", "test_start_indices.json"]:
    json.dump({"idx": [0]}, open(p, "w"))
print("wrote idx files")
PY
```
Notes:
- **Skip `flash-attn`** (we use sdpa). Skip `bleurt`, `opencv`, `wandb` unless needed.
- Metrics (`pycocoevalcap`) may need Java: `apt-get -y install default-jre` if BLEU/CIDEr errors.
- If `lightning-bolts` (`pl_bolts`) fails to import at runtime → apply the
  scheduler fallback in Task 6 Step 5.
- Use **one** T4 (`devices:[0]`) to avoid Kaggle notebook DDP spawn issues.

---

## Task 6 — Milestone M1: Vid-only smoke run

### 6.1 Overfit sanity (few batches, no OOM, finite loss)
```bash
python src/train.py experiment=how2sign-vid \
  trainer.max_epochs=1 +trainer.limit_train_batches=20 +trainer.limit_val_batches=5 \
  logger=csv
```
Expect: runs on `cuda`, `train/loss` finite (not NaN), no OOM. If OOM → set
`projector_type=conv_K5_P2_KP2_L2` and/or lower `limit_train_batches`.

### 6.2 Short real run with checkpointing
```bash
python src/train.py experiment=how2sign-vid trainer.max_epochs=3 \
  callbacks.model_checkpoint.every_n_train_steps=500 logger=csv
```
Expect: loss decreases; a `.ckpt` under the run's `output_dir`. Record `CKPT_PATH`.

### 6.3 Evaluate + inspect predictions
```bash
python src/eval.py experiment=how2sign-vid ckpt_path=CKPT_PATH \
  +trainer.limit_test_batches=50 logger=csv
```
Expect: BLEU-4 / ROUGE-L printed on the subset + a few grammatical English
predictions. **Save as M1 baseline below.**

### 6.4 (Only if `pl_bolts` broke) scheduler fallback
In `configs/model/vgg_slt.yaml` replace the `scheduler:` block with a stock
torch cosine schedule, re-run 6.1, then commit.

### M1 results (fill in)
- Command / CKPT: `TODO`
- BLEU-4: `TODO`  ROUGE-L: `TODO`
- Sample predictions: `TODO`

---

## Task 7 — Milestone M2: Vid+Prev ablation

### 7.1 Confirm prev reaches the prompt (guard against silent Vid-only)
Temporarily add, in `LanguageDecoder._process_predict` right after the
`if self.use_rec_prev and not self.use_gt_prev:` block builds `questions`,
`print(questions[0])`. Then:
```bash
python src/eval.py experiment=how2sign-vid+prev ckpt_path=M1_CKPT \
  +trainer.limit_test_batches=2 logger=csv
```
Expect the printed prompt to contain
`The previous context is the following: <a real sentence>` for non-first
sentences. Remove the print afterward.

### 7.2 Train Vid+Prev
```bash
python src/train.py experiment=how2sign-vid+prev trainer.max_epochs=3 \
  callbacks.model_checkpoint.every_n_train_steps=500 logger=csv
```
Record `M2_CKPT`.

### 7.3 Evaluate on the SAME subset as M1
```bash
python src/eval.py experiment=how2sign-vid+prev ckpt_path=M2_CKPT \
  +trainer.limit_test_batches=50 logger=csv
```

### M2 results / ablation (fill in)
| Model | BLEU-4 | ROUGE-L |
| --- | --- | --- |
| Vid (M1) | `TODO` | `TODO` |
| Vid+Prev (M2) | `TODO` | `TODO` |

- 2–3 qualitative examples where prev changed the output: `TODO`
- Demo is **done** when Vid+Prev is reported next to Vid (direction of change
  noted honestly, even if small on a 3-epoch run).

---

## Gotchas / verify-on-run checklist
- [ ] Feature dim really 1024; sequence length `T` (conv projector if long).
- [ ] tsv column names + clip-id format match the loader (Task 5.3).
- [ ] Qwen2.5-3B loads via `AutoModelForCausalLM`; `embed_tokens` resolves
      (`decoder.model.embed_tokens`) and pad/eos fallback works.
- [ ] fp16: watch for NaN; `gradient_clip_val: 1.0` is set.
- [ ] Checkpoint/resume works across the 12 h Kaggle session limit.
- [ ] Java present for `pycocoevalcap` metrics (else `apt-get install default-jre`).
