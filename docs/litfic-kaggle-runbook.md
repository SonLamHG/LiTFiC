# LiTFiC on Kaggle 2×T4 — Subset Runbook

Goal: **understand the flow** of the paper by running the *real* repo (hydra + Lightning
DDP + datamodule + LMDB + LoRA + eval) on a **small real subset of BOBSL**, with a
**paper-endorsed small LLM (Llama-3.2-3B)** — not to match paper numbers.

Locked decisions:
- **LLM = Llama-3.2-3B** (Appendix B.1 / Table A.4: comparable to Llama-3-8B; fits one T4).
- **Data = real BOBSL subset** (a few whole episodes carved from the 262 GB LMDBs).

Why not the full thing on Kaggle/Colab: the feature LMDB is 262 GB (can't fit Kaggle's
20 GB disk; streaming an mmap LMDB over Drive is unusably slow), and 2×T4 under Kaggle's
12 h/session, ~30 h/week quota cannot finish full training (paper used 4×H100 for days).
Full reproduction stays on rented A100/H100.

---

## Sizing the subset — bound by compute, not disk

Derived from the real figure **262 GB features / 1467 h** of BOBSL video:

| Unit | Feature | Pseudo-label | Total |
|---|---|---|---|
| per minute | ~3.0 MB | ~0.3 MB | **~3.3 MB** |
| per hour | ~180 MB | ~18 MB | **~200 MB** |
| **per episode** (~45 min avg, ~490 sentences) | ~135 MB | ~14 MB | **~150 MB** |

Smallest granularity is **one whole episode** (never isolated sentences — the
previous-sentence context and per-episode DDP sharding need intact episodes).

**Disk is not the binding constraint.** `/kaggle/working` is 20 GB and attached
`/kaggle/input` datasets can be far larger — you could fit ~130 episodes. The real
limit is **GPU time**: Kaggle gives ~12 h/session, ~30 h/week on 2×T4, and Llama-3.2-3B
with long prompts is slow (train ~1–3 sentences/s; eval, which generates ~50 tokens
auto-regressively, is much slower). So size the subset to what a session can actually
process, not to fill the disk.

**Recommended (to understand the flow):**

| Purpose | Episodes | ~Sentences | LMDB size |
|---|---|---|---|
| **TRAIN** (see the loop run) | 3 | ~1,500 | **~450 MB** |
| **EVAL** (see the eval flow + outputs) | 1 | ~490 | **~150 MB** |
| Both (3 train + 1 separate eval) | 4 | ~2,000 | **~600 MB** |

Plus shared metadata kept whole (subtitles, vocab, synonyms, info, BLIP captions):
~0.5–1.5 GB (optionally trim subtitles/captions to the chosen episodes → <100 MB).
**Total Kaggle upload ≈ 1–2 GB** — hundreds of times below the 262 GB full set.

Bigger subsets don't help understanding and only burn session quota; quality can't
approach paper level regardless (that needs 689k pairs + H100-days → the cloud runbook).
After running `subset_bobsl_lmdb.py`, `du -sh feats_lmdb/ pl_lmdb/` gives the exact size.

---

## Stage 0 — One-time data prep (on a big-disk machine, NOT Kaggle)

You have Google Drive 5 TB as the master store. Do the heavy step on a VM / machine
with local disk, then upload only the small subset to Kaggle.

1. **Get access + download** (see sizes):
   - BOBSL license + download: https://www.robots.ox.ac.uk/~vgg/data/bobsl/#download
   - CSLR2 data page (feature LMDB 262 GB, `bobsl.zip` 1.9 GB → 15 GB): https://gulvarol.github.io/cslr2/data.html
   - Needed: feature LMDB, pseudo-label LMDB, `bobsl.zip` (annotations/metadata),
     BLIP2 background captions, Llama weights (see Stage 2), extra JSONs from the
     LiTFiC share (README): `val/test_start_indices.json`, `prev_gt_captions.json`.
   Pull them from Drive to the VM's **local disk** (LMDB must be on a real FS).

2. **Carve the subset** (this repo's script, verified via `--selftest`):
   ```bash
   python scripts/subset_bobsl_lmdb.py \
     --feats-src /data/bobsl/lmdb-feats_vswin_t-bs256_float16 \
     --pl-src    /data/bobsl/lmdb-pl_vswin_t-bs256_float16 \
     --out-dir   /data/bobsl_subset \
     --train ep1 ep2  --val ep3  --test ep3
   ```
   Pick **whole episodes** (not isolated sentences) so the previous-sentence
   auto-regressive context and per-episode DDP sharding stay valid. Output:
   `feats_lmdb/`, `pl_lmdb/` (a few hundred MB), and `subset2episode.json`.

3. **Keep the small metadata whole** — subtitles pkl, `8697_vocab.pkl`, synonyms pkl,
   info pkl, BLIP2 captions: these are small and are filtered by episode at load time,
   so no subsetting needed. Just include them in the upload.

4. **Upload** `bobsl_subset/` + the metadata as a **Kaggle Dataset** (private,
   read-only mount at `/kaggle/input/...`). Total should be a few GB.

---

## Stage 1 — Kaggle notebook setup (2×T4)

Settings → Accelerator = **GPU T4 ×2**; Internet = **On**. Attach your dataset.

```bash
pip install -q lightning==2.3.0 hydra-core==1.3.2 hydra-colorlog==1.2.0 \
  transformers==4.45.2 peft==0.12.0 einops lmdb sentencepiece \
  pycocoevalcap bleurt_pytorch nltk rootutils rich torchmetrics \
  pytorch-lightning-bolts==0.3.2.post1 lightning-bolts==0.7.0
# NOTE: do NOT install flash-attn (Turing T4 unsupported) -- we override to eager.
python -c "import nltk; [nltk.download(x) for x in ['stopwords','wordnet','averaged_perceptron_tagger','punkt']]"
export PROJECT_ROOT=$(pwd)
```

Clone the repo (or add it as a Kaggle dataset/utility script).

---

## Stage 2 — Get Llama-3.2-3B

Gated model → accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B,
create an HF read token, add it in Kaggle **Add-ons → Secrets** as `HF_TOKEN`.

```python
from huggingface_hub import snapshot_download
import os
snapshot_download("meta-llama/Llama-3.2-3B", token=os.environ["HF_TOKEN"],
                  local_dir="/kaggle/working/Llama-3.2-3B")
```
(If you'd rather skip the token dance, `Qwen/Qwen2.5-3B` is ungated and drop-in — but
it's off-family from the paper.)

---

## Stage 3 — Point config at the subset (config-only, no code changes)

Edit `configs/paths/default.yaml` (or override on the CLI) so every path targets the
subset + metadata under `/kaggle/input/...`, and set `llm_root` to the downloaded model.
Regenerate the subset's start-indices once, on Kaggle, from the filtered subtitles
(the val/test dataloaders shard by these episode start offsets):

```python
# after paths point at the subset -- compute start indices for the subset episodes
from src.data.components.subtitles import Subtitles   # uses your subset2episode.json
# build a Subtitles(setname="val") / "public_test", then record the dataset index
# at each episode boundary into {"idx": [...]} and dump to val/test_start_indices.json
```
(Full snippet in the notebook template; it just walks `subtitles["episode_name"]` and
marks where the episode changes.)

---

## Stage 4 — EVAL the released checkpoint (see REAL, good translations)

Download `bobsl_all.ckpt` (README link) and run inference on the subset. The model was
trained on full BOBSL, so **outputs are paper-quality**; only the metric numbers are
noisy because the subset is tiny.

```bash
python src/eval.py \
  trainer=gpu trainer.devices=[0,1] trainer.precision=16-mixed \
  experiment=vid+pg+prev+bg \
  model.net.llm_config.decoder_config.attn_implementation=eager \
  paths.llm_root=/kaggle/working/Llama-3.2-3B \
  ckpt_path=/kaggle/working/bobsl_all.ckpt
```
Look at `logs/.../vis/<epoch>/cap.json` for GT vs prediction, and the logged BLEU/
ROUGE/CIDEr/BLEURT/IoU. **Caveat:** `bobsl_all.ckpt` was trained with Llama-3-8B; to
load it faithfully use 8B (won't fit T4). With 3B, treat Stage 4 as a *flow* check of
the eval loop; for real good outputs, either use 8B on a bigger GPU or rely on Stage 5.

---

## Stage 5 — TRAIN on the subset (see the training loop live)

```bash
python src/train.py \
  trainer=ddp trainer.devices=[0,1] trainer.precision=16-mixed \
  trainer.max_epochs=3 \
  experiment=vid+pg+prev+bg \
  data.batch_size=1 data.num_workers=2 \
  model.net.llm_config.decoder_config.attn_implementation=eager \
  paths.llm_root=/kaggle/working/Llama-3.2-3B
```
Expect: loss decreasing, LoRA (~0.7% params) + MLP training, per-epoch eval producing
`cap.json`. **Translation quality will be poor** — a few hundred sentences can't teach
open-vocabulary SLT (paper used 689 k). That's expected: Stage 5 confirms the *flow*
(hydra compose → DDP across 2×T4 → datamodule → LMDB → prompt assembly → LoRA → eval),
not the numbers.

If VRAM is tight: drop to Llama-3.2-1B, or the frozen-LLM variant (paper Table 2:
`model.net.llm_config.lora=False model.net.llm_config.freeze_decoder=True`, trains only
the MLP projector — the lightest setting the repo supports).

---

## What this proves vs. what it doesn't

| Confirmed on Kaggle 2×T4 | NOT attempted here |
|---|---|
| The real end-to-end pipeline runs on real (subset) BOBSL | Matching paper metrics |
| DDP across 2×T4, fp16 + eager attention | Full 689 k-pair training |
| Datamodule + LMDB loading + per-episode sharding | Llama-3-8B on T4 |
| Prompt assembly, LoRA training, eval + 5 metrics | Full SENT-TEST (20,870) eval |

Full reproduction (paper numbers) remains the rented A100/H100 runbook (Part C).

## Known adjustments (config, not new code)
- `precision=16-mixed` (T4 has no bf16) and `attn_implementation=eager` (no flash-attn on Turing).
- `mm_projector` output dim auto-follows the LLM via config; with a non-8B model the
  4096 default in `configs/model/vgg_slt.yaml` must match the chosen LLM's hidden size
  (Llama-3.2-3B = 3072). Override `model.net.mm_projector_config.hidden_size=3072`.
- Batch size 1–2 per T4; use `data.num_workers` modestly (Kaggle CPU is limited).
