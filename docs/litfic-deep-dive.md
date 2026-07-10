# LiTFiC — Deep Dive (Paper ↔ Code)

*Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues* — CVPR 2025 (arXiv:2501.09754v2). Jang, Raajesh, Momeni, Varol, Zisserman.

This document maps every part of the paper to where it lives in this repo, so you can
explain the method and point at the exact code that implements it.

---

## 1. Problem & core idea

**Task:** Sign Language Translation (SLT) — turn a continuous **British Sign Language**
signing video into an English sentence, open-vocabulary, on the large-scale **BOBSL**
dataset (~1,400h of interpreted TV with English subtitles).

**Central insight (the paper's contribution):** human interpreters use *context*, so
feed a pre-trained **LLM (Llama-3-8B)** not just the signing video but three extra
textual **contextual cues**, then LoRA-fine-tune it to generate the translation:

| Cue | What it is | Where extracted (paper) |
|-----|-----------|-------------------------|
| **Visual features (Vid)** | Video-Swin ISLR features, 768-d, sliding window stride 2 (~56/sentence) | encoder from [62] |
| **Pseudo-glosses (PG)** | Noisy sign-class predictions from the same ISLR model (~22/sentence) | ISLR classifier |
| **Previous sentence (Prev)** | Translation of the previous sentence; auto-regressive at inference | model's own prediction |
| **Background (BG)** | BLIP2 caption of the scene behind the signer, reduced to keywords | BLIP2 [44] |

Everything except Vid is text, so it drops straight into the LLM prompt. Vid is mapped
into the LLM token space by a small MLP.

**Result headline (Table 3, SENT-TEST):** Ours(Vid) already beats prior SOTA
(B-RT 37.8 vs Sign2GPT 35.2); full cues Vid+PrevPred+PG+BG → 40.3 B-RT.

---

## 2. Repo shape

Built on lightning-hydra-template. Entry points:
- `src/train.py` — training (Hydra config `configs/train.yaml`)
- `src/eval.py` — evaluation from a checkpoint (`configs/eval.yaml`)
- `src/llm_eval.py` — LLM-based (CLAIR/GPT-4o-mini) scoring of a caption file

Config composition (`configs/`): `data/bobsl.yaml`, `model/vgg_slt.yaml`,
`trainer/ddp.yaml`, `paths/default.yaml`, and the four `experiment/*` overlays.

---

## 3. Data pipeline

Flow: `SLTDataModule` → `Sentences` → (`Subtitles` + `LMDBLoader`) → `collate_fn_padd_t`.

### 3.1 `SLTDataModule` (`src/data/slt_datamodule.py`)
- Builds train/val/test `Sentences` datasets (`setup()`).
- **Val/test dataloaders shard by *episode*** using `val_start_indices.json` /
  `test_start_indices.json` (the `..._episode_ind_path`). Each GPU (rank) gets a
  contiguous episode range — this matters because **previous-sentence context is
  auto-regressive within an episode**, so batches must stay ordered and in-episode.
- batch_size=1 for val/test (ordered, auto-regressive); train uses configured batch.

### 3.2 `Sentences` (`src/data/components/sentence.py`)
`get_single_item()` assembles one training example:
- `features` ← `LMDBLoader.load_sequence()` for `[sub_start, sub_end]` frames (Video-Swin feats).
- `pls` ← pseudo-labels LMDB: per-window top classes, filtered by `pl_filter`,
  optionally **synonym-grouped** (`synonym_combine`) and repetition-averaged
  (`compress_and_average`) → a list of gloss words.
- `previous_context`, `question` (the instruction), `bg_description` ← from `Subtitles`.
- `spottings` ← optional oracle GT-derived sign annotations (only for the +Spot rows).
- `subtitle` ← the target English sentence (cleaned by `cleanup_sub`).

`collate_fn_padd_t()` pads features to a batch tensor + attention mask; all text
fields (subtitles, questions, pls, prev, bg, spottings) stay as Python lists — the
LLM tokenizes them per-sample inside the decoder.

### 3.3 Key config knobs (`configs/data/bobsl.yaml`)
- `subtitles_min/max_duration` 1–20s (paper's 689k pairs filter).
- `feats_lmdb_stride: 2` → the "~56 features/sentence" in the paper.
- `pl_synonym_grouping: True`, `pl_filter` — pseudo-gloss cleanup.
- `max_previous_sentences`, `aug_prev`, `train_cap_path/prob` — previous-sentence cue.

---

## 4. Prompt construction — the heart of the method

`LanguageDecoder._process()` (train) and `_process_predict()` (inference) in
`src/models/components/vgg_slt_modules/language_decoder.py` build the token sequence.
Order of assembly into `questions` string:

1. **Instruction** (base `question`): *"You are an AI assistant designed to interpret a
   video of a sign language signing sequence and translate it into English."*
2. **Previous context** (if `use_rec_prev`): `+ " The previous context is the following: {c}"`
3. **Spottings** (if `use_spottings`, oracle only): `+ " The following are some possible words present in the sentence: ..."`
4. **Pseudo-glosses** (if `use_pl_w_feats`): same phrasing, from `pls` (with `drop_words`).
5. **Background** (if `bg_desc`): `+ " Description of the background is: {keywords}"`
   (`get_unique_bg_words` dedups + optionally drops stopwords).
6. Then `+ " The following are the video tokens: "` and finally
   `".\nGenerated Sentence: "` before the target subtitle.

Per sample the embedding sequence is concatenated as:
```
[question tokens] + [projected video embeds] + [" Generated Sentence:" tokens] + [subtitle tokens]
```
**Label masking:** everything before the subtitle is set to `ignore_idx = -100`, so the
cross-entropy loss is computed **only over the subtitle tokens** (standard next-token
LM loss on the answer span). See the `cur_label` construction (lines ~148–156).

**Stochastic cue dropping (training only):** each cue is included only if
`random.random() <= mix_in_*_prob` (default 0.5) — this is the paper's "drop entire cue
with 50% prob" augmentation. At inference (`not self.training`) every available cue is
always included.

**Inference (`_process_predict` + `_predict`):** uses `rec_prev` (the model's own past
predictions, gathered by `CircularBuffer` in `slt_module.model_step`) as the previous
context, then `decoder.generate(max_new_tokens=50)`.

---

## 5. Model

`VggSLTNet` (`src/models/components/vgg_slt.py`) = `MMProjector` + `LanguageDecoder`.

- **`MMProjector`** (`mm_projector.py`): `mlp2x_gelu` = `Linear(768→4096) → GELU →
  Linear(4096→4096)`. (A `conv_*` variant with 1D temporal conv exists for How2Sign,
  per paper Appendix A.5.)
- **`LanguageDecoder`** (`language_decoder.py`):
  - Loads Llama-3-8B (`AutoModelForCausalLM`, `paths.llm_root`).
  - **LoRA** via `peft`: `r=4, alpha=16, dropout=0.05, target=['q_proj','v_proj']`
    (`configs/model/vgg_slt.yaml`). Only q/v projections train; token embeddings frozen.
  - precision `bf16-mixed` on GPU; `attn_implementation=flash_attention_2`.
  - Trainable params = MLP projector + LoRA adapters only.

---

## 6. Training loop

`SLTLitModule` (`src/models/slt_module.py`):
- **Loss:** `CELoss` over `outputs` vs masked `labels` (only subtitle span counts).
- **Optimizer/scheduler** (`configs/model/vgg_slt.yaml`): Adam lr=1e-4, wd=0;
  `LinearWarmupCosineAnnealingLR`, warmup 5 epochs, total = `trainer.max_epochs` (10 for
  BOBSL). `gradient_clip_val: 1.0`, `gradient_checkpointing_enable: True`.
- **Augmentations (paper Sec 3.3 / Table 2):**
  - *Drop words* — `drop_words(list, pct)` removes up to 50% of tokens in a cue.
  - *Drop cue* — the `mix_in_*_prob` gate above.
  - *PrevPred/PrevGT* — sample GT vs precomputed prediction for the previous sentence
    (`train_cap_path` / `aug_prev`).

### Mapping experiments → paper Table 1
| Config (`configs/experiment/`) | Cues | LLM flags set |
|---|---|---|
| `vid.yaml` | Vid | (all off) |
| `vid+pg.yaml` | +PG | `use_pl_w_feats` |
| `vid+pg+prev.yaml` | +Prev | `+ use_rec_prev` |
| `vid+pg+prev+bg.yaml` | +BG (full) | `+ use_bg_words, drop_bg_sw, bg_desc` |

---

## 7. Evaluation

- **`eval.py`** loads a ckpt and runs `trainer.test()`. Predictions accumulate in
  `slt_module` lists; `_eval()` / `get_cap_metrics()` compute metrics after all-gather
  across DDP ranks.
- **Metrics:** BLEU-4, ROUGE-L, CIDEr (pycocoevalcap), **BLEURT-20**
  (`bleurt_pytorch`), and **IoU** of token sets after lemmatization + synonym matching
  (`incorporate_syns`, `calculate_overlap_metrics`). Per-epoch dumps `vis/<epoch>/cap.json`.
- **`llm_eval.py`** — CLAIR-style GPT-4o-mini scoring 0–5 with 12 in-context examples
  (paper Sec 4.1 "LLM Evaluation"); needs an OpenAI API key, run on a caption file.
- **Auto-regressive context at test:** `CircularBuffer(context_len)` in `model_step`
  feeds the model's own previous predictions as `rec_prev`; buffer resets on episode
  change (`batch["video_names"][0]`).

---

## 8. Results reference (for later comparison)

**Table 1 — cues on SENT-VAL (B-RT / IoU / LLM):**
Vid 41.0/16.6/1.29 → +PG 41.8/17.5/1.40 → +PrevPred 42.5/18.1/1.45 → +BG 43.5/18.8/1.56.

**Table 3 — SENT-TEST (B4/B-RT/R-L/CIDEr/IoU/LLM), selected:**
- GFSLT 0.6/27.7/7.4/4.3/5.2/0.05 ; Sign2GPT(w/PGP) 0.9/35.2/11.4/16.1/8.7/0.49
- Ours(Vid) 2.6/37.8/15.6/37.5/13.6/0.95
- Ours(Vid+PrevPred+PG+BG) 3.3/40.3/16.9/41.9/14.8/1.20
- Oracle Ours(Vid+PrevGT+Spot+BG) 7.3/47.1/25.1/88.9/26.5/1.85

How2Sign Ours(Vid): 11.8/44.1/31.1/93.3/26.1/1.39.

---

## 9. Reproduction gotchas

- **VRAM:** Llama-3-8B + LoRA needs an 80GB-class GPU (paper: 4× H100, bs 2/GPU). A
  4GB local GPU cannot run the real model → local work is smoke-test only (Part B).
- **flash-attn:** `decoder_config.attn_implementation: flash_attention_2` is hardcoded;
  override to `eager` and use `precision=32-true` off supported hardware.
- **Gated / large data:** Llama-3-8B (HF access request), BOBSL feature+PL LMDBs
  (100s of GB), CSLR annotations (1.9GB), BLEURT-20, plus the extra JSONs
  (`val/test_start_indices.json`, `prev_gt_captions.json`, cleaned BLIP2 captions).
- **Paths:** every dataset path is wired through `configs/paths/default.yaml` and needs
  `PROJECT_ROOT` env var; see `configs/paths/README.md` for the download map.
- **Pretrained ckpt:** `bobsl_all.ckpt` (linked in README) lets you evaluate without
  training.

---

## 10. Part B — local smoke-test (verified)

`scripts/smoke_test.py` exercises the real `VggSLTNet` (MMProjector + LanguageDecoder)
with a tiny Llama-arch model (SmolLM-135M) instead of Llama-3-8B, on CPU. It bypasses
the lightning/hydra/LMDB stack (stubs out `src.utils.__init__`) and feeds a synthetic
batch shaped like `collate_fn_padd_t`, with the full Vid+PG+Prev+BG cue set.

Confirmed on this machine (torch 2.11 CPU, transformers 5.5.4):
- Prompt assembles and tokenizes; **loss is over 22 subtitle tokens only**, the rest
  masked to `-100` — proving the label-masking design in §4.
- **Trainable params = 0.74%** (1.0M) = MLP projector + LoRA adapters; backward flows.
- `decoder.generate()` produces a sentence auto-regressively.

Run: `PROJECT_ROOT=$(pwd) python scripts/smoke_test.py`
(override the model with `SMOKE_LLM=...`). This is a **liveness proof of the pipeline,
not paper numbers** — the projector/LoRA are untrained and the video tokens are random.
