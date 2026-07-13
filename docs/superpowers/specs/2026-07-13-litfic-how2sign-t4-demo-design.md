# LiTFiC → How2Sign demo on Kaggle 2×T4 — Design

**Date:** 2026-07-13
**Status:** Approved design (pending spec review)
**Scope:** Minimal, runnable demo — NOT a reproduction of Table 4 numbers.

## 1. Goal & non-goals

**Goal.** Run the LiTFiC training + inference pipeline on the How2Sign
(ASL → English) dataset on Kaggle (2×T4, 16 GB/GPU), producing real English
translations and BLEU-4/ROUGE on a test subset, with minimal engineering.
The demo must honor the paper's core thesis — *contextual cues help* — by
including the **previous-sentence** cue, not just video.

**Non-goals.**
- Matching the paper's Table 4 numbers (Vid+PG+PrevPred = 12.7 BLEU-4).
- Pseudo-gloss (PG) cue — requires a How2Sign ISLR classifier (Video-Swin,
  1,887 signs) that is **not released**. Deferred.
- Background (BG) cue — the paper does **not** use it on How2Sign. Excluded.
- Fine-tuning any video backbone or decoding the 80 h of raw video.
- LLM-based evaluation metric (needs an OpenAI API key). Excluded.

## 2. Key decisions (locked)

| Item | Decision |
| --- | --- |
| Cues | **Vid + Prev** (previous GT sentence). No PG, no BG. |
| Data | Public **I3D features** (`.npy` per sentence) + `.tsv` metadata from the UPC How2Sign SLT release (`imatge-upc/slt_how2sign_wicv2023`, hosted on dataverse). |
| LLM | **Qwen2.5-3B** (ungated) → LLM embedding dim **2048**. |
| Feature dim | `mm_hidden_size = 1024` (I3D — **verify on download**). |
| Projector | `mlp2x_gelu`; switch to `conv_K5_P2_KP2_L2` if sequences are long. |
| Precision | `16-mixed` (fp16 — T4 has no bf16 tensor cores). |
| Attention | `sdpa` (FlashAttention-2 requires Ampere+, unsupported on T4). |
| Grad ckpt | Keep enabled. |
| Trainer | Single T4 (`devices:[0]`), `accumulate_grad_batches` to raise effective batch; `ModelCheckpoint` per-N-steps for resume across the 12 h session limit. |
| Prev source | **GT previous sentence** (oracle) at train and test — documented as an optimistic simplification vs the paper's `PrevPred`. |

## 3. Why this honors the paper

Vid-only is exactly the *baseline row* the paper improves upon (11.8 BLEU-4).
The previous-sentence cue is the **cheapest** of the paper's How2Sign cues:
it is pure text recoverable from the `.tsv` (sentences within a video are an
ordered instructional monologue), it needs **no** extra model, and the prompt
+ code paths already exist (`use_rec_prev`, prompt *"The previous context is
the following:"*). The demo therefore reproduces the paper's central claim via
an ablation: **Vid vs Vid+Prev**.

## 4. Architecture & data flow

```
.tsv (sentence_id, video_id, translation, npy_path, duration)
   │  group by video_id, sort by sentence index
   │  → previous_context = prior sentence's translation ('' for first)
   ▼
How2SignSentences (NEW Dataset)  ── __getitem__ returns the SAME dict shape
   │        that collate_fn_padd_t already expects
   ▼
collate_fn_padd_t (REUSED, + rec_prev key added)
   ▼
VggSLTNet.forward (UNCHANGED)
   ├─ MMProjector: [T,1024] → [T',2048]
   └─ LanguageDecoder (Qwen2.5-3B + LoRA q_proj/v_proj)
        prompt = Initial + " The previous context is the following: {prev}"
                 + video tokens + "\nGenerated Sentence:" + target
```

The BOBSL-specific machinery (`Subtitles`, `LMDBLoader`, `subset2episode`,
`info_pkl`, temporal frame alignment, PG/synonyms/spottings) is **bypassed
entirely** — the new dataset reads `.npy` + `.tsv` directly.

## 5. Components to build / change

### 5.1 `How2SignSentences` dataset (new) — `src/data/components/how2sign.py`
- One clear purpose: turn `.tsv` + `.npy` into the item dict contract used by
  `collate_fn_padd_t`.
- Reads `.tsv`; builds an ordered per-video index → `previous_context`.
- `__getitem__` returns:
  `subtitle`=translation (via `cleanup_sub`), `features`=`np.load(npy)`→tensor
  `[T,1024]`, `question`=Initial task prompt (Table A.1), `previous_context`=GT
  prev, `pls`=None, `bg_description`=None, `spottings`=[], `sub_start/end`=0/dur,
  `id`=sentence_id.
- Optional target-word-drop augmentation (`remove_words`, max 20 %) on train,
  matching the paper's How2Sign overfitting mitigation.

### 5.2 Datamodule wiring — `src/data/slt_datamodule.py`
- Add a `dataset == "how2sign"` construction path that instantiates
  `How2SignSentences` instead of `Sentences` (the file already has two
  `how2sign` branches; extend rather than fork the class).

### 5.3 Previous-sentence at inference — collate/plumbing
- `collate_fn_padd_t` currently omits `rec_prev`, so at predict time
  `_process_predict` sets prev to `''` (no context at test). Add a `rec_prev`
  key carrying the GT previous sentence (as `[[prev]]` per item), and set
  `use_rec_prev: True`, `use_gt_prev: False`, `mix_in_prev_prob: 1.0`.
- Training already consumes `previous_contexts` when `use_rec_prev` is True.
- (Alternative considered: patch `_process_predict` to honor `use_gt_prev`.
  Rejected — editing core model code is more invasive than adding a collate key.)

### 5.4 New configs
- `configs/data/how2sign.yaml` — points at mounted Kaggle dataset paths;
  `dataset: how2sign`, `feats_dim: 1024`, cue flags off except prev.
- `configs/experiment/how2sign-vid.yaml` (M1) and
  `configs/experiment/how2sign-vid+prev.yaml` (M2).
- `configs/paths/how2sign.yaml` — `llm_root: Qwen/Qwen2.5-3B`, tsv/npy roots.
- Model overrides: `mm_hidden_size: 1024`, `hidden_size: 2048`,
  `attn_implementation: sdpa`.
- Trainer overrides: `precision: 16-mixed`, `devices: [0]`,
  `accumulate_grad_batches: 16`, `max_epochs: ~10`, checkpoint per-step.

### 5.5 Dependency guards for Kaggle
- `pl_bolts.optimizers...LinearWarmupCosineAnnealingLR` may fail to install on
  Kaggle. Fallback: a plain `torch.optim.lr_scheduler` (cosine + warmup) if the
  import breaks. Verify at setup.

## 6. Milestones

- **M1 — Vid-only smoke test.** Pipeline runs end-to-end on one T4 without OOM;
  training loss decreases; `generate` produces coherent English on a few test
  sentences; BLEU/ROUGE computed on a small subset. Proves the harness.
- **M2 — +Prev ablation.** Add GT previous sentence; re-run and report
  **Vid vs Vid+Prev** on the same subset to demonstrate the contextual-cue
  benefit (the paper's thesis).

## 7. Risks & things to verify on data download

1. **I3D feature dimension & sequence length** — confirm dim (assumed 1024) and
   avg `T`; if `T` is large, switch projector to the conv variant.
2. **Cue-arg guards** — ensure `pls=None`, `bg_description=None`, `spottings=[]`
   flow through `collate_fn_padd_t`, `VggSLTNet.forward`, and
   `LanguageDecoder._process/_process_predict` without crashing (all cue flags
   off means those branches are skipped, but verify None handling).
3. **Qwen tokenizer** — `LanguageDecoder` manually appends EOS and derives
   `embed_tokens = decoder.model.embed_tokens`; confirm this holds for
   `Qwen2ForCausalLM` and that pad/eos fallback works.
4. **`.tsv` sentence ordering** — confirm clip ids encode `video_id` +
   sentence index so previous-sentence reconstruction is correct.
5. **Kaggle ops** — 12 h/session + 30 h/week: verify checkpoint/resume works;
   upload features once as a private Kaggle Dataset (I3D ≈ a few GB, well within
   limits).
6. **fp16 stability** — watch for NaN/overflow; Lightning loss scaling should
   handle it, but keep `gradient_clip_val: 1.0`.

## 8. Verification / definition of done

- `M1`: one training run completes ≥1 epoch on T4, no OOM; eval prints sample
  predictions + a BLEU-4/ROUGE number on a held-out subset.
- `M2`: Vid+Prev run completes; a side-by-side Vid vs Vid+Prev metric table is
  produced on the same subset.
- All claims backed by actual command output (no "should work").
