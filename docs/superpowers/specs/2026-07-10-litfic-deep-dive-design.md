# LiTFiC Re-run for Deep Understanding — Design Spec

Date: 2026-07-10
Repo: `D:\code\LiTFiC` (reproduction of *Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues*, CVPR 2025, arXiv:2501.09754v2)

## Goal

Gain the deepest possible understanding of the LiTFiC paper by studying it against
its code, verifying the pipeline actually runs, and preparing a full reproduction
runbook — under the hard constraint that the local machine (GTX 1650, 4GB VRAM,
Windows) cannot run the real training (paper uses 4× H100, Llama-3-8B, BOBSL LMDB ~100s GB).

Chosen direction: **combined** — understanding doc + local smoke-test + cloud runbook.

## Hard Constraints

- Local GPU: GTX 1650, 4 GB VRAM. Cannot hold Llama-3-8B (even in LoRA/bf16).
- Real data (BOBSL video-feature LMDB, CSLR annotations 1.9 GB, Llama-3-8B ~16 GB,
  BLEURT-20) is not present locally and is very large / gated (Llama needs HF access).
- `flash-attn` (in requirements, hardcoded in `configs/model/vgg_slt.yaml`
  `decoder_config.attn_implementation: flash_attention_2`) will not build on this
  environment — smoke-test must override to `eager` + `32-true` precision.

## The Three Parts (execution order A → B → C)

### Part A — Deep-dive documentation (feasible now, no GPU)

Deliverable: `docs/litfic-deep-dive.md`. Maps paper ↔ code across:

1. **Problem & contributions** (paper Sec. 1) — SLT on BOBSL with contextual cues.
2. **Data pipeline** — `SLTDataModule` → `Sentences` (`sentence.py`) → `Subtitles`
   + `LMDBLoader` (video features + pseudo-labels) → `collate_fn_padd_t`. What one
   sample contains: `features`, `pls`, `previous_contexts`, `bg_description`,
   `spottings`, `subtitle` (target).
3. **Prompt construction** — exact trace of `LanguageDecoder._process()` /
   `_process_predict()`: order = instruction/question → prev → PG → BG → video tokens
   → "Generated Sentence:" → subtitle target; label masking (`-100`) so CE is only
   over the subtitle tokens.
4. **Model** — `MMProjector` (768→4096, `mlp2x_gelu`), `LanguageDecoder`
   (Llama-3-8B + LoRA r=4/α=16 on `q_proj`,`v_proj`; embed frozen).
5. **Training & augmentations** — map the 4 `configs/experiment/*` to Table 1 rows;
   `drop_words` / drop-cue (mix_in_*_prob) / PrevPred↔GT to Table 2. Optimizer/
   scheduler (Adam 1e-4, LinearWarmupCosineAnnealing, 5-epoch warmup, 10 epochs).
6. **Evaluation** — `eval.py`, DDP per-episode splitting via
   val/test start-indices; metrics in `slt_module`: BLEU-4, ROUGE-L, CIDEr,
   BLEURT, IoU (+synonym/lemmatize), and `llm_eval.py` (CLAIR / GPT-4o-mini).
7. **Results reference table** — paper Table 1/2/3 numbers for later comparison.
8. **Reproduction gotchas** — flash-attn, VRAM, gated data, path config.

### Part B — Local smoke-test (verify the pipeline)

Goal: run `src/train.py` for a few steps on CPU / 4GB GPU with:
- A tiny causal LM (e.g. `HuggingFaceTB/SmolLM-135M` or `sshleifer/tiny-gpt2`)
  replacing Llama-3-8B via config overrides (`paths.llm_root`, hidden size must be
  reconciled with `mm_projector_config.hidden_size`).
- A minimal mock dataset (a handful of fake samples) OR `configs/debug/limit`.
- `attn_implementation=eager`, `precision=32-true`, `lora` on a small target.
- Success = forward loss computes, prompt assembles correctly, one sentence
  generates. Not paper numbers — a liveness proof.

Out of scope for B: real BOBSL data, real metrics, matching paper scores.

### Part C — Cloud reproduction runbook (document only, no execution)

Deliverable: `docs/litfic-cloud-runbook.md`. Steps: provision 4× A100/H100; download
BOBSL features + CSLR annotations + BLIP2 captions + Llama-3-8B (HF gated) + extra
resources (start-indices, prev captions); edit `configs/paths/default.yaml`; per-
modality train commands; eval with pretrained `bobsl_all.ckpt`; rough time/cost.

## Non-Goals

- Reproducing paper metrics locally.
- Modifying the model/method.
- Downloading the real datasets.

## Success Criteria

- A: doc lets a reader explain any component and where it lives in code.
- B: a runnable command that exits cleanly having done ≥1 train step + 1 generate.
- C: a checklist someone with cloud GPUs could follow end-to-end.
