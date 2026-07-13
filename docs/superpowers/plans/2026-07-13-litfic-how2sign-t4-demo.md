# LiTFiC How2Sign T4 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the LiTFiC pipeline on How2Sign (ASL→English) with a Vid+Prev demo on Kaggle 2×T4, reusing public I3D features and Qwen2.5-3B.

**Architecture:** A new `How2SignSentences` dataset reads per-sentence `.npy` I3D features + a `.tsv` and emits the exact item dict the existing `collate_fn` contract expects, bypassing all BOBSL machinery (Subtitles/LMDB/PG/spottings). The model (`VggSLTNet`) and its forward path stay unchanged; the previous-sentence cue is threaded through a dedicated collate that adds a `rec_prev` key so GT-prev is present at both train and inference. New Hydra configs point at Kaggle-mounted data and apply T4 patches (fp16 + sdpa).

**Tech Stack:** PyTorch 2.3, PyTorch Lightning, Hydra, HuggingFace Transformers + PEFT (LoRA), Qwen2.5-3B, pytest.

## Global Constraints

- Precision: `16-mixed` (fp16) — T4 has no bf16 tensor cores. Never `bf16-mixed` on T4.
- Attention: `attn_implementation: sdpa` — FlashAttention-2 requires Ampere+, unsupported on T4.
- LLM: `Qwen/Qwen2.5-3B`, embedding dim **2048** → `mm_projector_config.hidden_size: 2048`.
- I3D feature dim: **1024** → `mm_hidden_size: 1024` and `feats_dim: 1024` (VERIFY on real download; single source of truth for the number).
- Cues: **Vid + Prev only**. `use_pl_w_feats=False`, `bg_desc=False`, `use_spottings=False`, `use_bg_words=False`. Prev = GT previous sentence (oracle).
- Prev flags for M2: `use_rec_prev: True`, `use_gt_prev: False`, `mix_in_prev_prob: 1.0`.
- Trainer: single GPU (`devices: [0]`), `accumulate_grad_batches: 16`, keep `gradient_checkpointing_enable: True` and `gradient_clip_val: 1.0`.
- Run all pytest from repo root: `python -m pytest -q`.
- Branch: `how2sign-t4-demo` (already created; spec committed).

---

### Task 1: `How2SignSentences` dataset

**Files:**
- Create: `src/data/components/how2sign.py`
- Test: `tests/data/test_how2sign.py`

**Interfaces:**
- Consumes: `src.utils.data_utils.cleanup_sub`, `remove_words`.
- Produces:
  - `INITIAL_PROMPT: str`
  - `parse_clip_id(clip_id: str) -> tuple[str, int]` → `(video_id, sent_idx)`
  - `class How2SignSentences(Dataset)` with `__init__(setname, tsv_dir, feats_dir, feats_dim=1024, use_prev=False, sub_aug_drop=False, aug_drop_pct=0.2, id_col="id", text_col="translation", npy_col="npy_path")`, `__len__`, `__getitem__(idx) -> dict`.
  - `__getitem__` dict keys (contract for collate): `subtitle:str, features:Tensor[T,1024], question:str, previous_context:str, pls:None, bg_description:None, spottings:list, sub_start:float, sub_end:float, video_name:str, id:str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_how2sign.py
import csv, os
import numpy as np
import torch
import pytest
from src.data.components.how2sign import How2SignSentences, parse_clip_id, INITIAL_PROMPT


def _make_dataset(tmp_path, setname="train"):
    feats_dir = tmp_path / "feats"; feats_dir.mkdir()
    tsv_dir = tmp_path / "tsv"; tsv_dir.mkdir()
    # video A: two ordered sentences; video B: one sentence
    rows = [
        {"id": "vidA_0-5-rgb_front", "translation": "first sentence", "npy_path": "vidA_0.npy"},
        {"id": "vidA_1-5-rgb_front", "translation": "second sentence", "npy_path": "vidA_1.npy"},
        {"id": "vidB_0-5-rgb_front", "translation": "solo sentence", "npy_path": "vidB_0.npy"},
    ]
    np.save(feats_dir / "vidA_0.npy", np.zeros((5, 1024), dtype=np.float32))
    np.save(feats_dir / "vidA_1.npy", np.zeros((7, 1024), dtype=np.float32))
    np.save(feats_dir / "vidB_0.npy", np.zeros((3, 1024), dtype=np.float32))
    with open(tsv_dir / f"how2sign_{setname}.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "translation", "npy_path"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(tsv_dir), str(feats_dir)


def test_parse_clip_id():
    assert parse_clip_id("--7E2sU6zP4_10-5-rgb_front") == ("--7E2sU6zP4", 10)
    assert parse_clip_id("vidA_0-5-rgb_front") == ("vidA", 0)


def test_len_and_feature_shape(tmp_path):
    tsv_dir, feats_dir = _make_dataset(tmp_path)
    ds = How2SignSentences("train", tsv_dir, feats_dir, feats_dim=1024, use_prev=True)
    assert len(ds) == 3
    item = ds[0]
    assert isinstance(item["features"], torch.Tensor)
    assert item["features"].shape[1] == 1024
    assert item["question"] == INITIAL_PROMPT
    assert item["pls"] is None and item["bg_description"] is None
    assert item["spottings"] == []


def test_previous_context_reconstruction(tmp_path):
    tsv_dir, feats_dir = _make_dataset(tmp_path)
    ds = How2SignSentences("train", tsv_dir, feats_dir, use_prev=True)
    by_id = {ds[i]["id"]: ds[i] for i in range(len(ds))}
    assert by_id["vidA_0-5-rgb_front"]["previous_context"] == ""
    assert by_id["vidA_1-5-rgb_front"]["previous_context"] == "first sentence"
    assert by_id["vidB_0-5-rgb_front"]["previous_context"] == ""


def test_prev_disabled_gives_empty(tmp_path):
    tsv_dir, feats_dir = _make_dataset(tmp_path)
    ds = How2SignSentences("train", tsv_dir, feats_dir, use_prev=False)
    assert all(ds[i]["previous_context"] == "" for i in range(len(ds)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_how2sign.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.components.how2sign'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/components/how2sign.py
"""How2Sign sentence dataset: reads per-sentence I3D .npy + .tsv metadata
and emits the item dict consumed by the collate/model contract. Bypasses all
BOBSL machinery (Subtitles/LMDB/pseudo-labels/spottings)."""
import os
import re
import csv
import random
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.data_utils import cleanup_sub, remove_words

INITIAL_PROMPT = (
    "You are an AI assistant designed to interpret a video of a sign language "
    "signing sequence and translate it into English."
)

_ID_RE = re.compile(r"^(?P<vid>.+?)_(?P<idx>\d+)(?:[-_].*)?$")


def parse_clip_id(clip_id: str) -> Tuple[str, int]:
    """Extract (video_id, sentence_index) from a How2Sign clip id such as
    '--7E2sU6zP4_10-5-rgb_front' → ('--7E2sU6zP4', 10)."""
    m = _ID_RE.match(clip_id)
    if not m:
        raise ValueError(f"Cannot parse clip id: {clip_id}")
    return m.group("vid"), int(m.group("idx"))


class How2SignSentences(Dataset):
    def __init__(
        self,
        setname: str,
        tsv_dir: str,
        feats_dir: str,
        feats_dim: int = 1024,
        use_prev: bool = False,
        sub_aug_drop: bool = False,
        aug_drop_pct: float = 0.2,
        id_col: str = "id",
        text_col: str = "translation",
        npy_col: str = "npy_path",
        **kwargs,
    ):
        self.setname = setname
        self.feats_dir = feats_dir
        self.feats_dim = feats_dim
        self.use_prev = use_prev
        self.sub_aug_drop = sub_aug_drop
        self.aug_drop_pct = aug_drop_pct
        self.id_col, self.text_col, self.npy_col = id_col, text_col, npy_col

        tsv_path = os.path.join(tsv_dir, f"how2sign_{setname}.tsv")
        rows = self._read_tsv(tsv_path)
        self.items = self._build_index(rows)

    def _read_tsv(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def _build_index(self, rows):
        # group by video, sort by sentence index, attach previous translation
        by_video = defaultdict(list)
        for r in rows:
            vid, idx = parse_clip_id(r[self.id_col])
            by_video[vid].append((idx, r))
        items = []
        for vid, lst in by_video.items():
            lst.sort(key=lambda x: x[0])
            prev_text = ""
            for _idx, r in lst:
                items.append(
                    {
                        "id": r[self.id_col],
                        "video_name": vid,
                        "text": r[self.text_col],
                        "npy": r[self.npy_col],
                        "prev": prev_text if self.use_prev else "",
                    }
                )
                prev_text = r[self.text_col]
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        feats = np.load(os.path.join(self.feats_dir, it["npy"]))
        feats = torch.from_numpy(np.asarray(feats, dtype=np.float32))
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        subtitle = cleanup_sub(it["text"])
        if self.sub_aug_drop and self.setname == "train" and random.random() < 0.5:
            subtitle = remove_words(subtitle, max_p=self.aug_drop_pct)
        return {
            "subtitle": subtitle,
            "features": feats,
            "question": INITIAL_PROMPT,
            "previous_context": it["prev"],
            "pls": None,
            "bg_description": None,
            "spottings": [],
            "sub_start": 0.0,
            "sub_end": 0.0,
            "video_name": it["video_name"],
            "id": it["id"],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_how2sign.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/components/how2sign.py tests/data/test_how2sign.py
git commit -m "feat(data): add How2SignSentences dataset (npy+tsv, GT prev)"
```

---

### Task 2: How2Sign collate with `rec_prev`

**Files:**
- Modify: `src/data/components/how2sign.py` (append collate function)
- Test: `tests/data/test_how2sign_collate.py`

**Interfaces:**
- Consumes: `src.data.components.sentence.pad_tensors_and_create_attention_masks`, `How2SignSentences.__getitem__` dict.
- Produces: `collate_fn_padd_h2s(batch: list[dict]) -> dict` with keys `features:Tensor[B,Tmax,1024], attn_masks:Tensor[B,Tmax], subtitles:list[str], questions:list[str], previous_contexts:list[str], pls:list[None], bg_description:list[None], spottings:list[list], rec_prev:list[list[str]], video_names:list[str], ids:list[str], start:list, end:list`.
- Note: `rec_prev[i]` is `[prev]` when prev is non-empty else `[]` — the format `LanguageDecoder._process_predict` expects (it does `". ".join(r)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_how2sign_collate.py
import csv
import numpy as np
import torch
from src.data.components.how2sign import How2SignSentences, collate_fn_padd_h2s


def _ds(tmp_path):
    feats_dir = tmp_path / "feats"; feats_dir.mkdir()
    tsv_dir = tmp_path / "tsv"; tsv_dir.mkdir()
    rows = [
        {"id": "vidA_0-5-rgb_front", "translation": "first", "npy_path": "a0.npy"},
        {"id": "vidA_1-5-rgb_front", "translation": "second", "npy_path": "a1.npy"},
    ]
    np.save(feats_dir / "a0.npy", np.zeros((5, 1024), dtype=np.float32))
    np.save(feats_dir / "a1.npy", np.zeros((7, 1024), dtype=np.float32))
    with open(tsv_dir / "how2sign_train.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "translation", "npy_path"], delimiter="\t")
        w.writeheader()
        [w.writerow(r) for r in rows]
    return How2SignSentences("train", str(tsv_dir), str(feats_dir), use_prev=True)


def test_collate_shapes_and_rec_prev(tmp_path):
    ds = _ds(tmp_path)
    batch = collate_fn_padd_h2s([ds[0], ds[1]])
    assert batch["features"].shape == (2, 7, 1024)   # padded to max T
    assert batch["attn_masks"].shape == (2, 7)
    assert batch["questions"][0] is not None
    # first sentence has no prev, second has prev "first"
    prevs = {i: p for i, p in zip(batch["ids"], batch["rec_prev"])}
    assert prevs["vidA_0-5-rgb_front"] == []
    assert prevs["vidA_1-5-rgb_front"] == ["first"]
    assert batch["pls"] == [None, None]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_how2sign_collate.py -q`
Expected: FAIL — `ImportError: cannot import name 'collate_fn_padd_h2s'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/data/components/how2sign.py
from src.data.components.sentence import pad_tensors_and_create_attention_masks


def collate_fn_padd_h2s(batch):
    features = [item["features"] for item in batch]
    padded_features, attn_masks = pad_tensors_and_create_attention_masks(
        features, padding_side="right"
    )
    prevs = [item["previous_context"] for item in batch]
    rec_prev = [[p] if p else [] for p in prevs]
    return {
        "features": padded_features,
        "attn_masks": attn_masks,
        "subtitles": [item["subtitle"] for item in batch],
        "questions": [item["question"] for item in batch],
        "previous_contexts": prevs,
        "pls": [item["pls"] for item in batch],
        "bg_description": [item["bg_description"] for item in batch],
        "spottings": [item["spottings"] for item in batch],
        "rec_prev": rec_prev,
        "start": [item["sub_start"] for item in batch],
        "end": [item["sub_end"] for item in batch],
        "video_names": [item["video_name"] for item in batch],
        "ids": [item["id"] for item in batch],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_how2sign_collate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/components/how2sign.py tests/data/test_how2sign_collate.py
git commit -m "feat(data): add collate_fn_padd_h2s carrying rec_prev for GT-prev at inference"
```

---

### Task 3: Wire `How2SignSentences` into the datamodule

**Files:**
- Modify: `src/data/slt_datamodule.py:57-67` (dataset construction block)
- Test: `tests/data/test_datamodule_how2sign.py`

**Interfaces:**
- Consumes: `How2SignSentences`, existing `SLTDataModule.__init__` signature.
- Produces: when `dataset == "how2sign"`, `setup()` builds `data_train/val/test` from `How2SignSentences`; `data_val`/`data_test` use `setname="test"` (mirrors the existing how2sign branch that maps val→test).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_datamodule_how2sign.py
import csv, json
import numpy as np
from src.data.slt_datamodule import SLTDataModule


def _prep(tmp_path):
    feats_dir = tmp_path / "feats"; feats_dir.mkdir()
    tsv_dir = tmp_path / "tsv"; tsv_dir.mkdir()
    for split in ["train", "test"]:
        rows = [{"id": f"vidA_{i}-5-rgb_front", "translation": f"s{i}", "npy_path": f"{split}{i}.npy"} for i in range(2)]
        for i in range(2):
            np.save(feats_dir / f"{split}{i}.npy", np.zeros((4, 1024), dtype=np.float32))
        with open(tsv_dir / f"how2sign_{split}.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "translation", "npy_path"], delimiter="\t")
            w.writeheader(); [w.writerow(r) for r in rows]
    idx = tmp_path / "idx.json"; idx.write_text(json.dumps([0]))
    return str(tsv_dir), str(feats_dir), str(idx)


def test_datamodule_builds_how2sign(tmp_path):
    tsv_dir, feats_dir, idx = _prep(tmp_path)
    dm = SLTDataModule(
        dataset="how2sign",
        val_episode_ind_path=idx,
        test_episode_ind_path=idx,
        dataset_config={"tsv_dir": tsv_dir, "feats_dir": feats_dir, "feats_dim": 1024, "use_prev": True},
        batch_size=2, num_workers=0,
        collate_fn="src.data.components.how2sign.collate_fn_padd_h2s",
        test_setname="test",
    )
    dm.setup()
    assert len(dm.data_train) == 2
    assert len(dm.data_test) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_datamodule_how2sign.py -q`
Expected: FAIL — `Sentences.__init__` errors on unexpected kwargs (`tsv_dir`/`feats_dir`), because the how2sign branch still builds `Sentences`.

- [ ] **Step 3: Write minimal implementation**

Replace the construction block in `src/data/slt_datamodule.py` (the `if not self.data_train ...` body, lines ~57-67):

```python
        if not self.data_train and not self.data_val and not self.data_test:
            if self.dataset == "how2sign":
                from src.data.components.how2sign import How2SignSentences
                self.data_train = How2SignSentences(**self.hparams.dataset_config, setname="train")
                self.data_val = How2SignSentences(**self.hparams.dataset_config, setname="test")
                self.data_test = How2SignSentences(**self.hparams.dataset_config, setname=self.test_setname)
            else:
                self.data_train = Sentences(**self.hparams.dataset_config, setname="train")
                self.data_val = Sentences(**self.hparams.dataset_config, setname="val")
                self.data_test = Sentences(**self.hparams.dataset_config, setname=self.test_setname)
```

Note: `How2SignSentences.__init__` already accepts `**kwargs`, so extra config keys are tolerated.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/data/test_datamodule_how2sign.py -q`
Expected: PASS. Also run the full suite: `python -m pytest -q` → all green (BOBSL path unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/data/slt_datamodule.py tests/data/test_datamodule_how2sign.py
git commit -m "feat(data): construct How2SignSentences for dataset=how2sign"
```

---

### Task 4: Hydra configs (data, paths, experiments) + compose test

**Files:**
- Create: `configs/data/how2sign.yaml`
- Create: `configs/paths/how2sign.yaml`
- Create: `configs/experiment/how2sign-vid.yaml`
- Create: `configs/experiment/how2sign-vid+prev.yaml`
- Test: `tests/configs/test_how2sign_configs.py`

**Interfaces:**
- Consumes: existing `configs/model/vgg_slt.yaml`, `configs/trainer/gpu.yaml`, `configs/train.yaml`.
- Produces: two experiment configs that compose without error and encode the Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
# tests/configs/test_how2sign_configs.py
import hydra
from hydra import compose, initialize_config_dir
import os

CONFIG_DIR = os.path.abspath("configs")


def _cfg(exp):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        return compose(config_name="train", overrides=[f"experiment={exp}"])


def test_vid_config_t4_patches():
    cfg = _cfg("how2sign-vid")
    assert cfg.trainer.precision == "16-mixed"
    assert cfg.model.net.llm_config.decoder_config.attn_implementation == "sdpa"
    assert cfg.model.net.mm_projector_config.mm_hidden_size == 1024
    assert cfg.model.net.mm_projector_config.hidden_size == 2048
    assert cfg.data.dataset == "how2sign"
    assert cfg.model.net.llm_config.use_rec_prev is False


def test_vidprev_enables_prev():
    cfg = _cfg("how2sign-vid+prev")
    assert cfg.model.net.llm_config.use_rec_prev is True
    assert cfg.model.net.llm_config.use_gt_prev is False
    assert cfg.model.net.llm_config.mix_in_prev_prob == 1.0
    assert cfg.data.dataset_config.use_prev is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/configs/test_how2sign_configs.py -q`
Expected: FAIL — `MissingConfigException` (experiment `how2sign-vid` not found).

- [ ] **Step 3: Write the config files**

```yaml
# configs/data/how2sign.yaml
_target_: src.data.slt_datamodule.SLTDataModule
batch_size: 1
num_workers: 2
pin_memory: True
collate_fn: src.data.components.how2sign.collate_fn_padd_h2s
dataset: how2sign
eval_data_size: -1
train_data_fraction: 1.0
ret_data_size: 100
val_episode_ind_path: ${paths.val_episode_ind_path}
test_episode_ind_path: ${paths.test_episode_ind_path}
test_setname: test
dataset_config:
  tsv_dir: ${paths.h2s_tsv_dir}
  feats_dir: ${paths.h2s_feats_dir}
  feats_dim: 1024
  use_prev: False
  sub_aug_drop: True
  aug_drop_pct: 0.2
```

```yaml
# configs/paths/how2sign.yaml
defaults:
  - default
# Override llm + How2Sign data roots (point these at the Kaggle mount)
llm_root: Qwen/Qwen2.5-3B
h2s_tsv_dir: /kaggle/input/how2sign-i3d/tsv
h2s_feats_dir: /kaggle/input/how2sign-i3d/feats
val_episode_ind_path: ./val_start_indices.json
test_episode_ind_path: ./test_start_indices.json
```

```yaml
# configs/experiment/how2sign-vid.yaml
# @package _global_
defaults:
  - override /data: how2sign
  - override /model: vgg_slt
  - override /paths: how2sign
  - override /callbacks: default
  - override /trainer: gpu

tags: ["how2sign", "vid"]
seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  precision: 16-mixed
  devices: [0]
  accumulate_grad_batches: 16
  gradient_clip_val: 1.0

data:
  batch_size: 1
  dataset_config:
    use_prev: False

model:
  net:
    mm_projector_config:
      mm_hidden_size: 1024
      hidden_size: 2048
      projector_type: mlp2x_gelu
    llm_config:
      decoder_config:
        attn_implementation: sdpa
      use_pl_w_feats: False
      use_rec_prev: False
      use_bg_words: False
      bg_desc: False
      use_spottings: False
```

```yaml
# configs/experiment/how2sign-vid+prev.yaml
# @package _global_
defaults:
  - override /data: how2sign
  - override /model: vgg_slt
  - override /paths: how2sign
  - override /callbacks: default
  - override /trainer: gpu

tags: ["how2sign", "vid+prev"]
seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  precision: 16-mixed
  devices: [0]
  accumulate_grad_batches: 16
  gradient_clip_val: 1.0

data:
  batch_size: 1
  dataset_config:
    use_prev: True

model:
  net:
    mm_projector_config:
      mm_hidden_size: 1024
      hidden_size: 2048
      projector_type: mlp2x_gelu
    llm_config:
      decoder_config:
        attn_implementation: sdpa
      use_pl_w_feats: False
      use_rec_prev: True
      use_gt_prev: False
      mix_in_prev_prob: 1.0
      use_bg_words: False
      bg_desc: False
      use_spottings: False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/configs/test_how2sign_configs.py -q`
Expected: PASS. (If `initialize_config_dir` errors on missing `hydra`/`extras` groups, add `overrides=[..., "hydra/job_logging=disabled"]` as needed — but the default `train.yaml` should compose.)

- [ ] **Step 5: Commit**

```bash
git add configs/data/how2sign.yaml configs/paths/how2sign.yaml configs/experiment/how2sign-vid.yaml configs/experiment/how2sign-vid+prev.yaml tests/configs/test_how2sign_configs.py
git commit -m "feat(config): add How2Sign data/paths/experiment configs with T4 patches"
```

---

### Task 5: Kaggle data acquisition + real-schema verification (integration)

**Files:**
- Create: `docs/superpowers/plans/kaggle-runbook.md` (operational notes + exact cells)
- Modify (if needed): `src/data/components/how2sign.py` (adjust `id_col`/`text_col`/`npy_col`/`parse_clip_id` to real schema)

**This is a run-and-observe task — no pytest.**

- [ ] **Step 1: Download the public I3D features + tsv**

On a machine with internet (or a Kaggle notebook with internet enabled), fetch the UPC How2Sign SLT release referenced by `imatge-upc/slt_how2sign_wicv2023` (I3D `.npy` per sentence + `.tsv` per split, hosted on the linked dataverse). Record the download URLs and total size in `kaggle-runbook.md`.

- [ ] **Step 2: Inspect the real `.tsv` schema and one `.npy`**

Run:
```bash
python - <<'PY'
import csv, numpy as np, glob, os
tsv = glob.glob("**/*train*.tsv", recursive=True)[0]
with open(tsv, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t"); print("COLUMNS:", r.fieldnames)
    row = next(r); print("ROW0:", row)
npy = glob.glob("**/*.npy", recursive=True)[0]
a = np.load(npy); print("NPY", os.path.basename(npy), a.shape, a.dtype)
PY
```
Expected observation: column names (map to `id_col`/`text_col`/`npy_col`), a sample clip id (confirm `parse_clip_id` regex), and the feature array shape → **confirm dim == 1024 and read the avg/max T**.

- [ ] **Step 3: Reconcile schema**

If columns differ (e.g. tgt text is `tgt_text`, path is `audio`/`feat`), update the defaults in `How2SignSentences.__init__` and/or pass `id_col`/`text_col`/`npy_col` via `dataset_config`. If `T` is large (say > ~150), set `projector_type: conv_K5_P2_KP2_L2` in both experiment configs and set `mm_hidden_size` accordingly (conv keeps dim, so still 1024). If dim ≠ 1024, update `feats_dim`/`mm_hidden_size` everywhere (Global Constraints).

- [ ] **Step 4: Package as a private Kaggle Dataset**

Upload `feats/` + `tsv/` as a private Kaggle Dataset named `how2sign-i3d` mounted at `/kaggle/input/how2sign-i3d/` (matches `configs/paths/how2sign.yaml`). Note the exact mount path; adjust the paths config if different.

- [ ] **Step 5: Commit any schema fixes**

```bash
git add src/data/components/how2sign.py docs/superpowers/plans/kaggle-runbook.md
git commit -m "chore(data): reconcile How2Sign real tsv/npy schema + runbook"
```

---

### Task 6: Milestone M1 — Vid-only smoke run on Kaggle T4 (integration)

**Files:** none (uses `src/train.py`, `src/eval.py`)

**This is a run-and-observe task — no pytest.** Verifies the harness end-to-end.

- [ ] **Step 1: Environment setup cell (Kaggle)**

```bash
pip install -r requirements.txt
pip install "transformers>=4.45" peft accelerate
# pl_bolts fallback: if the LinearWarmupCosineAnnealingLR import fails at runtime,
# note it here and switch model/scheduler to a torch cosine scheduler (see Task 6 Step 5).
export PROJECT_ROOT=$(pwd)
```

- [ ] **Step 2: Overfit sanity (tiny subset)**

Run a 1-epoch run limited to a few batches to confirm no OOM / no crash and that loss is finite:
```bash
python src/train.py experiment=how2sign-vid \
  trainer.max_epochs=1 +trainer.limit_train_batches=20 +trainer.limit_val_batches=5
```
Expected observation: process reaches training steps on `cuda`, `train/loss` printed and finite (not NaN), no OOM. If OOM: reduce further or switch `projector_type` to the conv variant and/or set `+trainer.limit_train_batches` lower.

- [ ] **Step 3: Short real training run with checkpointing**

```bash
python src/train.py experiment=how2sign-vid \
  trainer.max_epochs=3 \
  callbacks.model_checkpoint.every_n_train_steps=500
```
Expected observation: `train/loss` decreases across steps; a checkpoint `.ckpt` is written under the run's `output_dir`. Record the checkpoint path.

- [ ] **Step 4: Evaluate + inspect predictions**

```bash
python src/eval.py experiment=how2sign-vid ckpt_path={CKPT_PATH} \
  +trainer.limit_test_batches=50
```
Expected observation: BLEU-4 / ROUGE-L numbers printed for the subset, and at least a few English predictions that are grammatical (need not be accurate). Save the numbers as the **M1 baseline** in `kaggle-runbook.md`.

- [ ] **Step 5: If `pl_bolts` scheduler breaks — apply fallback**

Only if Step 1/2 failed importing `pl_bolts`: in `configs/model/vgg_slt.yaml` replace the scheduler with a stock torch cosine schedule (warmup handled by keeping `warmup_epochs` semantics via `torch.optim.lr_scheduler.CosineAnnealingLR`), re-run Step 2, then commit:
```bash
git add configs/model/vgg_slt.yaml docs/superpowers/plans/kaggle-runbook.md
git commit -m "fix(train): torch cosine scheduler fallback for Kaggle"
```

- [ ] **Step 6: Record M1 result**

Append the M1 command, checkpoint path, and metric numbers to `kaggle-runbook.md` and commit.
```bash
git add docs/superpowers/plans/kaggle-runbook.md
git commit -m "docs: record M1 vid-only smoke results"
```

---

### Task 7: Milestone M2 — Vid+Prev ablation on Kaggle T4 (integration)

**Files:** none

**This is a run-and-observe task — no pytest.** Demonstrates the paper's thesis.

- [ ] **Step 1: Confirm prev reaches the prompt (debug print)**

Temporarily add a one-off print in `LanguageDecoder._process_predict` after the `questions = [q + ' The previous context...` line to log one prompt, then run:
```bash
python src/eval.py experiment=how2sign-vid+prev ckpt_path={M1_CKPT} \
  +trainer.limit_test_batches=2
```
Expected observation: printed prompt contains `The previous context is the following: <a real sentence>` for non-first sentences. Remove the debug print afterward. (This catches the `rec_prev` plumbing silently degrading to Vid-only.)

- [ ] **Step 2: Train Vid+Prev**

```bash
python src/train.py experiment=how2sign-vid+prev \
  trainer.max_epochs=3 \
  callbacks.model_checkpoint.every_n_train_steps=500
```
Expected observation: training runs, loss finite/decreasing, checkpoint written. Record path.

- [ ] **Step 3: Evaluate Vid+Prev on the same subset**

```bash
python src/eval.py experiment=how2sign-vid+prev ckpt_path={M2_CKPT} \
  +trainer.limit_test_batches=50
```
Expected observation: BLEU-4/ROUGE-L on the **same** 50-batch subset as M1.

- [ ] **Step 4: Produce the ablation table**

In `kaggle-runbook.md`, record a side-by-side table: `Vid` (M1) vs `Vid+Prev` (M2) on the identical subset (BLEU-4, ROUGE-L), plus 2-3 qualitative examples where prev changed the output. Demo is "done" when Vid+Prev is reported alongside Vid (direction of change noted honestly, even if small on a short run).

- [ ] **Step 5: Commit results + open PR**

```bash
git add docs/superpowers/plans/kaggle-runbook.md
git commit -m "docs: record M2 vid+prev ablation results"
git push -u origin how2sign-t4-demo
```
Then open a PR from `how2sign-t4-demo` summarizing the demo, milestones, and the Vid vs Vid+Prev table.

---

## Self-Review

**Spec coverage:**
- Vid+Prev, public I3D, Qwen2.5-3B, T4 patches → Tasks 1-4 (code/config), 6-7 (runs). ✓
- Bypass BOBSL machinery → Task 1 (new dataset). ✓
- rec_prev plumbing for prev-at-inference → Task 2 + Task 7 Step 1 verification. ✓
- Datamodule how2sign branch → Task 3. ✓
- T4 fp16/sdpa/single-GPU/grad-accum/checkpoint-resume → Task 4 configs + Task 6. ✓
- Milestones M1/M2 → Tasks 6/7. ✓
- Risks: I3D dim/length → Task 5 Step 2-3; tokenizer Qwen → exercised in Task 6 Step 2; tsv ordering → Task 1 tests + Task 5; pl_bolts → Task 6 Step 5; fp16 stability → grad clip in configs + Task 6 observation. ✓
- Non-goals (PG/BG/LLM-eval/reproduction) → excluded by config flags; not implemented. ✓

**Placeholder scan:** `{CKPT_PATH}`/`{M1_CKPT}`/`{M2_CKPT}` are runtime-produced paths recorded during execution (legitimate), not plan placeholders. Real download URLs are intentionally discovered in Task 5 Step 1 (internet required at execution). No TODO/TBD in code.

**Type consistency:** `How2SignSentences.__getitem__` dict keys ↔ `collate_fn_padd_h2s` reads ↔ `VggSLTNet.forward`/`LanguageDecoder` expected keys all aligned (`features/attn_masks/subtitles/questions/previous_contexts/pls/bg_description/spottings/rec_prev`). `mm_hidden_size=1024`, `hidden_size=2048` consistent across configs and tests. `parse_clip_id` signature identical in impl and tests.
