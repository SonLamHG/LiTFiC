# Kaggle Runbook — LiTFiC How2Sign Vid+Prev demo (single T4)

Paste these cells **in order** into a fresh Kaggle notebook and run top-to-bottom.
Branch `how2sign-t4-demo` on `github.com/sonlamhg/LiTFiC`. Data: CSUC Dataverse
DOI `10.34810/data693`. LLM: Qwen2.5-3B. T4 patches (fp16 + sdpa) are baked into
the configs.

## Prerequisites (UI, do once)
- Right panel → **Settings → Accelerator → GPU T4 ×2** (or T4 x1).
- **Internet → On** (Kaggle notebooks: on by default for phone-verified accounts).
- Do **not** install `flash-attn` (unsupported on T4).

---

### Cell 1 — Clone code
```python
!git clone -b how2sign-t4-demo https://github.com/sonlamhg/LiTFiC.git /kaggle/working/LiTFiC
%cd /kaggle/working/LiTFiC
!git log --oneline -3
```

### Cell 2 — Install deps (no flash-attn)
```python
# single line — multi-line `\` continuation breaks inside a Kaggle `!` cell
!pip install -q lightning==2.3.0 torchmetrics hydra-core==1.3.2 hydra-colorlog==1.2.0 omegaconf rich rootutils einops lmdb transformers==4.45.2 peft==0.12.0 sentencepiece lightning-utilities==0.11.2 nltk pycocoevalcap
import lightning, hydra, transformers, peft  # verify they really installed
print("deps OK", lightning.__version__, transformers.__version__)
```
(The LR scheduler is now vendored in-repo, so `lightning-bolts`/`pl_bolts` is no longer needed.)

### Cell 3 — Env + episode-index files (whole-set iteration)
```python
import os, json
os.environ["PROJECT_ROOT"] = "/kaggle/working/LiTFiC"
for p in ["val_start_indices.json", "test_start_indices.json"]:
    json.dump({"idx": [0]}, open(p, "w"))
# common path overrides for every train/eval call:
OVR = "paths.h2s_tsv_dir=/kaggle/working/LiTFiC/data/tsv paths.h2s_feats_dir=/kaggle/temp/feats"
print("env ready:", OVR)
```

### Cell 4 — Download data (I3D features + manifests)
`train.zip` is 7.78 GB → ~15–20 min; extracted to `/kaggle/temp` (session scratch,
avoids the 20 GB /kaggle/working cap; re-download each new session).
```python
import os
os.makedirs("/kaggle/working/LiTFiC/data/tsv", exist_ok=True)
os.makedirs("/kaggle/temp/feats", exist_ok=True)
# Full URLs (a shell $VAR does NOT expand inside a Kaggle ! cell -> empty download)
!wget -O /kaggle/working/LiTFiC/data/tsv/cvpr23.fairseq.i3d.train.how2sign.tab https://dataverse.csuc.cat/api/access/datafile/51923
!wget -O /kaggle/working/LiTFiC/data/tsv/cvpr23.fairseq.i3d.test.how2sign.tab https://dataverse.csuc.cat/api/access/datafile/53222
!wget -q -O /kaggle/temp/train.zip https://dataverse.csuc.cat/api/access/datafile/51543 && unzip -q -o /kaggle/temp/train.zip -d /kaggle/temp/feats/ && rm /kaggle/temp/train.zip
!wget -q -O /kaggle/temp/test.zip https://dataverse.csuc.cat/api/access/datafile/51538 && unzip -q -o /kaggle/temp/test.zip -d /kaggle/temp/feats/ && rm /kaggle/temp/test.zip
!ls -la /kaggle/working/LiTFiC/data/tsv/
!echo "npy count:"; find /kaggle/temp/feats -name '*.npy' | wc -l
```
> Tight on disk? Skip `train.zip`; download `val.zip` (ID 51922, 435 MB) instead and
> also grab the val manifest (ID 51537) as the train set for a smaller smoke test.

### Cell 5 — Verify schema (READ THE OUTPUT)
```python
import csv, glob, os, numpy as np
tab = "/kaggle/working/LiTFiC/data/tsv/cvpr23.fairseq.i3d.train.how2sign.tab"
print("tab size:", os.path.getsize(tab), "bytes")   # 0 bytes => download failed
with open(tab, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    print("COLUMNS:", r.fieldnames)
    row = next(r)
print("id=", row["id"], "| translation=", row["translation"])
print("offset/length=", row["signs_offset"], row["signs_length"])
npy = glob.glob("/kaggle/temp/feats/**/*.npy", recursive=True)[0]
print("NPY", os.path.basename(npy), np.load(npy).shape)
```
**Check:** feature dim (last number) should be **1024**; the npy row count should be
**≥ `signs_length`** (the loader slices `[offset : offset+length]`, already wired).

### Cell 6 — (optional) confirm the prev cue reaches a batch (CPU, no LLM)
```python
from src.data.components.how2sign import How2SignSentences, collate_fn_padd_h2s
ds = How2SignSentences("train", "/kaggle/working/LiTFiC/data/tsv", "/kaggle/temp/feats",
                       use_prev=True, npy_col="signs_file",
                       tsv_name_tmpl="cvpr23.fairseq.i3d.{setname}.how2sign.tab")
b = collate_fn_padd_h2s([ds[0], ds[1], ds[2]])
print("prev examples:", b["rec_prev"])  # non-first sentences should carry a prior sentence
```

### Cell 7 — M1 sanity (no OOM, finite loss)
```python
!python src/train.py experiment=how2sign-vid {OVR} trainer.max_epochs=1 +trainer.limit_train_batches=20 +trainer.limit_val_batches=5 logger=csv
```
> OOM? add `model.net.mm_projector_config.projector_type=conv_K5_P2_KP2_L2` to the command.

### Cell 8 — M1 train (Vid-only, 3 epochs, checkpoints)
```python
!python src/train.py experiment=how2sign-vid {OVR} trainer.max_epochs=3 callbacks.model_checkpoint.every_n_train_steps=500 logger=csv
```

### Cell 9 — M1 eval on a test subset
```python
import glob, os
CKPT = sorted(glob.glob("/kaggle/working/LiTFiC/**/*.ckpt", recursive=True), key=os.path.getmtime)[-1]
print("using", CKPT)
!python src/eval.py experiment=how2sign-vid {OVR} ckpt_path="{CKPT}" +trainer.limit_test_batches=50 logger=csv
```

### Cell 10 — M2 train (Vid+Prev, 3 epochs)
```python
!python src/train.py experiment=how2sign-vid+prev {OVR} trainer.max_epochs=3 callbacks.model_checkpoint.every_n_train_steps=500 logger=csv
```

### Cell 11 — M2 eval + ablation
```python
import glob, os
CKPT2 = sorted(glob.glob("/kaggle/working/LiTFiC/**/*.ckpt", recursive=True), key=os.path.getmtime)[-1]
print("using", CKPT2)
!python src/eval.py experiment=how2sign-vid+prev {OVR} ckpt_path="{CKPT2}" +trainer.limit_test_batches=50 logger=csv
```
Record BLEU-4 / ROUGE-L for **Vid (M1)** vs **Vid+Prev (M2)** on the same 50-batch
subset → that ablation is the demo's point.

---

## Notes / gotchas
- Kaggle session ≤ 12 h, 30 h/week → keep `max_epochs` small (3–5); checkpoints let you resume.
- The LR scheduler is vendored (`src/models/components/lr_scheduler.py`) — no pl_bolts dep.
- If metrics error on Java: `!apt-get -y install default-jre`.
- Qwen2.5-3B downloads from HuggingFace on first run (needs Internet On); no token required.
- Column mapping (`signs_file`/`translation`) + `.tab` filename are already wired in
  `configs/data/how2sign.yaml`; overrides above only redirect the data roots.
```
