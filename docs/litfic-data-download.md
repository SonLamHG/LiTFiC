# Getting BOBSL data + where to carve the subset

## 1. License (only you can do this)

The 262 GB features are gated by a **BBC BOBSL Terms-of-Use agreement**:
1. Start the agreement from https://www.robots.ox.ac.uk/~vgg/data/bobsl/ (links to the BBC R&D form).
2. After approval, BBC emails you a **personal password**.
3. Files are then downloadable via password-protected HTTP under
   `https://thor.robots.ox.ac.uk/bobsl/v1.4/...`.

The small **CSLR annotations** (`bobsl.zip`, 1.9 GB → ~15 GB) come separately from the
CSLR2 page (Google Drive): https://gulvarol.github.io/cslr2/data.html

## 2. What to download (for this repo)

| Item | Size | Used as |
|---|---|---|
| SWIN FEATURES V2 → LMDB | **262 GB** | `vid_features_lmdb` |
| Swin-V2 features **pseudo-labels** → LMDB | ~1.1–1.8 GB | `annotations_pkl` |
| Background Captions BLIP2 | ~38 MB | `blip_cap_path` |
| CSLR annotations `bobsl.zip` | 1.9 GB | vocab / info / subtitles / synonyms / splits |
| LiTFiC share (README) | small | `val/test_start_indices.json`, `prev_gt_captions.json` |

## 3. Download to Drive

Use `notebooks/colab_download_bobsl.ipynb`: mount Drive, paste your password + the exact
🔒 URLs, run the download cell. It uses `wget -c` (resume) with `--tries=0`, so a dropped
Colab session is recovered by **re-running the cell** — it continues each partial file.
262 GB over a typical Colab link is roughly 2–6 h; Google Drive's 750 GB/day write quota
covers it in one day.

## 4. ⚠️ Carve on a big-disk VM, not on Drive-FUSE

The subset carve must **read** the 262 GB LMDB with random access (mmap + B-tree seeks).
Over Colab's Drive FUSE mount this is slow and can fail, and Colab has **no 262 GB local
disk** to copy it to first. So:

**Recommended pipeline**
1. Full data lives on **Drive** (durable master copy — done in step 3).
2. Spin a **VM with a ≥300 GB local SSD** (any cloud with a data-science image).
3. Copy the LMDB **from Drive to the VM's local disk** (sequential copy is fast, e.g.
   `rclone copy gdrive:bobsl/<feats_lmdb> /local/ssd/<feats_lmdb>`).
4. Run the carve on local disk:
   ```bash
   python scripts/subset_bobsl_lmdb.py \
     --feats-src /local/ssd/<feats_lmdb> --pl-src /local/ssd/<pl_lmdb> \
     --out-dir /local/ssd/bobsl_subset --train ep1 ep2 --val ep3 --test ep3
   ```
5. Upload the small `bobsl_subset/` (a few hundred MB) to Kaggle as a private Dataset.

**If you have no VM:** you can *try* `notebooks/colab_subset.ipynb` to carve directly from
Drive — it uses range-scans (mostly sequential per episode) so it may work, just slowly.
Treat it as best-effort; if the FUSE mmap on the 262 GB file stalls, fall back to the VM.

## 5. After you have the subset

Continue with `docs/litfic-kaggle-runbook.md` (Stage 1 onward) to run the paper's flow on
Kaggle 2×T4 with Llama-3.2-3B.
