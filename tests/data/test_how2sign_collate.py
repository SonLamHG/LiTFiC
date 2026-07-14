import csv

import numpy as np

from src.data.components.how2sign import How2SignSentences, collate_fn_padd_h2s


def _ds(tmp_path):
    feats_dir = tmp_path / "feats"; feats_dir.mkdir()
    tsv_dir = tmp_path / "tsv"; tsv_dir.mkdir()
    rows = [
        {"id": "vidA_0-5-rgb_front", "translation": "first", "npy_path": "a0.npy"},
        {"id": "vidA_1-5-rgb_front", "translation": "second", "npy_path": "a1.npy"},
    ]
    np.save(feats_dir / "a0.npy", np.zeros((5, 1024), dtype=np.float32))   # < MIN_VIDEO_FRAMES -> padded to 8
    np.save(feats_dir / "a1.npy", np.zeros((10, 1024), dtype=np.float32))  # batch max
    with open(tsv_dir / "how2sign_train.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "translation", "npy_path"], delimiter="\t")
        w.writeheader()
        [w.writerow(r) for r in rows]
    return How2SignSentences("train", str(tsv_dir), str(feats_dir), use_prev=True)


def test_collate_shapes_and_rec_prev(tmp_path):
    ds = _ds(tmp_path)
    batch = collate_fn_padd_h2s([ds[0], ds[1]])
    # a0 (5 frames) is min-frame-padded to 8, then both are batch-padded to max T=10
    assert batch["features"].shape == (2, 10, 1024)
    assert batch["attn_masks"].shape == (2, 10)
    assert batch["questions"][0] is not None
    # first sentence has no prev, second has prev "first"
    prevs = {i: p for i, p in zip(batch["ids"], batch["rec_prev"])}
    assert prevs["vidA_0-5-rgb_front"] == []
    assert prevs["vidA_1-5-rgb_front"] == ["first"]
    assert batch["pls"] == [None, None]
