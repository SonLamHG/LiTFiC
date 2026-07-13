import csv

import numpy as np
import torch

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
