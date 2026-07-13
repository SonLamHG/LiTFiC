import csv
import json

import numpy as np

from src.data.slt_datamodule import SLTDataModule


def _prep(tmp_path):
    feats_dir = tmp_path / "feats"; feats_dir.mkdir()
    tsv_dir = tmp_path / "tsv"; tsv_dir.mkdir()
    for split in ["train", "test"]:
        rows = [
            {"id": f"vidA_{i}-5-rgb_front", "translation": f"s{i}", "npy_path": f"{split}{i}.npy"}
            for i in range(2)
        ]
        for i in range(2):
            np.save(feats_dir / f"{split}{i}.npy", np.zeros((4, 1024), dtype=np.float32))
        with open(tsv_dir / f"how2sign_{split}.tsv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "translation", "npy_path"], delimiter="\t")
            w.writeheader()
            [w.writerow(r) for r in rows]
    idx = tmp_path / "idx.json"; idx.write_text(json.dumps({"idx": [0]}))
    return str(tsv_dir), str(feats_dir), str(idx)


def test_datamodule_builds_how2sign(tmp_path):
    tsv_dir, feats_dir, idx = _prep(tmp_path)
    dm = SLTDataModule(
        dataset="how2sign",
        val_episode_ind_path=idx,
        test_episode_ind_path=idx,
        dataset_config={"tsv_dir": tsv_dir, "feats_dir": feats_dir, "feats_dim": 1024, "use_prev": True},
        batch_size=2,
        num_workers=0,
        collate_fn="src.data.components.how2sign.collate_fn_padd_h2s",
        test_setname="test",
    )
    dm.setup()
    assert len(dm.data_train) == 2
    assert len(dm.data_test) == 2
