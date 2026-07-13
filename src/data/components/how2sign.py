"""How2Sign sentence dataset: reads per-sentence I3D .npy + .tsv metadata
and emits the item dict consumed by the collate/model contract. Bypasses all
BOBSL machinery (Subtitles/LMDB/pseudo-labels/spottings) so the demo path stays
self-contained and CPU-testable."""
import os
import re
import csv
import random
from collections import defaultdict
from typing import List, Tuple

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
    '--7E2sU6zP4_10-5-rgb_front' -> ('--7E2sU6zP4', 10)."""
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

    def _read_tsv(self, path: str) -> List[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def _build_index(self, rows: List[dict]) -> List[dict]:
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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
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


def _pad_features(tensor_list: List[torch.Tensor]):
    """Right-pad a list of [T_i, C] feature tensors to [B, T_max, C] and build
    the matching [B, T_max] attention mask. Local, self-contained helper so the
    How2Sign path does not import BOBSL's sentence/lmdb modules."""
    max_len = max(t.size(0) for t in tensor_list)
    padded, masks = [], []
    for t in tensor_list:
        pad_len = max_len - t.size(0)
        padded.append(torch.nn.functional.pad(t, (0, 0, 0, pad_len), "constant", 0))
        masks.append(torch.cat([torch.ones(t.size(0)), torch.zeros(pad_len)]))
    return torch.stack(padded), torch.stack(masks)


def collate_fn_padd_h2s(batch: List[dict]) -> dict:
    """Collate How2Sign items into the batch dict expected by VggSLTNet.forward.
    Adds a ``rec_prev`` key (list of ``[prev]`` / ``[]`` per item) so the GT
    previous sentence reaches LanguageDecoder._process_predict at inference."""
    features = [item["features"] for item in batch]
    padded_features, attn_masks = _pad_features(features)
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
