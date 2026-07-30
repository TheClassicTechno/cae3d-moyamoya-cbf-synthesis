#!/usr/bin/env python3
"""
External dataset loader for Week 7–compatible eval (other cerebrovascular diseases).
Reads a JSON that lists (pre, post) NIfTI pairs; returns a Dataset that yields (pre_t, post_t)
in same shape and range as Week7VolumePairs3D. See EXTERNAL_DATASET_CONTRACT.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from week7_preprocess import TARGET_SHAPE, load_volume


def load_external_pairs(json_path: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    Load pairs from external dataset JSON. Returns (pairs, root_dir).
    JSON: {"root": optional, "pairs": [{"pre": path, "post": path}, ...]}.
    Paths are relative to root (or to JSON dir if no root). Missing file → skip with warning.
    """
    with open(json_path) as f:
        data = json.load(f)
    root = data.get("root", "")
    if not root:
        root = str(Path(json_path).parent)
    root = os.path.abspath(root)
    pairs = []
    for item in data.get("pairs", []):
        pre = item.get("pre", "")
        post = item.get("post", "")
        if not pre or not post:
            continue
        pre_path = pre if os.path.isabs(pre) else os.path.join(root, pre)
        post_path = post if os.path.isabs(post) else os.path.join(root, post)
        if not os.path.isfile(pre_path):
            print("Skip (pre missing):", pre_path)
            continue
        if not os.path.isfile(post_path):
            print("Skip (post missing):", post_path)
            continue
        pairs.append((pre_path, post_path))
    return pairs, root


class ExternalWeek7Pairs(Dataset):
    """Dataset of (pre, post) from external JSON; same preprocessing as Week 7 (load_volume, TARGET_SHAPE)."""

    def __init__(
        self,
        json_path: str,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    ):
        self.pairs, _ = load_external_pairs(json_path)
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pre_path, post_path = self.pairs[idx]
        pre = load_volume(pre_path, target_shape=self.target_shape)
        post = load_volume(post_path, target_shape=self.target_shape)
        pre_t = torch.from_numpy(pre).unsqueeze(0).float()
        post_t = torch.from_numpy(post).unsqueeze(0).float()
        return pre_t, post_t
