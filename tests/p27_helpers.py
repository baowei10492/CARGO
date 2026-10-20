"""Shared fixtures for the P27 tests: synthetic tiles and a tiny backbone."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from rs_cd.datasets.localizer_pairs import Source


def tile(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return cv2.resize((rng.random((size // 16, size // 16, 3)) * 255).astype(np.uint8), (size, size),
                      interpolation=cv2.INTER_CUBIC)


def make_dataset(root: Path, name: str, tiles: int, size: int, *, split: str = "train",
                 splits: Sequence[str] | None = None) -> Source:
    """``splits`` gives each tile its own split; otherwise every tile is ``split``."""

    folder = root / name
    for sub in ("A", "B", "label"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    splits = list(splits) if splits is not None else [split] * tiles
    rows = []
    for index in range(tiles):
        t1 = tile(size, index)
        t2 = np.roll(t1, 1, axis=1)
        mask = np.zeros((size, size), np.uint8)
        mask[10:60, 10:60] = 255
        stem = f"{name}_{index}"
        for sub, array in (("A", t1), ("B", t2), ("label", mask)):
            Image.fromarray(array).save(folder / sub / f"{stem}.png")
        rows.append({"sample_id": stem, "split": splits[index], "t1_path": f"A/{stem}.png",
                     "t2_path": f"B/{stem}.png", "mask_path": f"label/{stem}.png", "parent_id": ""})
    manifest = folder / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return Source(name=name, dataset_root=str(folder), manifest=str(manifest), tile_size=size, mask_threshold=127,
                  split=split)


class TinyBackbone(nn.Module):
    """Translation-equivariant stand-in with DINOv3's stride-4 / stride-8 layout."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.stem = nn.Conv2d(3, 12, kernel_size=4, stride=4)
        self.down = nn.Conv2d(12, 24, kernel_size=2, stride=2)
        self.requires_grad_(False)

    def forward(self, images):
        stride4 = torch.tanh(self.stem(images))
        return stride4, torch.tanh(self.down(stride4))
