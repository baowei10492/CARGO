"""Training samples for the P27 localizer (docs/P27_OWN_LOCALIZER_DESIGN.md §4).

Every sample is a 256 px crop of a real bi-temporal pair (T1 = reference,
T2 = target) from the training split of one of several datasets, with:
  * optional synthetic change pasted into T2 (and into the change label),
  * independent photometric jitter and an independent scale + translation
    perturbation for two target copies (P1, P2), so the relative transform
    between them is known exactly whatever the pair's own misregistration,
  * the pasted, unjittered T2 as the reference of a self-pair.
A deterministic hash moves a fixed fraction of tiles to a development
hold-out; validation and test splits are never read here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rs_cd.affine_crops import cut_border_free
from rs_cd.affine_registration import perturb, perturbation_matrix
from rs_cd.datasets.levir import PairedChangeDataset
from rs_cd.robustness import _stable_seed

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Source:
    name: str
    dataset_root: str
    manifest: str
    tile_size: int
    mask_threshold: int
    weight: float = 1.0
    split: str = "train"


@dataclass(frozen=True)
class PerturbationMixture:
    identity: float = 0.3
    small: float = 0.2
    small_translation: float = 3.0
    small_scale: float = 0.01
    large_translation: float = 32.0
    large_scale: tuple[float, float] = (0.75, 1.333)
    rotation: float = 0.0


@dataclass(frozen=True)
class SyntheticChange:
    probability: float = 0.5
    max_fraction: float = 0.4
    max_patches: int = 6
    min_size: int = 16
    max_size: int = 96


@dataclass(frozen=True)
class Photometric:
    brightness: float = 0.15
    contrast: tuple[float, float] = (0.8, 1.2)
    channel_gain: tuple[float, float] = (0.9, 1.1)
    gamma: tuple[float, float] = (0.8, 1.25)
    blur_probability: float = 0.2
    blur_sigma: tuple[float, float] = (0.3, 1.2)
    noise_probability: float = 0.3
    noise_sigma: float = 0.03


def in_holdout(sample_id: str, fraction: float, seed: int) -> bool:
    return _stable_seed(int(seed), str(sample_id), "p27-holdout", 0, "split") / float(2**31) < float(fraction)


def sample_matrix(rng: np.random.Generator, mixture: PerturbationMixture, size: int) -> np.ndarray:
    draw = rng.random()
    if draw < mixture.identity:
        angle, scale, tx, ty = 0.0, 1.0, 0.0, 0.0
    elif draw < mixture.identity + mixture.small:
        tx, ty = rng.uniform(-mixture.small_translation, mixture.small_translation, 2)
        scale = math.exp(rng.uniform(math.log(1 - mixture.small_scale), math.log(1 + mixture.small_scale)))
        angle = 0.0
    else:
        tx, ty = rng.uniform(-mixture.large_translation, mixture.large_translation, 2)
        low, high = mixture.large_scale
        scale = math.exp(rng.uniform(math.log(low), math.log(high)))
        angle = float(rng.uniform(-mixture.rotation, mixture.rotation)) if mixture.rotation else 0.0
    return perturbation_matrix(angle, scale, float(tx), float(ty), width=size, height=size)


def paste_changes(
    target: np.ndarray, donor: np.ndarray, rng: np.random.Generator, settings: SyntheticChange
) -> tuple[np.ndarray, np.ndarray]:
    height, width = target.shape[:2]
    out = target.copy()
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(int(rng.integers(1, settings.max_patches + 1))):
        h, w = (int(v) for v in rng.integers(settings.min_size, settings.max_size + 1, 2))
        h, w = min(h, height), min(w, width)
        if (mask.sum() + h * w) / float(height * width) > settings.max_fraction:
            break
        y, x = int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1))
        dy, dx = int(rng.integers(0, donor.shape[0] - h + 1)), int(rng.integers(0, donor.shape[1] - w + 1))
        out[y : y + h, x : x + w] = donor[dy : dy + h, dx : dx + w]
        mask[y : y + h, x : x + w] = True
    return out, mask


def photometric(image: np.ndarray, rng: np.random.Generator, settings: Photometric) -> np.ndarray:
    unit = image.astype(np.float32) / 255.0
    contrast = rng.uniform(*settings.contrast)
    brightness = rng.uniform(-settings.brightness, settings.brightness)
    gain = rng.uniform(*settings.channel_gain, size=3).astype(np.float32)
    unit = ((unit - 0.5) * contrast + 0.5 + brightness) * gain
    unit = np.clip(unit, 0.0, 1.0) ** rng.uniform(*settings.gamma)
    if rng.random() < settings.blur_probability:
        unit = cv2.GaussianBlur(unit, (0, 0), float(rng.uniform(*settings.blur_sigma)))
    if rng.random() < settings.noise_probability:
        unit = unit + rng.normal(0.0, rng.uniform(0.0, settings.noise_sigma), unit.shape).astype(np.float32)
    return (np.clip(unit, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normalize(image: np.ndarray) -> torch.Tensor:
    unit = image.astype(np.float32) / 255.0
    return torch.from_numpy(((unit - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())


class LocalizerPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        sources: Sequence[Source],
        *,
        samples_per_epoch: int,
        crop_size: int = 256,
        holdout_fraction: float = 0.05,
        seed: int = 3407,
        mixture: PerturbationMixture = PerturbationMixture(),
        synthetic: SyntheticChange = SyntheticChange(),
        jitter: Photometric = Photometric(),
        border_free_margin: int | None = None,
    ) -> None:
        """``border_free_margin`` (P30): targets are cut from the warped crop plus this many
        real pixels on every side, so no target has a zero border."""
        if any(source.split != "train" for source in sources):
            raise ValueError("the localizer trains on training splits only")
        self.sources = list(sources)
        self.crop_size = int(crop_size)
        self.border_free_margin = None if border_free_margin is None else int(border_free_margin)
        if self.border_free_margin is not None:
            small = [s.name for s in self.sources if s.tile_size < self.crop_size + 2 * self.border_free_margin]
            if small:
                raise ValueError(f"tiles of {small} are too small for a {self.border_free_margin} px margin")
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.mixture, self.synthetic, self.jitter = mixture, synthetic, jitter
        self.epoch = 0
        self.datasets = []
        self.rows: list[list[int]] = []
        for source in self.sources:
            dataset = PairedChangeDataset(source.dataset_root, source.manifest, split="train",
                                          normalization="unit", mask_threshold=source.mask_threshold)
            keep = [i for i, row in enumerate(dataset.rows) if not in_holdout(row.sample_id, holdout_fraction, seed)]
            if not keep:
                raise ValueError(f"{source.name}: no training tiles left after the hold-out")
            self.datasets.append(dataset)
            self.rows.append(keep)
        weights = np.array([source.weight for source in self.sources], dtype=np.float64)
        self.source_probabilities = weights / weights.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _random_crop(self, rng: np.random.Generator, source_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.sources[source_index]
        dataset = self.datasets[source_index]
        row = self.rows[source_index][int(rng.integers(len(self.rows[source_index])))]
        t1, t2, mask = dataset._load_arrays(dataset.rows[row])
        if t1.shape[:2] != (source.tile_size, source.tile_size):
            raise ValueError(f"{source.name}: tile is {t1.shape[:2]}, expected {source.tile_size}")
        size = self.crop_size
        top = int(rng.integers(0, source.tile_size - size + 1))
        left = int(rng.integers(0, source.tile_size - size + 1))
        window = (slice(top, top + size), slice(left, left + size))
        return t1[window], t2[window], mask[window] > source.mask_threshold

    def _random_region(self, rng: np.random.Generator, source_index: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(T1 crop, T2 crop, change, T2 region = crop plus the margin on every side)."""
        source = self.sources[source_index]
        dataset = self.datasets[source_index]
        row = self.rows[source_index][int(rng.integers(len(self.rows[source_index])))]
        t1, t2, mask = dataset._load_arrays(dataset.rows[row])
        if t1.shape[:2] != (source.tile_size, source.tile_size):
            raise ValueError(f"{source.name}: tile is {t1.shape[:2]}, expected {source.tile_size}")
        size, margin = self.crop_size, self.border_free_margin
        top = int(rng.integers(margin, source.tile_size - size - margin + 1))
        left = int(rng.integers(margin, source.tile_size - size - margin + 1))
        window = (slice(top, top + size), slice(left, left + size))
        region = t2[top - margin:top + size + margin, left - margin:left + size + margin].copy()
        return t1[window], t2[window], mask[window] > source.mask_threshold, region

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(_stable_seed(self.seed, f"{self.epoch}|{index}", "p27-sample", 0, "draw"))
        source_index = int(rng.choice(len(self.sources), p=self.source_probabilities))
        margin = self.border_free_margin
        if margin is None:
            t1, t2, change = self._random_crop(rng, source_index)
        else:
            t1, t2, change, region = self._random_region(rng, source_index)
        if rng.random() < self.synthetic.probability:
            donor_source = int(rng.choice(len(self.sources), p=self.source_probabilities))
            _, donor, _ = self._random_crop(rng, donor_source)
            t2, pasted = paste_changes(t2, donor, rng, self.synthetic)
            change = change | pasted
        m1 = sample_matrix(rng, self.mixture, self.crop_size)
        m2 = sample_matrix(rng, self.mixture, self.crop_size)
        if margin is None:
            p1 = perturb(photometric(t2, rng, self.jitter), m1)
            p2 = perturb(photometric(t2, rng, self.jitter), m2)
        else:
            size = self.crop_size
            region[margin:margin + size, margin:margin + size] = t2  # the pasted changes move with the crop
            p1 = cut_border_free(photometric(region, rng, self.jitter), m1, margin, size)
            p2 = cut_border_free(photometric(region, rng, self.jitter), m2, margin, size)
        return {
            "reference": normalize(t1),
            "self_reference": normalize(t2),
            "target1": normalize(p1),
            "target2": normalize(p2),
            "matrix1": torch.from_numpy(m1.astype(np.float32)),
            "matrix2": torch.from_numpy(m2.astype(np.float32)),
            "change": torch.from_numpy(change.astype(np.float32))[None],
            "source": source_index,
        }


def holdout_rows(source: Source, fraction: float, seed: int) -> tuple[PairedChangeDataset, list[int]]:
    dataset = PairedChangeDataset(source.dataset_root, source.manifest, split="train",
                                  normalization="unit", mask_threshold=source.mask_threshold)
    rows = [i for i, row in enumerate(dataset.rows) if in_holdout(row.sample_id, fraction, seed)]
    return dataset, rows


def sources_from_config(specs: Sequence[dict[str, Any]]) -> list[Source]:
    return [
        Source(
            name=str(spec["name"]), dataset_root=str(spec["dataset_root"]), manifest=str(spec["manifest"]),
            tile_size=int(spec["tile_size"]), mask_threshold=int(spec["mask_threshold"]),
            weight=float(spec.get("weight", 1.0)), split=str(spec.get("split", "train")),
        )
        for spec in specs
    ]
