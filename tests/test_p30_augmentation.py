"""P30 augmentation baseline: border-free perturbed crops for the detector."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from rs_cd.affine_crops import cut_border_free
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.levir import affine_misregistration_matrix

from p27_helpers import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _dataset(source, **overrides) -> PairedChangeDataset:
    options = dict(split="train", image_size=64, normalization="none", augment=True, random_flip=False,
                   random_rotate90=False, affine_misregistration_probability=1.0, affine_max_translation_px=6.0,
                   affine_scale_range=(0.8, 0.8), affine_context_margin=16)
    options.update(overrides)
    return PairedChangeDataset(source.dataset_root, source.manifest, **options)


def test_perturbed_crop_is_cut_from_real_context(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 2, 128)
    dataset = _dataset(source)
    raw = dataset._load_arrays(dataset.rows[0])
    random.seed(5)
    sample = dataset[0]
    random.seed(5)
    assert random.random() < 1.0  # the perturbation draw
    x0, y0 = random.randint(0, 64), random.randint(0, 64)
    x0, y0 = min(max(x0, 16), 48), min(max(y0, 16), 48)
    level = random.randint(1, 3)
    matrix = affine_misregistration_matrix(level, random.random(), random.random(), random.random(), size=64,
                                           max_translation_px=6.0, scale_range=(0.8, 0.8), levels=3)
    region = raw[1][y0 - 16:y0 + 64 + 16, x0 - 16:x0 + 64 + 16]
    expected = cut_border_free(np.ascontiguousarray(region), matrix, 16, 64)
    t2 = sample["image_t2"].permute(1, 2, 0).numpy().astype(np.uint8)
    assert sample["affine_level"] == level and np.array_equal(t2, expected)
    assert (t2.sum(axis=2) == 0).sum() == 0 and bool(sample["valid_mask"].all())
    t1 = sample["image_t1"].permute(1, 2, 0).numpy().astype(np.uint8)
    assert np.array_equal(t1, raw[0][y0:y0 + 64, x0:x0 + 64])


def test_flips_and_turns_keep_t1_t2_and_mask_together(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 1, 128)
    plain = _dataset(source, affine_misregistration_probability=1e-12)
    turned = _dataset(source, affine_misregistration_probability=1e-12, random_flip=True, random_rotate90=True)
    for seed in range(6):
        random.seed(seed)
        a = plain[0]
        random.seed(seed)
        b = turned[0]
        assert a["affine_level"] == b["affine_level"] == 0
        # Unperturbed: T2 is the crop itself, so T1 - T2 and the mask transform alike.
        for key in ("image_t1", "image_t2", "mask"):
            assert sorted(a[key].flatten().tolist()) == sorted(b[key].flatten().tolist())


def test_margin_needs_room(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 1, 64)
    with pytest.raises(ValueError, match="margin"):
        _dataset(source)[0]
    with pytest.raises(ValueError, match="crop size"):
        _dataset(source, image_size=None)


def test_p30_configs_change_only_the_margin() -> None:
    for name in ("levir", "dsifn", "whu"):
        p29 = yaml.safe_load((ROOT / f"configs/p29/dinov3_aug_{name}.yaml").read_text(encoding="utf-8"))
        p30 = yaml.safe_load((ROOT / f"configs/p30/dinov3_aug_{name}.yaml").read_text(encoding="utf-8"))
        assert p30["data"].pop("affine_context_margin") == 64
        assert p30["experiment"].pop("name") == f"p30_dinov3_aug_{name}"
        p29["experiment"].pop("name")
        assert p30 == p29
