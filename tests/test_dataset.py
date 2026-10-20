from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.manifest import build_manifest, read_manifest


def _write_sample(root: Path, split: str, sample_id: str, size=(40, 32)) -> None:
    for folder in ("A", "B", "label"):
        (root / split / folder).mkdir(parents=True, exist_ok=True)
    t1 = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    t2 = t1.copy()
    t2[5:15, 7:19] = 255
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[5:15, 7:19] = 255
    Image.fromarray(t1).save(root / split / "A" / f"{sample_id}.png")
    Image.fromarray(t2).save(root / split / "B" / f"{sample_id}.png")
    Image.fromarray(mask).save(root / split / "label" / f"{sample_id}.png")


def test_manifest_and_dataset_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"{split}_001")
    manifest = tmp_path / "manifest.csv"
    rows = build_manifest(root, manifest)
    assert len(rows) == 3
    assert len(read_manifest(manifest, "train")) == 1
    parent_manifest = tmp_path / "parent_manifest.csv"
    parent_rows = build_manifest(
        root, parent_manifest, parent_id_mode="sample_id"
    )
    assert parent_rows[0].parent_id == parent_rows[0].sample_id

    dataset = PairedChangeDataset(
        root,
        manifest,
        split="train",
        image_size=24,
        normalization="unit",
        augment=False,
    )
    sample = dataset[0]
    assert sample["image_t1"].shape == (3, 24, 24)
    assert sample["image_t2"].shape == (3, 24, 24)
    assert sample["mask"].shape == (1, 24, 24)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["id"] == "train_001"

    cached_dataset = PairedChangeDataset(
        root,
        manifest,
        split="train",
        image_size=24,
        normalization="unit",
        augment=False,
        cache_mode="ram",
    )
    cached_sample = cached_dataset[0]
    torch.testing.assert_close(
        cached_sample["image_t1"], sample["image_t1"]
    )
    torch.testing.assert_close(
        cached_sample["image_t2"], sample["image_t2"]
    )
    torch.testing.assert_close(cached_sample["mask"], sample["mask"])


def test_manifest_rejects_ids_that_repeat_across_splits(
    tmp_path: Path,
) -> None:
    # DSIFN-CD restarts filenames inside every split, so bare stems make
    # one sample_id denote three different image pairs.
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, "1")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_manifest(root, tmp_path / "manifest.csv")


def test_split_stem_ids_are_unique_and_carry_the_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, "1")
    rows = build_manifest(
        root,
        tmp_path / "manifest.csv",
        sample_id_mode="split_stem",
        parent_id_mode="sample_id",
    )
    ids = sorted(row.sample_id for row in rows)
    assert ids == ["test_1", "train_1", "val_1"]
    assert all(row.parent_id == row.sample_id for row in rows)
    assert all(
        row.t1_path.startswith(f"{row.split}/") for row in rows
    )


def test_manifest_rejects_unknown_sample_id_mode(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_sample(root, "train", "1")
    with pytest.raises(ValueError, match="sample_id_mode"):
        build_manifest(
            root,
            tmp_path / "manifest.csv",
            splits=("train",),
            sample_id_mode="uuid",
        )


def test_manifest_rejects_unpaired_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"{split}_001")
    (root / "train" / "B" / "train_001.png").unlink()
    with pytest.raises(ValueError, match="Unpaired files"):
        build_manifest(root, tmp_path / "manifest.csv")


def test_dataset_uses_explicit_mask_threshold(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"{split}_001", size=(16, 16))
    mask_path = root / "train" / "label" / "train_001.png"
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0, 0] = 100
    mask[0, 1] = 156
    Image.fromarray(mask).save(mask_path)
    manifest = tmp_path / "manifest.csv"
    build_manifest(root, manifest)
    dataset = PairedChangeDataset(
        root,
        manifest,
        split="train",
        image_size=None,
        normalization="unit",
        mask_threshold=127,
    )
    loaded = dataset[0]["mask"]
    assert float(loaded[0, 0, 0]) == 0.0
    assert float(loaded[0, 0, 1]) == 1.0


def test_positive_aware_crop_contains_change(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"{split}_001", size=(64, 64))
    manifest = tmp_path / "manifest.csv"
    build_manifest(root, manifest)
    dataset = PairedChangeDataset(
        root,
        manifest,
        split="train",
        image_size=16,
        normalization="unit",
        augment=True,
        random_flip=False,
        random_rotate90=False,
        positive_crop_probability=1.0,
        positive_crop_min_pixels=1,
    )
    for _ in range(10):
        assert int(dataset[0]["mask"].sum()) >= 1


def test_training_misregistration_shifts_only_requested_temporal_image(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"{split}_001", size=(64, 64))
    manifest = tmp_path / "manifest.csv"
    build_manifest(root, manifest)
    clean = PairedChangeDataset(
        root,
        manifest,
        split="train",
        normalization="unit",
        augment=False,
    )[0]
    shifted = PairedChangeDataset(
        root,
        manifest,
        split="train",
        normalization="unit",
        augment=True,
        random_flip=False,
        random_rotate90=False,
        misregistration_probability=1.0,
        misregistration_pixels=(4,),
        misregistration_target="t2",
    )[0]
    torch.testing.assert_close(shifted["image_t1"], clean["image_t1"])
    torch.testing.assert_close(shifted["mask"], clean["mask"])
    assert not torch.equal(shifted["image_t2"], clean["image_t2"])
    assert shifted["misregistration_target"] == "t2"
    assert max(
        abs(int(shifted["misregistration_dx"])),
        abs(int(shifted["misregistration_dy"])),
    ) == 4
