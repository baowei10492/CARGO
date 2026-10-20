from pathlib import Path

import torch

from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.manifest import build_manifest
from rs_cd.robustness import (
    CorruptedPairDataset,
    CorruptionSpec,
    OracleInverseShiftDataset,
    PairedShiftSupportDataset,
)
from tests.test_dataset import _write_sample


def _base_dataset(tmp_path: Path) -> PairedChangeDataset:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        _write_sample(root, split, f"{split}_001", size=(64, 64))
    manifest = tmp_path / "manifest.csv"
    build_manifest(root, manifest)
    return PairedChangeDataset(
        root,
        manifest,
        split="val",
        image_size=None,
        normalization="imagenet",
        augment=False,
    )


def test_corruptions_are_deterministic(tmp_path: Path) -> None:
    base = _base_dataset(tmp_path)
    for name in ("brightness", "gaussian_blur", "gaussian_noise", "shift"):
        wrapped = CorruptedPairDataset(
            base,
            CorruptionSpec(name=name, severity=2, mode="t2", seed=3407),
        )
        first = wrapped[0]
        second = wrapped[0]
        torch.testing.assert_close(first["image_t1"], second["image_t1"])
        torch.testing.assert_close(first["image_t2"], second["image_t2"])
        torch.testing.assert_close(first["mask"], second["mask"])
        torch.testing.assert_close(first["valid_mask"], second["valid_mask"])


def test_one_sided_shift_keeps_mask_and_marks_invalid_border(
    tmp_path: Path,
) -> None:
    base = _base_dataset(tmp_path)
    original = base[0]
    wrapped = CorruptedPairDataset(
        base,
        CorruptionSpec(name="shift", severity=3, mode="t2", seed=3407),
    )
    sample = wrapped[0]
    torch.testing.assert_close(sample["image_t1"], original["image_t1"])
    torch.testing.assert_close(sample["mask"], original["mask"])
    assert float(sample["valid_mask"].min()) == 0.0
    assert float(sample["valid_mask"].max()) == 1.0
    assert not torch.equal(sample["image_t2"], original["image_t2"])


def test_common_shift_moves_mask_with_both_images(tmp_path: Path) -> None:
    base = _base_dataset(tmp_path)
    original = base[0]
    wrapped = CorruptedPairDataset(
        base,
        CorruptionSpec(name="shift", severity=2, mode="common", seed=3407),
    )
    sample = wrapped[0]
    assert not torch.equal(sample["image_t2"], original["image_t2"])
    assert not torch.equal(sample["mask"], original["mask"])
    assert float(sample["valid_mask"].min()) == 0.0
    assert sample["shift_dx"] == 0.0
    assert sample["shift_dy"] == 0.0


def test_t1_shift_records_relative_offset_with_reversed_sign(
    tmp_path: Path,
) -> None:
    base = _base_dataset(tmp_path)
    t1 = CorruptedPairDataset(
        base,
        CorruptionSpec(name="shift", severity=2, mode="t1", seed=3407),
    )[0]
    t2 = CorruptedPairDataset(
        base,
        CorruptionSpec(name="shift", severity=2, mode="t2", seed=3407),
    )[0]
    assert t1["shift_dx"] == -t2["shift_dx"]
    assert t1["shift_dy"] == -t2["shift_dy"]


def test_independent_shift_records_difference_between_branches(
    tmp_path: Path,
) -> None:
    base = _base_dataset(tmp_path)
    independent = CorruptedPairDataset(
        base,
        CorruptionSpec(
            name="shift",
            severity=2,
            mode="independent",
            seed=3407,
        ),
    )
    sample = independent[0]
    expected_t1 = independent._applied_translation("val_001", "t1")
    expected_t2 = independent._applied_translation("val_001", "t2")
    assert sample["shift_dx"] == expected_t2[0] - expected_t1[0]
    assert sample["shift_dy"] == expected_t2[1] - expected_t1[1]


def test_oracle_inverse_recovers_shifted_t2_in_valid_interior(
    tmp_path: Path,
) -> None:
    base = _base_dataset(tmp_path)
    shifted = CorruptedPairDataset(
        base,
        CorruptionSpec(name="shift", severity=2, mode="t2", seed=3407),
    )
    recovered = OracleInverseShiftDataset(shifted, target="t2")[0]
    clean = base[0]
    valid = recovered["valid_mask"].bool().expand_as(clean["image_t2"])
    torch.testing.assert_close(
        recovered["image_t2"][valid],
        clean["image_t2"][valid],
    )
    torch.testing.assert_close(recovered["mask"], clean["mask"])


def test_raw_and_oracle_can_use_identical_paired_shift_support(
    tmp_path: Path,
) -> None:
    base = _base_dataset(tmp_path)
    shifted = CorruptedPairDataset(
        base,
        CorruptionSpec(name="shift", severity=3, mode="t2", seed=3407),
    )
    ordinary_raw = shifted[0]
    paired_raw = PairedShiftSupportDataset(shifted)[0]
    oracle = OracleInverseShiftDataset(shifted, target="t2")[0]
    torch.testing.assert_close(
        paired_raw["valid_mask"],
        oracle["valid_mask"],
    )
    assert paired_raw["paired_shift_support"] is True
    assert (
        int(paired_raw["valid_mask"].sum())
        <= int(ordinary_raw["valid_mask"].sum())
    )
