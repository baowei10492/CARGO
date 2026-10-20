import pytest
import torch

from rs_cd.external_diffregcd import (
    ConfusionCounts,
    clean_reproduction_audit,
    nonoverlapping_tiles,
    same_support_decision,
    stable_inference_seed,
    support_digest,
)
from rs_cd.geometry import interior_valid_mask
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec


def test_nonoverlapping_tiles_cover_levir_image() -> None:
    tiles = nonoverlapping_tiles(1024, 1024, 256)
    assert len(tiles) == 16
    assert tiles[0] == (0, 0, 256, 256)
    assert tiles[-1] == (768, 768, 1024, 1024)
    assert sum(
        (bottom - top) * (right - left)
        for top, left, bottom, right in tiles
    ) == 1024 * 1024


def test_nonoverlapping_tiles_reject_partial_edge() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        nonoverlapping_tiles(1000, 1024, 256)


def test_confusion_counts_include_author_compatible_metrics() -> None:
    counts = ConfusionCounts()
    counts.update(
        true_positive=8,
        false_positive=2,
        false_negative=2,
        true_negative=88,
    )
    result = counts.compute()
    assert result["change_f1"] == pytest.approx(0.8)
    assert result["change_iou"] == pytest.approx(8 / 12)
    assert result["background_f1"] == pytest.approx(88 / 90)
    assert result["mean_f1_author_compatibility"] == pytest.approx(
        (0.8 + 88 / 90) / 2
    )
    assert result["overall_accuracy"] == pytest.approx(0.96)
    assert result["valid_pixels"] == 100


def test_support_digest_and_gate_are_ordered_and_strict() -> None:
    digest = support_digest(
        [("a", b"\x01\x00"), ("b", b"\x00\x01")]
    )
    support = {
        "clean:s0:common": {
            "sha256": digest,
            "valid_pixels": 4,
        },
        "shift:s3:t2": {
            "sha256": digest,
            "valid_pixels": 4,
        },
    }
    assert same_support_decision(support)["passed"] is True
    support["shift:s3:t2"]["valid_pixels"] = 3
    assert same_support_decision(support)["passed"] is False


def test_common_random_seed_does_not_depend_on_scenario() -> None:
    first = stable_inference_seed(3407, "val_1", 0)
    second = stable_inference_seed(3407, "val_1", 0)
    other_tile = stable_inference_seed(3407, "val_1", 2)
    assert first == second
    assert first != other_tile


def test_fixed_border_makes_all_p13_shift_support_identical() -> None:
    class SinglePair(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            assert index == 0
            return {
                "image_t1": torch.zeros(3, 32, 32),
                "image_t2": torch.zeros(3, 32, 32),
                "mask": torch.zeros(1, 32, 32),
                "id": "sample",
            }

    rows = {}
    specs = [
        CorruptionSpec("clean", 0, "common", seed=3407),
        *[
            CorruptionSpec("shift", severity, "t2", seed=3407)
            for severity in (1, 2, 3)
        ],
    ]
    for spec in specs:
        sample = CorruptedPairDataset(
            SinglePair(),
            spec,
            normalization="unit",
        )[0]
        valid = sample["valid_mask"] * interior_valid_mask(
            1,
            32,
            32,
            8,
        ).squeeze(0)
        packed = (valid >= 0.5).numpy().astype("uint8").tobytes()
        rows[spec.key] = {
            "sha256": support_digest([("sample", packed)]),
            "valid_pixels": int(valid.sum()),
        }
    assert same_support_decision(rows)["passed"] is True


def test_clean_reproduction_is_reporting_only() -> None:
    rows = [
        {
            "scenario": "clean:s0:common",
            "method": method,
            "change_f1": 0.87,
            "mean_f1_author_compatibility": 0.92,
            "mean_iou_author_compatibility": 0.88,
            "overall_accuracy": 0.987,
        }
        for method in ("official_unmodified", "intended_warped")
    ]
    result = clean_reproduction_audit(
        rows,
        {
            "change_f1": 0.872,
            "mean_f1": 0.929,
            "mean_iou": 0.881,
            "overall_accuracy": 0.987,
            "evaluation_distribution_note": "different protocol",
        },
    )
    assert result["status"] == "manual_review_required"
    assert result["used_for_path_selection"] is False
    assert result["paths"]["official_unmodified"]["change_f1"][
        "delta"
    ] == pytest.approx(-0.002)
