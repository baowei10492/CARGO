from __future__ import annotations

import pytest

from tools.diagnose_p23_screen import (
    CLEAN,
    PHYSICAL,
    classify,
    clean_offsets_versus_reference,
    diagnose,
    residual_spread,
    val_row,
)


def _row(tile, scenario, gt, est, *, converged=True, candidate="roma", trigger=False, change="0.02"):
    return {
        "set": "val_tiles", "id": tile, "candidate": candidate, "scenario": scenario,
        "gt_dx": str(gt[0]), "gt_dy": str(gt[1]), "est_dx": str(est[0]), "est_dy": str(est[1]),
        "converged": str(converged), "trigger": str(trigger), "change_fraction": change,
    }


def test_classify_separates_offsets_from_failures() -> None:
    clean = _row("t", CLEAN, (0, 0), (5, 0))
    assert classify(_row("t", PHYSICAL[0], (12, 0), (13, 1)), clean, 2.0)["kind"] == "tracked"
    explained = classify(_row("t", PHYSICAL[1], (24, 0), (29, 0)), clean, 2.0)
    assert explained["kind"] == "offset_explained"
    assert explained["raw_error"] == pytest.approx(5.0)
    assert explained["diff_error"] == pytest.approx(0.0)
    failed = classify(_row("t", PHYSICAL[2], (48, 0), (10, 3)), clean, 2.0)
    assert failed["kind"] == "tracking_failure"
    lost = classify(_row("t", PHYSICAL[2], (48, 0), (0, 0), converged=False), clean, 2.0)
    assert lost["kind"] == "nonconverged" and lost["diff_error"] is None
    no_reference = classify(
        _row("t", PHYSICAL[2], (48, 0), (40, 0)), _row("t", CLEAN, (0, 0), (0, 0), converged=False), 2.0
    )
    assert no_reference["kind"] == "missed_without_clean_reference"


def test_residual_spread_ignores_nonconverged_rows() -> None:
    rows = [
        _row("t", PHYSICAL[0], (12, 0), (17, 0)),
        _row("t", PHYSICAL[1], (24, 0), (29, 1)),
        _row("t", PHYSICAL[2], (48, 0), (0, 0), converged=False),
    ]
    assert residual_spread(rows) == pytest.approx(1.0)
    assert residual_spread(rows[:1]) is None


def test_val_row_parses_strip_rows_only() -> None:
    assert val_row("whu_val_r14c37") == 14
    assert val_row("whu_p22_block_y00000_x00000") is None


def test_diagnose_counts_kinds_bins_rows_and_stable_tiles() -> None:
    rows = []
    for tile, offset, change in (("whu_val_r02c35", 5, "0.01"), ("whu_val_r15c35", 0, "0.30")):
        rows.append(_row(tile, CLEAN, (0, 0), (offset, 0), change=change))
        for scenario, shift in zip(PHYSICAL, (12, 24, 48)):
            estimate = (shift + offset, 0) if offset else (3, 3)
            rows.append(_row(tile, scenario, (shift, 0), estimate, change=change))
    report = diagnose(rows, "roma")["sets"]["val_tiles"]
    assert report["clean"]["median_offset"] == pytest.approx(2.5)
    s3 = report["scenarios"][PHYSICAL[2]]
    assert s3["raw_pck"] == 0.0 and s3["diff_pck"] == 0.5
    assert s3["kinds"] == {"offset_explained": 1, "tracking_failure": 1}
    assert report["physical_tiles"]["stable_residual_tiles"] == 1
    assert report["change_bins"][0]["kinds"] == {"offset_explained": 3}
    assert report["change_bins"][3]["kinds"] == {"tracking_failure": 3}
    assert report["row_groups"][0]["kinds"] == {"offset_explained": 3}
    assert report["row_groups"][1]["kinds"] == {"tracking_failure": 3}


def test_clean_offsets_compare_only_where_the_reference_fired() -> None:
    rows = [
        _row("a", CLEAN, (0, 0), (0, -12), candidate="p7c", trigger=True),
        _row("a", CLEAN, (0, 0), (1, -11), candidate="roma"),
        _row("b", CLEAN, (0, 0), (12, -12), candidate="p7c", trigger=True),
        _row("b", CLEAN, (0, 0), (0, 0), candidate="roma"),
        _row("c", CLEAN, (0, 0), (0, 12), candidate="p7c", trigger=False),
        _row("c", CLEAN, (0, 0), (0, 12), candidate="roma"),
    ]
    result = clean_offsets_versus_reference(rows, "roma", "p7c")
    assert result["reference_triggered"] == 2
    assert result["candidate_agrees"] == 1 and result["ids"] == ["a"]
