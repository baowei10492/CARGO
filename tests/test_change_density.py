"""Preregistered change-density diagnostics (docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md).

Selection and verdict code decide what the two follow-ups can conclude, so
the branches are pinned before any result exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from rs_cd.change_density import (
    assign_bin,
    association_verdict,
    average_ranks,
    block_change_counts,
    overall_association,
    probe_bin_stats,
    probe_decision,
    select_stratified,
    spearman,
    window_change_fractions,
)

A1 = {
    "min_windows_per_bin": 8,
    "works_pck1_min": 0.90,
    "works_clean_q_max": 0.10,
    "fails_pck1_max": 0.50,
    "fails_clean_q_min": 0.50,
}
A2 = {"rho_min": 0.30, "rho_null_max": 0.10, "min_samples_per_bin": 10, "high_over_low_min": 2.0}


def test_bins_are_half_open_and_the_last_includes_one() -> None:
    assert assign_bin(0.0) == 0
    assert assign_bin(0.05) == 1
    assert assign_bin(0.1499) == 2
    assert assign_bin(0.15) == 3
    assert assign_bin(1.0) == 3


def test_average_ranks_share_ties() -> None:
    np.testing.assert_allclose(average_ranks([0.0, 0.0, 5.0, 1.0]), [1.5, 1.5, 4.0, 3.0])


def test_spearman_is_one_for_monotone_data_and_none_when_constant() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 25, 90]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [0, 0, 0, 0]) is None
    assert spearman([1, 2], [1, 2]) is None


def test_block_counts_and_window_fractions_match_a_direct_count() -> None:
    label = np.zeros((40, 50), dtype=np.uint8)
    label[0:10, 0:10] = 255   # one full block
    label[10:15, 10:20] = 255  # half of another block
    counts = block_change_counts(label, block=10, columns=5, rows=4, threshold=127)
    assert counts[0, 0] == 100 and counts[1, 1] == 50 and counts.sum() == 150
    windows = window_change_fractions(counts, blocks_per_window=2, block=10)
    assert len(windows) == 3 * 4
    first = next(w for w in windows if w["left"] == 0 and w["top"] == 0)
    assert first["change_fraction"] == pytest.approx(150 / 400)


def test_stratified_selection_respects_spacing_cap_and_bins() -> None:
    windows = [
        {"left": x, "top": y, "change_fraction": 0.2 if x >= 1536 else 0.01}
        for y in range(0, 4096, 512)
        for x in range(0, 4096, 512)
    ]
    chosen = select_stratified(windows, per_bin=3, min_spacing=1536)
    for index in {c["bin"] for c in chosen}:
        members = [c for c in chosen if c["bin"] == index]
        assert len(members) <= 3
        for a in members:
            for b in members:
                if a is not b:
                    assert max(abs(a["left"] - b["left"]), abs(a["top"] - b["top"])) >= 1536
    assert chosen == select_stratified(windows, per_bin=3, min_spacing=1536)


def _windows(bin_index: int, n: int, *, s3_hits: int, clean_q: int) -> list[dict]:
    return [
        {"bin": bin_index, "s2_hit": True, "s3_hit": i < s3_hits, "clean_q": i < clean_q,
         "s2_error": 0.0, "s3_error": 0.0 if i < s3_hits else 40.0}
        for i in range(n)
    ]


def test_probe_reproduced_when_high_change_windows_fail() -> None:
    windows = _windows(0, 10, s3_hits=10, clean_q=0) + _windows(3, 10, s3_hits=2, clean_q=9)
    decision = probe_decision(probe_bin_stats(windows), A1)
    assert decision["verdict"] == "reproduced_outside_val" and not decision["merged_high_bin"]


def test_probe_not_reproduced_when_high_change_windows_work() -> None:
    windows = _windows(0, 10, s3_hits=10, clean_q=0) + _windows(3, 10, s3_hits=10, clean_q=1)
    assert probe_decision(probe_bin_stats(windows), A1)["verdict"] == "not_reproduced"


def test_probe_partial_between_the_two_thresholds() -> None:
    windows = _windows(0, 10, s3_hits=10, clean_q=0) + _windows(3, 10, s3_hits=7, clean_q=3)
    assert probe_decision(probe_bin_stats(windows), A1)["verdict"] == "partial"


def test_probe_merges_sparse_high_bins_and_reports_insufficiency() -> None:
    merged = _windows(0, 10, s3_hits=10, clean_q=0) + _windows(2, 5, s3_hits=1, clean_q=5) + _windows(3, 4, s3_hits=0, clean_q=4)
    decision = probe_decision(probe_bin_stats(merged), A1)
    assert decision["merged_high_bin"] and decision["verdict"] == "reproduced_outside_val"
    sparse = _windows(0, 10, s3_hits=10, clean_q=0) + _windows(2, 2, s3_hits=0, clean_q=2) + _windows(3, 3, s3_hits=0, clean_q=3)
    assert probe_decision(probe_bin_stats(sparse), A1)["verdict"] == "inconclusive_insufficient_high_change_windows"


def test_probe_requires_the_instrument_to_work_on_low_change_windows() -> None:
    windows = _windows(0, 10, s3_hits=6, clean_q=0) + _windows(3, 10, s3_hits=0, clean_q=10)
    assert probe_decision(probe_bin_stats(windows), A1)["verdict"] == "inconclusive_low_bin_failed"


def test_association_branches() -> None:
    change = [0.01] * 20 + [0.3] * 20
    associated = association_verdict(change, [0.0] * 20 + [8.0] * 20, thresholds=A2)
    assert associated["verdict"] == "associated" and associated["spearman"] > 0.8
    unrelated = association_verdict(change, [4.0, 0.0] * 20, thresholds=A2)
    assert unrelated["verdict"] == "not_associated"
    few_high = association_verdict([0.01] * 30 + [0.3] * 5, [0.0] * 30 + [8.0] * 5, thresholds=A2)
    assert few_high["verdict"] == "weak_or_inconclusive"


def test_overall_association_needs_both_datasets() -> None:
    assert overall_association("associated", "associated") == "cross_dataset_supported"
    assert overall_association("not_associated", "not_associated") == "cross_dataset_not_supported"
    assert overall_association("associated", "weak_or_inconclusive") == "mixed"
