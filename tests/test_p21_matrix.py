"""The P21 summariser decides what the paper may claim, so pin its verdicts.

The dangerous outputs are not crashes but plausible-looking verdicts: a cell
that passes because displacement never hurt the detector, or a matrix that
reads as uniform success because an inconvenient cell was excluded. Both are
checked here against synthetic cells.
"""
from __future__ import annotations

import pytest
import torch  # noqa: F401  (keeps the suite's import environment consistent)

from tools.summarize_p21_matrix import (
    break_even_fraction,
    metrics_from_scenario_rows,
    summarise_cell,
)

GATES = {
    "raw_shift8_drop_min": 0.05,
    "clean_f1_drop_max": 0.001,
    "shift_mean_gain_min": 0.020,
    "mean_capture_report_only": 0.40,
}


def _cell(
    *,
    raw_clean=0.90,
    raw_shift=(0.86, 0.80, 0.66),
    routed_clean=0.90,
    routed_shift=(0.897, 0.869, 0.819),
    oracle_shift=(0.900, 0.887, 0.876),
    always_clean=0.82,
    always_shift=(0.899, 0.876, 0.873),
):
    keys = ("shift:s1:t2", "shift:s2:t2", "shift:s3:t2")

    def row(clean, shifts):
        out = {"clean:s0:common": clean}
        out.update(dict(zip(keys, shifts)))
        return out

    return {
        "raw": row(raw_clean, raw_shift),
        "always_aligned": row(always_clean, always_shift),
        "q_gate": row(0.885, (0.899, 0.876, 0.852)),
        "safe_veto_registration": row(routed_clean, routed_shift),
        "safe_veto_task": row(routed_clean, routed_shift),
        "loss_oracle": row(raw_clean, oracle_shift),
    }


def test_healthy_cell_passes_both_gates() -> None:
    summary = summarise_cell(
        _cell(), detector="BIT", dataset="LEVIR-CD", gates=GATES
    )
    assert summary["verdict"] == "step_necessary_and_effective"
    assert summary["gates"]["clean_f1_drop"]["passed"]
    assert summary["gates"]["shift_mean_gain"]["passed"]
    assert 0.0 < summary["capture"]["mean_capture_rate"] <= 1.0


def test_robust_detector_is_recorded_not_passed() -> None:
    """A detector displacement barely hurts must not count as a success."""

    summary = summarise_cell(
        _cell(raw_shift=(0.895, 0.890, 0.885), routed_shift=(0.896, 0.891, 0.886),
              oracle_shift=(0.897, 0.893, 0.889)),
        detector="RobustNet",
        dataset="DSIFN-CD",
        gates=GATES,
    )
    assert summary["verdict"] == "no_step_needed"
    assert not summary["preconditions"]["step_needed"]


def test_clean_damage_fails_even_with_a_large_gain() -> None:
    """Buying shifted gain with clean loss is exactly what the design forbids."""

    summary = summarise_cell(
        _cell(routed_clean=0.86),  # 4 pp clean drop
        detector="Greedy",
        dataset="LEVIR-CD",
        gates=GATES,
    )
    assert summary["verdict"] == "effective_but_harms_clean"
    assert not summary["gates"]["clean_f1_drop"]["passed"]


def test_router_that_does_not_capture_fails() -> None:
    summary = summarise_cell(
        _cell(routed_shift=(0.861, 0.801, 0.662)),  # barely above raw
        detector="Flat",
        dataset="GZ-CD",
        gates=GATES,
    )
    assert summary["verdict"] == "step_ineffective"
    assert summary["capture"]["mean_capture_rate"] < 0.1


def test_capture_is_reported_per_scenario() -> None:
    summary = summarise_cell(
        _cell(), detector="BIT", dataset="LEVIR-CD", gates=GATES
    )
    assert set(summary["capture"]["per_scenario"]) == {"2px", "4px", "8px"}
    eight = summary["capture"]["per_scenario"]["8px"]
    assert eight["available_gain"] > eight["captured_gain"] > 0


def test_break_even_matches_the_reported_levir_value() -> None:
    """The frozen numbers must reproduce the p = 0.20 stated in the paper."""

    p = break_even_fraction(0.9007, 0.8617, 0.8176, 0.8829)
    assert abs(p - 0.2033) < 5e-4


def test_break_even_is_none_for_parallel_lines() -> None:
    assert break_even_fraction(0.9, 0.8, 0.9, 0.8) is None


def test_missing_variant_is_an_error_not_a_silent_pass() -> None:
    cell = _cell()
    del cell["loss_oracle"]
    try:
        summarise_cell(cell, detector="X", dataset="Y", gates=GATES)
    except KeyError as error:
        assert "loss_oracle" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a KeyError for the missing oracle")


def test_classifier_csv_pivots_into_the_same_cell() -> None:
    """The server path reads the classifier CSV, not a hand-made JSON."""

    cell = _cell()
    rows = [
        {"scenario": scenario, "method": variant, "f1": repr(value), "samples": "64"}
        for variant, by_scenario in cell.items()
        for scenario, value in by_scenario.items()
    ]
    metrics = metrics_from_scenario_rows(rows)
    assert metrics == cell
    summary = summarise_cell(metrics, detector="BIT", dataset="LEVIR-CD", gates=GATES)
    assert summary["verdict"] == "step_necessary_and_effective"


def test_duplicate_csv_rows_are_an_error() -> None:
    row = {"scenario": "clean:s0:common", "method": "raw", "f1": "0.9"}
    with pytest.raises(ValueError, match="duplicate"):
        metrics_from_scenario_rows([row, dict(row)])
