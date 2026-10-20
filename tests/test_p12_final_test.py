import pytest

from tools.evaluate_final_dual_path_test import final_test_summary


def _rows():
    rows = []
    for scenario, raw, primary in (
        ("clean:s0:common", 0.90, 0.90),
        ("shift:s1:t2", 0.80, 0.85),
        ("shift:s2:t2", 0.70, 0.80),
        ("shift:s3:t2", 0.60, 0.75),
    ):
        rows.extend(
            [
                {"scenario": scenario, "method": "raw", "f1": raw},
                {
                    "scenario": scenario,
                    "method": "safe_veto_task",
                    "f1": primary,
                },
            ]
        )
    return rows


def test_final_summary_is_reporting_only() -> None:
    summary = final_test_summary(
        _rows(),
        primary_method="safe_veto_task",
    )
    assert summary["reporting_only_no_gate"] is True
    assert summary["clean"]["primary_minus_raw"] == pytest.approx(0.0)
    assert summary["mean_shift_gain"] == pytest.approx(0.10)
    assert summary["test_result_used_for_model_selection"] is False


def test_final_summary_requires_all_scenarios() -> None:
    with pytest.raises(ValueError):
        final_test_summary(
            _rows()[:-2],
            primary_method="safe_veto_task",
        )
