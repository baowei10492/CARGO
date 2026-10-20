from __future__ import annotations

import pytest

from rs_cd.external_diffregcd import (
    external_test_summary,
    official_path_freeze_decision,
)
from tools.summarize_p13_external_comparison import comparison_summary


def _validation_result(delta: float = 0.001) -> dict:
    metrics = (
        "change_f1",
        "mean_f1_author_compatibility",
        "mean_iou_author_compatibility",
        "overall_accuracy",
    )
    return {
        "decision": {"engineering_gates_passed": True},
        "support_gate": {"passed": True},
        "clean_reproduction_audit": {
            "paths": {
                "official_unmodified": {
                    metric: {"delta": delta} for metric in metrics
                }
            }
        },
    }


def test_official_path_freeze_uses_clean_only() -> None:
    result = _validation_result()
    result["scenario_results"] = [
        {
            "scenario": "shift:s3:t2",
            "method": "official_unmodified",
            "change_f1": 0.0,
        }
    ]
    decision = official_path_freeze_decision(result)
    assert decision["status"] == "passed"
    assert decision["selected_external_path"] == "official_unmodified"
    assert decision["shift_metrics_used_for_path_selection"] is False


def test_official_path_freeze_rejects_bad_clean_reproduction() -> None:
    decision = official_path_freeze_decision(
        _validation_result(delta=0.02)
    )
    assert decision["status"] == "failed"
    assert decision["test_authorized"] is False


def _external_rows() -> list[dict]:
    return [
        {
            "scenario": scenario,
            "method": "official_unmodified",
            "change_f1": f1,
            "tp": 10,
            "fp": 2,
            "fn": 2,
            "tn": 86,
        }
        for scenario, f1 in (
            ("clean:s0:common", 0.87),
            ("shift:s1:t2", 0.80),
            ("shift:s2:t2", 0.70),
            ("shift:s3:t2", 0.60),
        )
    ]


def _project_rows(valid_pixels: int = 100) -> list[dict]:
    return [
        {
            "scenario": scenario,
            "method": "safe_veto_task",
            "f1": f1,
            "tp": 10,
            "fp": 2,
            "fn": 2,
            "tn": valid_pixels - 14,
        }
        for scenario, f1 in (
            ("clean:s0:common", 0.90),
            ("shift:s1:t2", 0.85),
            ("shift:s2:t2", 0.80),
            ("shift:s3:t2", 0.75),
        )
    ]


def test_external_test_summary_is_reporting_only() -> None:
    summary = external_test_summary(_external_rows())
    assert summary["reporting_only_no_selection"] is True
    assert summary["scenario_change_f1"]["shift:s3:t2"] == pytest.approx(
        0.60
    )
    assert summary["test_result_used_for_model_selection"] is False


def test_same_support_comparison() -> None:
    summary = comparison_summary(
        _project_rows(),
        _external_rows(),
        project_method="safe_veto_task",
        external_method="official_unmodified",
    )
    assert summary["status"] == "completed"
    assert summary["same_valid_pixel_count_all_scenarios"] is True
    assert summary["mean_project_minus_external_shift_f1"] == (
        pytest.approx(0.10)
    )


def test_comparison_rejects_different_support() -> None:
    summary = comparison_summary(
        _project_rows(valid_pixels=99),
        _external_rows(),
        project_method="safe_veto_task",
        external_method="official_unmodified",
    )
    assert summary["status"] == "failed"
    assert summary["same_valid_pixel_count_all_scenarios"] is False

