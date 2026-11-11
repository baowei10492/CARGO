import pytest

from tools.evaluate_selective_prediction import (
    expected_calibration_error,
    risk_coverage,
    selective_metrics,
)


def _rows(entries, probability_key="safe_task_probability"):
    rows = []
    for scenario, items in entries.items():
        for index, (probability, advantage, q_gate) in enumerate(items):
            rows.append(
                {
                    "id": f"{scenario}_{index}",
                    "scenario": scenario,
                    "advantage": advantage,
                    "q_gate": q_gate,
                    probability_key: probability,
                }
            )
    return rows


def test_perfect_calibration_has_zero_ece() -> None:
    # Two bins, each with an empirical rate equal to its confidence.
    probabilities = [0.25] * 4 + [0.75] * 4
    labels = [True, False, False, False, True, True, True, False]
    result = expected_calibration_error(probabilities, labels, bins=4)
    assert result["expected_calibration_error"] == pytest.approx(0.0)


def test_overconfident_model_has_positive_ece() -> None:
    probabilities = [0.95] * 10
    labels = [True] * 5 + [False] * 5
    result = expected_calibration_error(probabilities, labels, bins=10)
    assert result["expected_calibration_error"] == pytest.approx(0.45)
    filled = [b for b in result["bins"] if b["samples"]]
    assert len(filled) == 1
    assert filled[0]["empirical_safe_rate"] == pytest.approx(0.5)


def test_ece_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False], bins=4)
    with pytest.raises(ValueError):
        expected_calibration_error([], [], bins=4)


def test_risk_coverage_uses_q_candidates_as_denominator() -> None:
    rows = _rows(
        {
            "shift:s3:t2": [
                (0.9, 0.10, True),
                (0.8, 0.05, True),
                (0.2, -0.05, True),
                # Not a q candidate: the veto never had the option.
                (0.99, -0.90, False),
            ]
        }
    )
    result = risk_coverage(
        rows,
        probability_key="safe_task_probability",
        grid_points=11,
    )
    assert result["candidates"] == 3
    assert result["harmful_candidates"] == 1
    assert result["base_risk_at_full_coverage"] == pytest.approx(1 / 3)


def test_perfect_ranking_matches_the_oracle_area() -> None:
    # Probability order equals benefit order: nothing better is possible.
    rows = _rows(
        {
            "shift:s3:t2": [
                (0.9, 0.30, True),
                (0.7, 0.10, True),
                (0.3, -0.10, True),
                (0.1, -0.30, True),
            ]
        }
    )
    result = risk_coverage(
        rows,
        probability_key="safe_task_probability",
        grid_points=101,
    )
    assert result["excess_aurc"] == pytest.approx(0.0, abs=0.02)


def test_inverted_ranking_is_worse_than_oracle() -> None:
    rows = _rows(
        {
            "shift:s3:t2": [
                (0.1, 0.30, True),
                (0.3, 0.10, True),
                (0.7, -0.10, True),
                (0.9, -0.30, True),
            ]
        }
    )
    result = risk_coverage(
        rows,
        probability_key="safe_task_probability",
        grid_points=101,
    )
    assert result["excess_aurc"] > 0.2


def test_per_scenario_stratification_exposes_pooled_confound() -> None:
    # Within each scenario the probability carries no information, but
    # pooled it separates the scenarios perfectly -- which is exactly the
    # confound the per-scenario view exists to expose.
    rows = _rows(
        {
            "clean:s0:common": [
                (0.2, -0.1, True),
                (0.3, 0.1, True),
                (0.3, -0.1, True),
                (0.2, 0.1, True),
            ],
            "shift:s3:t2": [
                (0.8, -0.1, True),
                (0.9, 0.1, True),
                (0.9, -0.1, True),
                (0.8, 0.1, True),
            ],
        }
    )
    report = selective_metrics(
        rows,
        probability_key="safe_task_probability",
        gate_key="safe_veto_task_gate",
        bins=5,
        grid_points=21,
        deployed_tau=None,
    )
    for entry in report["per_scenario"].values():
        assert entry["auroc"] == pytest.approx(0.5, abs=0.26)
    assert report["reporting_only_no_gate"] is True


def test_deployed_operating_point_is_reported_per_scenario() -> None:
    rows = _rows(
        {
            "clean:s0:common": [
                (0.99, -0.20, True),
                (0.10, -0.10, True),
            ],
            "shift:s3:t2": [
                (0.99, 0.30, True),
                (0.10, 0.05, True),
            ],
        }
    )
    report = selective_metrics(
        rows,
        probability_key="safe_task_probability",
        gate_key="safe_veto_task_gate",
        bins=5,
        grid_points=21,
        deployed_tau=0.95,
    )
    clean = report["per_scenario"]["clean:s0:common"][
        "deployed_operating_point"
    ]
    shift = report["per_scenario"]["shift:s3:t2"][
        "deployed_operating_point"
    ]
    assert clean["accepted"] == 1 and clean["harmful_accepted"] == 1
    assert clean["risk"] == pytest.approx(1.0)
    assert shift["accepted"] == 1 and shift["harmful_accepted"] == 0
    assert report["deployed_tau"] == pytest.approx(0.95)


def test_single_class_scenario_is_flagged_not_crashed() -> None:
    rows = _rows(
        {
            "clean:s0:common": [
                (0.2, -0.1, True),
                (0.3, -0.2, True),
            ],
            "shift:s3:t2": [
                (0.8, 0.1, True),
                (0.9, 0.2, True),
            ],
        }
    )
    report = selective_metrics(
        rows,
        probability_key="safe_task_probability",
        gate_key="safe_veto_task_gate",
        bins=5,
        grid_points=21,
        deployed_tau=None,
    )
    for entry in report["per_scenario"].values():
        assert entry["auroc"] is None
        assert entry["degenerate_reason"] == "single_class_in_scenario"
