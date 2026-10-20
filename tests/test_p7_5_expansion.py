import csv

import pytest
import torch

from rs_cd.dual_path import (
    ROUTER_FEATURE_NAMES,
    SAFE_REGISTRATION_FEATURE_NAMES,
    SAFE_TASK_FEATURE_NAMES,
    TILE_CONSISTENCY_FEATURE_NAMES,
    tile_consistency_features,
)
from tools.screen_tile_consistency_features import tile_features_by_key
from tools.train_evaluate_safe_alignment_classifier import (
    _decision,
    _feature_universe,
    _join_tile_features,
    _tile_feature_config,
)


def test_reduction_matches_screening_implementation() -> None:
    # The screening AUROCs were measured on tile_features_by_key; the
    # deployment-side reduction must agree exactly on shared inputs.
    records = []
    tiles = [(2.0, 0.0), (-4.0, 2.0), (0.0, 0.0), (6.0, -2.0)]
    for index, (dx, dy) in enumerate(tiles):
        records.append(
            {
                "id": "img",
                "scenario": "clean:s0:common",
                "tile_index": index,
                "est_dx": dx,
                "est_dy": dy,
            }
        )
    records.append(
        {
            "id": "img",
            "scenario": "clean:s0:common",
            "tile_index": -1,  # whole-image estimate
            "est_dx": 1.0,
            "est_dy": 1.0,
        }
    )
    screening = tile_features_by_key(records)[("img", "clean:s0:common")]
    deployment = tile_consistency_features(tiles, (1.0, 1.0))
    for name in TILE_CONSISTENCY_FEATURE_NAMES:
        assert deployment[name] == pytest.approx(screening[name])


def test_reduction_falls_back_to_centre_without_whole() -> None:
    tiles = [(2.0, 0.0), (0.0, 2.0)]
    reduced = tile_consistency_features(tiles, None)
    assert reduced["tile_max_pairwise_distance"] == pytest.approx(
        8.0**0.5
    )
    with pytest.raises(ValueError):
        tile_consistency_features([(0.0, 0.0)], None)


def test_feature_universe_legacy_path_is_unchanged() -> None:
    assert _tile_feature_config({}) is None
    assert _tile_feature_config({"tile_features": {"enabled": False}}) is None
    assert _feature_universe(None) == ROUTER_FEATURE_NAMES


def test_feature_universe_appends_canonical_order() -> None:
    config = _tile_feature_config(
        {
            "tile_features": {
                "enabled": True,
                "tile_size": 512,
                # Deliberately scrambled order in the yaml.
                "names": [
                    "tile_agreement_within_4px",
                    "tile_max_pairwise_distance",
                    "tile_dispersion_radius",
                    "tile_agreement_within_2px",
                ],
            }
        }
    )
    assert config["names"] == TILE_CONSISTENCY_FEATURE_NAMES
    universe = _feature_universe(config)
    # The prefix must stay ROUTER_FEATURE_NAMES so the frozen P7-2 task
    # router bundle keeps indexing valid columns.
    assert universe[: len(ROUTER_FEATURE_NAMES)] == ROUTER_FEATURE_NAMES
    assert universe[len(ROUTER_FEATURE_NAMES) :] == (
        TILE_CONSISTENCY_FEATURE_NAMES
    )
    with pytest.raises(ValueError):
        _tile_feature_config(
            {
                "tile_features": {
                    "enabled": True,
                    "tile_size": 512,
                    "names": ["tile_magnitude_spread"],
                }
            }
        )


def test_join_tile_features_requires_full_coverage(tmp_path) -> None:
    rows = [
        {"id": "a", "scenario": "clean:s0:common"},
        {"id": "a", "scenario": "shift:s3:t2"},
    ]
    path = tmp_path / "tiles.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "scenario", *TILE_CONSISTENCY_FEATURE_NAMES],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "a",
                "scenario": "clean:s0:common",
                **{name: 1.0 for name in TILE_CONSISTENCY_FEATURE_NAMES},
            }
        )
    with pytest.raises(RuntimeError):
        _join_tile_features(rows, path, TILE_CONSISTENCY_FEATURE_NAMES)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "scenario", *TILE_CONSISTENCY_FEATURE_NAMES],
        )
        writer.writerow(
            {
                "id": "a",
                "scenario": "shift:s3:t2",
                **{name: 2.0 for name in TILE_CONSISTENCY_FEATURE_NAMES},
            }
        )
    joined = _join_tile_features(
        rows,
        path,
        TILE_CONSISTENCY_FEATURE_NAMES,
    )
    assert joined == 2
    assert rows[1]["tile_dispersion_radius"] == pytest.approx(2.0)


def _scenario_rows(clean, s1, s2, s8, method="safe_veto_task"):
    rows = []
    for scenario, raw, primary in (
        ("clean:s0:common", 0.90, clean),
        ("shift:s1:t2", 0.85, s1),
        ("shift:s2:t2", 0.80, s2),
        ("shift:s3:t2", 0.70, s8),
    ):
        rows.append(
            {
                "scenario": scenario,
                "method": "raw",
                "f1": raw,
                "aligned_selection_rate": None,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "f1": primary,
                "aligned_selection_rate": 0.5,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "method": "loss_oracle",
                "f1": raw + (0.0 if scenario.startswith("clean") else 0.2),
                "aligned_selection_rate": 0.5,
            }
        )
    return rows


_PROTOCOL = {
    "primary_method": "safe_veto_task",
    "seed": 3407,
    "shift_severities": [1, 2, 3],
    "gates": {
        "clean_f1_drop_max": 0.001,
        "shift_mean_f1_gain_min": 0.020,
        "shift8_f1_gain_min": 0.050,
        "shift8_oracle_recovery_min": 0.50,
    },
    "improvement_gates": {
        "shift8_gain_improvement_min": 0.010,
        "shift_mean_gain_drop_max": 0.002,
    },
}


def test_decision_improved_routes_to_phase_b() -> None:
    baseline = _scenario_rows(0.90, 0.90, 0.88, 0.83)
    current = _scenario_rows(0.90, 0.905, 0.885, 0.845)
    decision = _decision(
        current,
        protocol=_PROTOCOL,
        baseline_rows=baseline,
    )
    assert decision["status"] == "passed"
    assert decision["improvement_passed"] is True
    assert decision["next_action"] == "p7_5_phase_b_three_seeds"


def test_decision_neutral_keeps_p7_3() -> None:
    baseline = _scenario_rows(0.90, 0.90, 0.88, 0.83)
    current = _scenario_rows(0.90, 0.90, 0.88, 0.835)  # +0.005 only
    decision = _decision(
        current,
        protocol=_PROTOCOL,
        baseline_rows=baseline,
    )
    assert decision["status"] == "passed"
    assert decision["improvement_passed"] is False
    assert decision["regressed_against_baseline"] is False
    assert (
        decision["next_action"]
        == "p7_5_expansion_neutral_keep_p7_3_as_final"
    )


def test_decision_regressed_is_labelled_regressed_not_neutral() -> None:
    # The real P7-5 shape: absolute gates clear, but both paired deltas
    # are negative. Routing is the same (keep P7-3); the LABEL must not
    # claim neutrality.
    baseline = _scenario_rows(0.90, 0.90, 0.88, 0.83)
    current = _scenario_rows(0.90, 0.895, 0.875, 0.820)
    decision = _decision(
        current,
        protocol=_PROTOCOL,
        baseline_rows=baseline,
    )
    assert decision["status"] == "passed"
    assert decision["improvement_passed"] is False
    assert decision["regressed_against_baseline"] is True
    assert (
        decision["next_action"]
        == "p7_5_expansion_regressed_keep_p7_3_as_final"
    )


def test_decision_s8_bought_by_hurting_small_shifts_is_rejected() -> None:
    baseline = _scenario_rows(0.90, 0.90, 0.88, 0.83)
    # S8 improves a lot but S1/S2 regress: shift-mean delta < -0.002.
    current = _scenario_rows(0.90, 0.88, 0.86, 0.86)
    decision = _decision(
        current,
        protocol=_PROTOCOL,
        baseline_rows=baseline,
    )
    assert decision["status"] == "passed"
    assert decision["improvement_passed"] is False
    assert decision["regressed_against_baseline"] is True


def test_decision_base_gate_failure_discards_expansion() -> None:
    baseline = _scenario_rows(0.90, 0.90, 0.88, 0.83)
    current = _scenario_rows(0.885, 0.905, 0.885, 0.845)  # clean drop
    decision = _decision(
        current,
        protocol=_PROTOCOL,
        baseline_rows=baseline,
    )
    assert decision["status"] == "failed"
    assert (
        decision["next_action"]
        == "p7_5_expansion_harmful_discard_keep_p7_3"
    )


def test_decision_without_improvement_block_keeps_legacy_routes() -> None:
    protocol = {
        key: value
        for key, value in _PROTOCOL.items()
        if key != "improvement_gates"
    }
    decision = _decision(
        _scenario_rows(0.90, 0.905, 0.885, 0.845),
        protocol=protocol,
    )
    assert "improvement_gates" not in decision
    assert (
        decision["next_action"]
        == "freeze_structure_then_multi_seed_and_s2looking_transfer"
    )


def test_declared_improvement_without_baseline_raises() -> None:
    with pytest.raises(ValueError):
        _decision(
            _scenario_rows(0.90, 0.905, 0.885, 0.845),
            protocol=_PROTOCOL,
            baseline_rows=None,
        )


def test_expanded_sets_do_not_touch_frozen_names() -> None:
    assert len(SAFE_REGISTRATION_FEATURE_NAMES) == 10
    assert len(SAFE_TASK_FEATURE_NAMES) == 23
    assert not (
        set(TILE_CONSISTENCY_FEATURE_NAMES) & set(ROUTER_FEATURE_NAMES)
    )


def test_risk_coverage_attribution_separates_causes() -> None:
    from tools.compare_risk_coverage import (
        attribute_regression,
        retained_benefit_curve,
    )

    def rows(retained_by_tau):
        out = []
        for tau, retained in retained_by_tau.items():
            out.append(
                {
                    "tau": str(tau),
                    "scenario": "shift:s3:t2",
                    "accepted_benefit": str(retained),
                    "q_benefit": "1.0",
                }
            )
            out.append(
                {
                    "tau": str(tau),
                    "scenario": "clean:s0:common",
                    "accepted_benefit": "0.0",
                    "q_benefit": "0.0",
                }
            )
        return out

    # Same curve for both, but the expanded run's tau moved up: the loss
    # is entirely threshold placement, not discrimination.
    curve = {0.90: 0.80, 0.95: 0.60, 0.99: 0.30}
    result = attribute_regression(
        rows(curve),
        rows(curve),
        baseline_tau=0.90,
        expanded_tau=0.99,
    )
    assert result["discrimination_component"] == pytest.approx(0.0)
    assert result["threshold_component"] == pytest.approx(-0.50)
    assert result["dominant_cause"] == "threshold_placement"
    assert result["attribution_only_not_a_verdict"] is True

    # Same tau, worse curve: discrimination.
    worse = {0.90: 0.55, 0.95: 0.40, 0.99: 0.20}
    result = attribute_regression(
        rows(curve),
        rows(worse),
        baseline_tau=0.90,
        expanded_tau=0.90,
    )
    assert result["threshold_component"] == pytest.approx(0.0)
    assert result["discrimination_component"] == pytest.approx(-0.25)
    assert result["dominant_cause"] == "discrimination"

    # Clean rows carry no shift benefit and must not enter the curve.
    assert retained_benefit_curve(rows(curve))[0] == (0.90, 0.80)


def test_risk_coverage_uses_linear_interpolation_between_grid_points() -> None:
    from tools.compare_risk_coverage import _interpolate

    curve = [(0.0, 1.0), (1.0, 0.0)]
    assert _interpolate(curve, 0.25) == pytest.approx(0.75)
    assert _interpolate(curve, -1.0) == pytest.approx(1.0)
    assert _interpolate(curve, 2.0) == pytest.approx(0.0)
