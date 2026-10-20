import math
import random

import pytest
import torch

from rs_cd.dual_path import (
    ROUTER_FEATURE_NAMES,
    SAFE_TASK_FEATURE_NAMES,
)
from tools.train_evaluate_safe_alignment_classifier import (
    _decision,
    _prepare_training_tensors,
    _stable_split_assignment,
    train_safe_alignment_classifier,
)


def _protocol(tmp_overrides: dict | None = None) -> dict:
    protocol = {
        "seed": 3407,
        "split_seed": 3407,
        "shift_severities": [1, 2, 3],
        "partition": {
            "fit_fraction": 0.70,
            "early_stop_fraction": 0.15,
        },
        "label": {"benefit_margin": 0.0},
        "weights": {
            "magnitude_scale": 0.05,
            "magnitude_floor": 0.05,
            "magnitude_cap": 4.0,
            "unsafe_weight": 3.0,
            "clean_unsafe_extra_weight": 2.0,
        },
        "classifier": {
            "hidden_dims": [16, 8],
            "dropout": 0.0,
            "max_epochs": 40,
            "early_stop_patience": 10,
            "batch_size": 64,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
        },
        "risk": {
            "probability_floor": 0.5,
            "threshold_epsilon": 0.000001,
            "coverage_grid_points": 11,
        },
        "gates": {
            "clean_f1_drop_max": 0.001,
            "shift_mean_f1_gain_min": 0.020,
            "shift8_f1_gain_min": 0.050,
            "shift8_oracle_recovery_min": 0.50,
        },
        "primary_method": "safe_veto_task",
    }
    if tmp_overrides:
        protocol.update(tmp_overrides)
    return protocol


def _synthetic_rows(sample_count: int = 80) -> list[dict]:
    generator = random.Random(7)
    scenarios = (
        "clean:s0:common",
        "shift:s1:t2",
        "shift:s2:t2",
        "shift:s3:t2",
    )
    rows = []
    for index in range(sample_count):
        sample_id = f"synthetic_{index}"
        for scenario in scenarios:
            clean = scenario.startswith("clean")
            if clean:
                benefit = generator.uniform(-0.5, 0.02)
            else:
                benefit = generator.uniform(-0.05, 0.8)
            q_active = (not clean) or generator.random() < 0.10
            row = {
                "id": sample_id,
                "scenario": scenario,
                "raw_task_loss": 1.0,
                "aligned_task_loss": 1.0 - benefit,
                "aligned_loss_advantage": benefit,
                "existing_q_selected_aligned": q_active,
            }
            signal = 1.0 if benefit > 0 else -1.0
            for feature_index, name in enumerate(
                ROUTER_FEATURE_NAMES
            ):
                noise = generator.gauss(0.0, 0.3)
                if name in SAFE_TASK_FEATURE_NAMES:
                    row[name] = signal + noise
                else:
                    row[name] = noise
            rows.append(row)
    return rows


def test_stable_split_assignment_is_deterministic_and_covers() -> None:
    counts = {"fit": 0, "early_stop": 0, "risk_calibration": 0}
    for index in range(600):
        first = _stable_split_assignment(
            f"id_{index}",
            split_seed=3407,
            fit_fraction=0.70,
            early_stop_fraction=0.15,
        )
        second = _stable_split_assignment(
            f"id_{index}",
            split_seed=3407,
            fit_fraction=0.70,
            early_stop_fraction=0.15,
        )
        assert first == second
        counts[first] += 1
    assert counts["fit"] > 300
    assert counts["early_stop"] > 40
    assert counts["risk_calibration"] > 40


def test_prepared_tensors_share_split_across_scenarios() -> None:
    rows = _synthetic_rows(40)
    prepared = _prepare_training_tensors(rows, protocol=_protocol())
    split_by_id: dict[str, str] = {}
    for row, split in zip(rows, prepared["splits"]):
        previous = split_by_id.setdefault(row["id"], split)
        assert previous == split


def test_prepared_weights_follow_preregistered_formula() -> None:
    rows = _synthetic_rows(20)
    protocol = _protocol()
    prepared = _prepare_training_tensors(rows, protocol=protocol)
    weights = protocol["weights"]
    for index, row in enumerate(rows):
        benefit = row["aligned_loss_advantage"]
        expected = min(
            max(
                abs(benefit) / weights["magnitude_scale"],
                weights["magnitude_floor"],
            ),
            weights["magnitude_cap"],
        )
        safe = benefit > protocol["label"]["benefit_margin"]
        if not safe:
            expected *= weights["unsafe_weight"]
            if row["scenario"].startswith("clean"):
                expected *= weights["clean_unsafe_extra_weight"]
        assert math.isclose(
            float(prepared["weight"][index]),
            expected,
            rel_tol=1e-5,
        )
        assert bool(prepared["safe"][index]) == safe


def test_training_vetoes_all_clean_unsafe_risk_samples(tmp_path) -> None:
    rows = _synthetic_rows(80)
    protocol = _protocol()
    prepared = _prepare_training_tensors(rows, protocol=protocol)
    bundle = train_safe_alignment_classifier(
        "safe_alignment_task",
        rows,
        prepared,
        SAFE_TASK_FEATURE_NAMES,
        protocol=protocol,
        output_dir=tmp_path,
    )
    tau = float(bundle["probability_threshold"])
    assert tau >= 0.5
    assert 1 <= int(bundle["best_epoch"]) <= 40
    probability = bundle["probability"]
    risk_mask = prepared["split_masks"]["risk_calibration"]
    clean_unsafe = (
        risk_mask & prepared["clean"] & (~prepared["safe"])
    )
    assert bool(clean_unsafe.any())
    assert bool((probability[clean_unsafe] < tau).all())
    assert (tmp_path / "safe_alignment_task.pth").exists()
    assert (tmp_path / "safe_alignment_task_summary.json").exists()
    assert (tmp_path / "safe_alignment_task_risk_coverage.csv").exists()
    audit = bundle["summary"]["risk_calibration_q_veto_audit"]
    clean_entry = audit["scenarios"]["clean:s0:common"]
    assert clean_entry["veto_harmful"] == 0
    expected_rows = (
        protocol["risk"]["coverage_grid_points"] * 4
    )
    assert len(bundle["coverage_rows"]) == expected_rows
    with torch.no_grad():
        reloaded = torch.load(
            tmp_path / "safe_alignment_task.pth",
            weights_only=False,
        )
    assert reloaded["feature_names"] == list(SAFE_TASK_FEATURE_NAMES)


def _scenario_row(scenario, method, f1, selection=None):
    return {
        "scenario": scenario,
        "method": method,
        "f1": f1,
        "aligned_selection_rate": selection,
    }


def test_decision_gates_pass_and_fail_cases() -> None:
    protocol = _protocol()
    scenarios = {
        "clean:s0:common": {"raw": 0.90, "primary": 0.8998},
        "shift:s1:t2": {"raw": 0.86, "primary": 0.89},
        "shift:s2:t2": {"raw": 0.80, "primary": 0.88},
        "shift:s3:t2": {"raw": 0.68, "primary": 0.86},
    }
    rows = []
    for scenario, values in scenarios.items():
        rows.append(_scenario_row(scenario, "raw", values["raw"]))
        rows.append(
            _scenario_row(
                scenario,
                "safe_veto_task",
                values["primary"],
                selection=0.5,
            )
        )
        rows.append(_scenario_row(scenario, "loss_oracle", 0.89))
    decision = _decision(rows, protocol=protocol)
    assert decision["status"] == "passed"
    assert decision["primary_method"] == "safe_veto_task"
    assert set(decision["gates"]) == {
        "clean_primary_f1_drop",
        "shift_mean_primary_f1_gain",
        "shift8_primary_f1_gain",
        "shift8_oracle_gain_recovery",
    }

    harmful = [dict(row) for row in rows]
    for row in harmful:
        if (
            row["method"] == "safe_veto_task"
            and row["scenario"] == "clean:s0:common"
        ):
            row["f1"] = 0.88
    failed = _decision(harmful, protocol=protocol)
    assert failed["status"] == "failed"
    assert not failed["gates"]["clean_primary_f1_drop"]["passed"]


def test_prepare_rejects_invalid_partition_fractions() -> None:
    rows = _synthetic_rows(10)
    protocol = _protocol(
        {
            "partition": {
                "fit_fraction": 0.90,
                "early_stop_fraction": 0.20,
            }
        }
    )
    with pytest.raises(ValueError):
        _prepare_training_tensors(rows, protocol=protocol)
