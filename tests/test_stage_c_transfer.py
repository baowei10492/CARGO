import pytest
import torch

from tools.evaluate_safe_alignment_transfer import (
    conditional_pck_rates,
    merge_detector_state,
    transfer_decision,
    unlabeled_gate_decision,
)


def _protocol() -> dict:
    return {
        "seed": 3407,
        "shift_severities": [1, 2, 3],
        "unlabeled_gate": {"shift8_conditional_pck1_min": 0.50},
        "gates": {
            "clean_f1_drop_max": 0.001,
            "shift8_oracle_recovery_min": 0.50,
            "shift4_oracle_recovery_min": 0.30,
            "shift2_f1_gain_min": -0.001,
            "shift8_oracle_gain_min": 0.005,
        },
        "primary_method": "safe_veto_registration",
    }


def test_merge_detector_state_replaces_only_detector_keys() -> None:
    task_state = {
        "backbone.stem.weight": torch.zeros(2),
        "decoder.head.weight": torch.zeros(2),
        "registration.descriptor.weight": torch.zeros(2),
    }
    detector_state = {
        "backbone.stem.weight": torch.ones(2),
        "decoder.head.weight": torch.full((2,), 2.0),
    }
    merged, replaced = merge_detector_state(task_state, detector_state)
    assert replaced == 2
    assert torch.equal(merged["backbone.stem.weight"], torch.ones(2))
    assert torch.equal(
        merged["decoder.head.weight"],
        torch.full((2,), 2.0),
    )
    assert torch.equal(
        merged["registration.descriptor.weight"],
        torch.zeros(2),
    )


def test_merge_detector_state_rejects_key_mismatch() -> None:
    task_state = {
        "backbone.stem.weight": torch.zeros(2),
        "registration.descriptor.weight": torch.zeros(2),
    }
    with pytest.raises(RuntimeError):
        merge_detector_state(
            task_state,
            {"backbone.other.weight": torch.ones(2)},
        )
    with pytest.raises(RuntimeError):
        merge_detector_state(task_state, {})


def test_conditional_pck_rates() -> None:
    predicted = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [0.0, 2.0]]
    )
    target = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    rates = conditional_pck_rates(predicted, target)
    assert rates["pck1"] == pytest.approx(0.5)
    assert rates["pck2"] == pytest.approx(0.75)
    assert rates["pck4"] == pytest.approx(0.75)
    assert rates["mean_epe_pixels"] == pytest.approx(
        (0.0 + 1.0 + 5.0 + 2.0) / 4.0
    )


def test_unlabeled_gate_decision() -> None:
    protocol = _protocol()
    diagnostics = [
        {"scenario": "clean:s0:common", "pck1": 0.10},
        {"scenario": "shift:s1:t2", "pck1": 0.90},
        {"scenario": "shift:s2:t2", "pck1": 0.80},
        {"scenario": "shift:s3:t2", "pck1": 0.55},
    ]
    gates = unlabeled_gate_decision(diagnostics, protocol=protocol)
    assert gates["shift8_conditional_pck1"]["passed"]
    diagnostics[-1]["pck1"] = 0.40
    gates = unlabeled_gate_decision(diagnostics, protocol=protocol)
    assert not gates["shift8_conditional_pck1"]["passed"]


def _rows(primary_f1: dict, raw_f1: dict, oracle_f1: dict) -> list:
    rows = []
    for scenario in raw_f1:
        rows.append(
            {
                "scenario": scenario,
                "method": "raw",
                "f1": raw_f1[scenario],
                "aligned_selection_rate": None,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "method": "safe_veto_registration",
                "f1": primary_f1[scenario],
                "aligned_selection_rate": 0.4,
            }
        )
        rows.append(
            {
                "scenario": scenario,
                "method": "loss_oracle",
                "f1": oracle_f1[scenario],
                "aligned_selection_rate": 0.9,
            }
        )
    return rows


def test_transfer_decision_passes_with_relative_recovery() -> None:
    protocol = _protocol()
    raw = {
        "clean:s0:common": 0.710,
        "shift:s1:t2": 0.707,
        "shift:s2:t2": 0.698,
        "shift:s3:t2": 0.678,
    }
    oracle = {
        "clean:s0:common": 0.7105,
        "shift:s1:t2": 0.7105,
        "shift:s2:t2": 0.7105,
        "shift:s3:t2": 0.7105,
    }
    primary = {
        "clean:s0:common": 0.710,
        "shift:s1:t2": 0.708,
        "shift:s2:t2": 0.704,
        "shift:s3:t2": 0.697,
    }
    decision = transfer_decision(
        _rows(primary, raw, oracle),
        protocol=protocol,
    )
    # shift8 oracle gain = 0.0325, recovery = 0.019/0.0325 = 58.5%
    # shift4 oracle gain = 0.0125, recovery = 0.006/0.0125 = 48%
    assert decision["gates"]["shift8_transferred_oracle_gain"]["passed"]
    assert decision["gates"]["shift8_oracle_gain_recovery"]["passed"]
    assert decision["gates"]["shift4_oracle_gain_recovery"]["passed"]
    assert decision["gates"]["shift2_primary_f1_gain"]["passed"]
    assert decision["status"] == "passed"


def test_transfer_decision_fails_without_transferred_headroom() -> None:
    protocol = _protocol()
    raw = {
        "clean:s0:common": 0.710,
        "shift:s1:t2": 0.707,
        "shift:s2:t2": 0.698,
        "shift:s3:t2": 0.678,
    }
    # Oracle barely above raw: the aligned expert did not transfer.
    oracle = {
        "clean:s0:common": 0.710,
        "shift:s1:t2": 0.7071,
        "shift:s2:t2": 0.6981,
        "shift:s3:t2": 0.6801,
    }
    primary = dict(raw)
    decision = transfer_decision(
        _rows(primary, raw, oracle),
        protocol=protocol,
    )
    assert not decision["gates"]["shift8_transferred_oracle_gain"][
        "passed"
    ]
    assert decision["status"] == "failed"


def test_transfer_decision_fails_on_clean_damage() -> None:
    protocol = _protocol()
    raw = {
        "clean:s0:common": 0.710,
        "shift:s1:t2": 0.707,
        "shift:s2:t2": 0.698,
        "shift:s3:t2": 0.678,
    }
    oracle = {
        "clean:s0:common": 0.7105,
        "shift:s1:t2": 0.7105,
        "shift:s2:t2": 0.7105,
        "shift:s3:t2": 0.7105,
    }
    primary = {
        "clean:s0:common": 0.705,
        "shift:s1:t2": 0.708,
        "shift:s2:t2": 0.704,
        "shift:s3:t2": 0.697,
    }
    decision = transfer_decision(
        _rows(primary, raw, oracle),
        protocol=protocol,
    )
    assert not decision["gates"]["clean_primary_f1_drop"]["passed"]
    assert decision["status"] == "failed"
