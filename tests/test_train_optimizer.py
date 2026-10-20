"""Optimizer and schedule builders behind tools/train.py.

The default must remain exactly the recipe every frozen run used, and each
published P21 schedule must keep the shape its source repository gives it
when rescaled to this project's epoch budget.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rs_cd.optimization import build_optimizer, build_scheduler

ROOT = Path(__file__).resolve().parents[1]


def _run(training: dict, epochs: int):
    optimizer = build_optimizer(
        [{"params": [torch.nn.Parameter(torch.zeros(1))]}], training
    )
    scheduler = build_scheduler(optimizer, training, epochs)
    rates = []
    for _ in range(epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return optimizer, rates


def test_default_is_the_frozen_adamw_cosine_recipe() -> None:
    optimizer, rates = _run({"learning_rate": 3e-4, "weight_decay": 1e-4}, 40)

    reference_optimizer = torch.optim.AdamW(
        [torch.nn.Parameter(torch.zeros(1))], lr=3e-4, weight_decay=1e-4
    )
    reference = torch.optim.lr_scheduler.CosineAnnealingLR(
        reference_optimizer, T_max=40
    )
    expected = []
    for _ in range(40):
        expected.append(reference_optimizer.param_groups[0]["lr"])
        reference_optimizer.step()
        reference.step()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["weight_decay"] == 1e-4
    assert rates == expected


def test_sgd_carries_the_published_momentum_and_decay() -> None:
    optimizer = build_optimizer(
        [{"params": [torch.nn.Parameter(torch.zeros(1))]}],
        {"optimizer": "sgd", "learning_rate": 0.01, "momentum": 0.9,
         "weight_decay": 5e-4},
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9
    assert optimizer.defaults["weight_decay"] == 5e-4
    assert optimizer.defaults["nesterov"] is False


def test_linear_is_the_bit_cd_rule_over_this_budget() -> None:
    _, rates = _run(
        {"optimizer": "sgd", "learning_rate": 0.01, "scheduler": "linear"}, 40
    )
    for epoch, rate in enumerate(rates):
        assert rate == pytest.approx(0.01 * (1.0 - epoch / 41.0))


def test_poly_is_the_rschange_rule_over_this_budget() -> None:
    _, rates = _run(
        {"optimizer": "sgd", "learning_rate": 0.01, "scheduler": "poly",
         "poly_power": 0.9},
        40,
    )
    for epoch, rate in enumerate(rates):
        assert rate == pytest.approx(0.01 * (1.0 - epoch / 40.0) ** 0.9)


def test_step_keeps_the_published_fraction_of_training() -> None:
    """CDxLSTM decays every 50 of 200 epochs, so every 10 of 40 here."""

    _, rates = _run(
        {"optimizer": "sgd", "learning_rate": 0.05, "scheduler": "step",
         "step_fraction": 0.25, "step_gamma": 0.1},
        40,
    )
    assert rates[9] == pytest.approx(0.05)
    assert rates[10] == pytest.approx(0.005)
    assert rates[39] == pytest.approx(0.00005)


def test_unknown_names_are_errors_not_silent_defaults() -> None:
    parameters = [{"params": [torch.nn.Parameter(torch.zeros(1))]}]
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(parameters, {"optimizer": "lamb", "learning_rate": 0.1})
    optimizer = build_optimizer(parameters, {"learning_rate": 0.1})
    with pytest.raises(ValueError, match="scheduler"):
        build_scheduler(optimizer, {"scheduler": "warmup"}, 10)


def test_train_entry_point_uses_the_builders() -> None:
    source = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")
    assert "build_optimizer(parameter_groups, training)" in source
    assert "build_scheduler(optimizer, training, epochs)" in source
    assert "torch.optim.AdamW(" not in source
