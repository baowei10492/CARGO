"""Optimizers and per-epoch learning-rate schedules for tools/train.py.

The defaults (AdamW, cosine over the run's epochs) are the recipe every
frozen run used and must not change. The other options exist for P21, where
each published detector is trained with its own optimizer (protocol section
2, amendment A) and its schedule is rescaled to this project's epoch budget
by keeping its shape:

  linear : BIT_CD models/networks.py, lr * (1 - e / (E + 1))
  poly   : rschange rscd/optimizers/build_optimizer.py 'Poly',
           lr * (1 - e / E) ** power
  step   : rschange 'step', decay by gamma every step_fraction * E epochs
           (CDxLSTM decays every 50 of 200 epochs, so step_fraction 0.25)

Both source repositories step the schedule once per epoch, as tools/train.py
does, so the rescaling changes only E.
"""
from __future__ import annotations

from typing import Any

import torch


def build_optimizer(
    parameter_groups: list[dict[str, Any]],
    training: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(training.get("optimizer", "adamw")).lower()
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(
            parameter_groups, lr=learning_rate, weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=float(training.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=False,
        )
    raise ValueError(f"unsupported optimizer {name!r}; expected adamw or sgd")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    name = str(training.get("scheduler", "cosine")).lower()
    total = max(1, int(epochs))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    if name == "linear":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0 - epoch / float(total + 1)
        )
    if name == "poly":
        power = float(training.get("poly_power", 0.9))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / float(total)) ** power,
        )
    if name == "step":
        step_size = max(1, round(total * float(training["step_fraction"])))
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=float(training.get("step_gamma", 0.1)),
        )
    raise ValueError(
        f"unsupported scheduler {name!r}; expected cosine, linear, poly or step"
    )
