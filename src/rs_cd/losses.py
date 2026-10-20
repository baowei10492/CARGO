from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    targets = targets.float()
    if valid is not None:
        probabilities = probabilities * valid
        targets = targets * valid
    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class BinaryChangeLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        if self.bce_weight < 0 or self.dice_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if self.bce_weight == 0 and self.dice_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        self.register_buffer(
            "_pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        targets: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``valid`` (same shape as targets, 0/1) leaves pixels out of both terms."""
        logits = outputs["change_logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("outputs['change_logits'] must be a Tensor")
        targets = targets.float()
        if valid is None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self._pos_weight
            )
        else:
            # float32: a half-precision sum over a batch of crops overflows.
            valid = valid.float()
            bce = F.binary_cross_entropy_with_logits(
                logits.float(), targets, weight=valid, pos_weight=self._pos_weight,
                reduction="sum",
            ) / valid.sum().clamp_min(1.0)
        dice = soft_dice_loss(logits, targets, valid=valid)
        total = self.bce_weight * bce + self.dice_weight * dice
        aux_loss = outputs.get("aux_loss")
        aux_metrics = outputs.get("aux_metrics", {})
        if isinstance(aux_loss, torch.Tensor):
            total = total + aux_loss
        return total, {
            "loss": float(total.detach()),
            "bce": float(bce.detach()),
            "dice": float(dice.detach()),
            **(
                {
                    key: float(value)
                    for key, value in aux_metrics.items()
                }
                if isinstance(aux_metrics, dict)
                else {}
            ),
        }
