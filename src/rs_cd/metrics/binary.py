from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryChangeMetrics:
    threshold: float = 0.5
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def reset(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    @torch.no_grad()
    def update(
        self,
        logits_or_probabilities: torch.Tensor,
        targets: torch.Tensor,
        from_logits: bool = True,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        values = (
            torch.sigmoid(logits_or_probabilities)
            if from_logits
            else logits_or_probabilities
        )
        predictions = values >= self.threshold
        truth = targets >= 0.5
        if valid_mask is not None:
            valid = valid_mask >= 0.5
            if valid.shape != truth.shape:
                valid = torch.broadcast_to(valid, truth.shape)
            predictions = predictions[valid]
            truth = truth[valid]
        self.true_positive += int(torch.logical_and(predictions, truth).sum())
        self.false_positive += int(
            torch.logical_and(predictions, torch.logical_not(truth)).sum()
        )
        self.false_negative += int(
            torch.logical_and(torch.logical_not(predictions), truth).sum()
        )
        self.true_negative += int(
            torch.logical_and(
                torch.logical_not(predictions), torch.logical_not(truth)
            ).sum()
        )

    def compute(self) -> dict[str, float | int]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "oa": accuracy,
        }
