import math

import torch

from rs_cd.losses import BinaryChangeLoss
from rs_cd.metrics import BinaryChangeMetrics


def test_binary_metrics_known_confusion_matrix() -> None:
    probabilities = torch.tensor([[[[0.9, 0.8], [0.7, 0.1]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    metrics = BinaryChangeMetrics(threshold=0.5)
    metrics.update(probabilities, targets, from_logits=False)
    result = metrics.compute()
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tn"] == 1
    assert math.isclose(result["precision"], 2 / 3)
    assert math.isclose(result["recall"], 1.0)
    assert math.isclose(result["f1"], 0.8)
    assert math.isclose(result["iou"], 2 / 3)
    assert math.isclose(result["oa"], 0.75)


def test_loss_is_finite_for_empty_mask() -> None:
    criterion = BinaryChangeLoss()
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    targets = torch.zeros_like(logits)
    loss, components = criterion(
        {"change_logits": logits, "aux": None}, targets
    )
    assert torch.isfinite(loss)
    assert all(math.isfinite(value) for value in components.values())
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_binary_metrics_respect_valid_mask() -> None:
    probabilities = torch.tensor([[[[0.9, 0.9], [0.1, 0.1]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    valid_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    metrics = BinaryChangeMetrics(threshold=0.5)
    metrics.update(
        probabilities,
        targets,
        from_logits=False,
        valid_mask=valid_mask,
    )
    result = metrics.compute()
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 1
