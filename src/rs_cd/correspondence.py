from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class OffsetProbeResult:
    dx: int
    dy: int
    best_score: float
    second_score: float

    @property
    def margin(self) -> float:
        return self.best_score - self.second_score


@torch.no_grad()
def global_cosine_offset(
    reference: torch.Tensor,
    target: torch.Tensor,
    radius: int,
    valid_mask: torch.Tensor | None = None,
    spatial_step: int = 1,
) -> OffsetProbeResult:
    """Find the target offset that best matches a reference feature map.

    A returned ``(dx, dy)`` means that reference position ``(x, y)`` is
    matched against target position ``(x + dx, y + dy)``.
    """

    if reference.ndim != 3 or target.ndim != 3:
        raise ValueError("reference and target must have shape CxHxW")
    if reference.shape != target.shape:
        raise ValueError("reference and target feature shapes must match")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if spatial_step < 1:
        raise ValueError("spatial_step must be positive")
    _, height, width = reference.shape
    normalized_reference = F.normalize(
        reference.float(), dim=0, eps=1e-6
    )
    normalized_target = F.normalize(target.float(), dim=0, eps=1e-6)
    if valid_mask is None:
        valid = torch.ones(
            height,
            width,
            dtype=torch.bool,
            device=reference.device,
        )
    else:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(0)
        if valid_mask.shape != (height, width):
            raise ValueError("valid_mask spatial shape must match features")
        valid = valid_mask >= 0.5

    candidates: list[tuple[float, int, int]] = []
    for dy in range(-radius, radius + 1):
        reference_y0 = max(0, -dy)
        reference_y1 = min(height, height - dy)
        for dx in range(-radius, radius + 1):
            reference_x0 = max(0, -dx)
            reference_x1 = min(width, width - dx)
            reference_slice = normalized_reference[
                :,
                reference_y0:reference_y1:spatial_step,
                reference_x0:reference_x1:spatial_step,
            ]
            target_slice = normalized_target[
                :,
                reference_y0 + dy : reference_y1 + dy : spatial_step,
                reference_x0 + dx : reference_x1 + dx : spatial_step,
            ]
            valid_slice = valid[
                reference_y0:reference_y1:spatial_step,
                reference_x0:reference_x1:spatial_step,
            ]
            similarities = (reference_slice * target_slice).sum(dim=0)
            if bool(valid_slice.any()):
                score = float(similarities[valid_slice].mean())
            else:
                score = float("-inf")
            candidates.append((score, dx, dy))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_dx, best_dy = candidates[0]
    second_score = (
        candidates[1][0] if len(candidates) > 1 else best_score
    )
    return OffsetProbeResult(
        dx=best_dx,
        dy=best_dy,
        best_score=best_score,
        second_score=second_score,
    )
