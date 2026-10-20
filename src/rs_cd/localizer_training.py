"""Losses, inference wrapper and P25-style evaluation for the P27 localizer."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    AffineRanges,
    affine_parameters,
    compose as np_compose,
    is_success,
    mean_displacement as np_mean_displacement,
    perturb,
    perturbation_matrix,
    transfer_error,
)
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.similarity_geometry import compose, grid, invert, mean_displacement, transfer_distance


# ---------------------------------------------------------------- losses
def charbonnier(distance: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    return torch.sqrt(distance * distance + eps * eps) - eps


def misalignment_target(matrix: torch.Tensor, points: torch.Tensor, low: float = 0.5, high: float = 2.0) -> torch.Tensor:
    """0 below ``low`` px of mean displacement, 1 above ``high``, linear between."""

    return ((mean_displacement(matrix, points) - low) / (high - low)).clamp(0.0, 1.0)


def localizer_loss(
    out1: dict[str, Any],
    out2: dict[str, Any],
    out_self: dict[str, Any],
    matrix1: torch.Tensor,
    matrix2: torch.Tensor,
    change: torch.Tensor,
    *,
    weights: dict[str, float],
    stage_weights: Sequence[float],
    points: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    stages1, stages2, stages_self = out1["stage_matrices"], out2["stage_matrices"], out_self["stage_matrices"]
    if not len(stages1) == len(stages2) == len(stages_self) == len(stage_weights):
        raise ValueError("stage weights do not match the number of stages")
    relative_truth = compose(matrix2, invert(matrix1))
    absolute = relative = self_pair = matrix1.new_zeros(())
    for w, a1, a2, a_self in zip(stage_weights, stages1, stages2, stages_self):
        absolute = absolute + w * 0.5 * (
            charbonnier(transfer_distance(a1, matrix1, points)).mean()
            + charbonnier(transfer_distance(a2, matrix2, points)).mean()
        )
        relative = relative + w * charbonnier(
            transfer_distance(compose(a2, invert(a1)), relative_truth, points)
        ).mean()
        self_pair = self_pair + w * charbonnier(transfer_distance(a_self, matrix1, points)).mean()

    coarse_size = out1["change_logits"].shape[-2:]
    change_target = (F.adaptive_avg_pool2d(change, coarse_size)[:, 0] >= 0.25).float()
    change_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(out1["change_logits"].float(), change_target)
        + F.binary_cross_entropy_with_logits(out2["change_logits"].float(), change_target)
    )
    self_change = F.binary_cross_entropy_with_logits(
        out_self["change_logits"].float(), torch.zeros_like(change_target)
    )
    misaligned = (
        F.binary_cross_entropy_with_logits(out1["misaligned_logit"].float(), misalignment_target(matrix1, points))
        + F.binary_cross_entropy_with_logits(out2["misaligned_logit"].float(), misalignment_target(matrix2, points))
        + F.binary_cross_entropy_with_logits(out_self["misaligned_logit"].float(), misalignment_target(matrix1, points))
    ) / 3.0
    total = (
        weights["relative"] * relative
        + weights["absolute"] * absolute
        + weights["self"] * self_pair
        + weights["change"] * (change_loss + 0.5 * self_change)
        + weights["misaligned"] * misaligned
    )
    parts = {
        "relative": float(relative.detach()),
        "absolute": float(absolute.detach()),
        "self": float(self_pair.detach()),
        "change": float(change_loss.detach()),
        "misaligned": float(misaligned.detach()),
        "total": float(total.detach()),
    }
    return total, parts


def batched_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    """One pass over (R, P1), (R, P2) and (T2, P1)."""

    size = batch["reference"].shape[0]
    references = torch.cat([batch["reference"], batch["reference"], batch["self_reference"]]).to(device, non_blocking=True)
    targets = torch.cat([batch["target1"], batch["target2"], batch["target1"]]).to(device, non_blocking=True)
    output = model(references, targets)

    def part(index: int) -> dict[str, Any]:
        window = slice(index * size, (index + 1) * size)
        return {
            "stage_matrices": [matrix[window] for matrix in output["stage_matrices"]],
            "change_logits": output["change_logits"][window],
            "misaligned_logit": output["misaligned_logit"][window],
        }

    return part(0), part(1), part(2)


# ------------------------------------------------------------ inference
@dataclass(frozen=True)
class LocalizerEstimate:
    matrix: np.ndarray
    converged: bool
    inliers: int
    inlier_fraction: float | None
    misaligned_probability: float


class TrainedLocalizer:
    """numpy uint8 pairs in, 2 x 3 reference -> target matrices out."""

    def __init__(self, model: torch.nn.Module, device: torch.device, *, amp: bool = True) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.amp = bool(amp) and device.type == "cuda"

    @torch.no_grad()
    def estimate_many(self, references: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> list[LocalizerEstimate]:
        reference = torch.stack([normalize(image) for image in references]).to(self.device)
        target = torch.stack([normalize(image) for image in targets]).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            output = self.model(reference, target)
        matrices = output["matrix"].float().cpu().numpy().astype(np.float64)
        probabilities = torch.sigmoid(output["misaligned_logit"].float()).cpu().numpy()
        return [
            LocalizerEstimate(matrix=matrices[i], converged=bool(np.isfinite(matrices[i]).all()), inliers=-1,
                              inlier_fraction=None, misaligned_probability=float(probabilities[i]))
            for i in range(len(references))
        ]


def step1_rows(
    localizer: TrainedLocalizer,
    *,
    dataset: str,
    crop_id: str,
    t1: np.ndarray,
    t2: np.ndarray,
    change_fraction: float,
    ranges: AffineRanges,
    levels: Sequence[int],
    seed: int,
    points: np.ndarray,
    tolerance: float,
    trigger_probability: float,
    make_target: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """P25 step 1 rows for one crop (T1 reference, T2 perturbed), all levels in one batch.

    ``make_target(matrix)`` builds a perturbed T2 other than ``perturb(t2, matrix)``
    (P30: the same matrix applied without a zero border); the clean level is T2 itself.
    """

    size = t1.shape[0]
    all_levels = [CLEAN_LEVEL, *levels]
    params = [affine_parameters(crop_id, level, ranges, seed) for level in all_levels]
    matrices = [perturbation_matrix(*p, width=size, height=size) for p in params]
    build = make_target if make_target is not None else (lambda m: perturb(t2, m))
    targets = [t2 if level == CLEAN_LEVEL else build(m) for level, m in zip(all_levels, matrices)]
    start = time.perf_counter()
    estimates = localizer.estimate_many([t1] * len(targets), targets)
    seconds = (time.perf_counter() - start) / len(targets)
    clean = estimates[0]
    rows = []
    for level, (angle, scale, tx, ty), matrix, estimate in zip(all_levels, params, matrices, estimates):
        ate_raw = transfer_error(estimate.matrix, matrix, points, width=size, height=size)
        ate_diff = None
        if level != CLEAN_LEVEL and clean.converged:
            ate_diff = transfer_error(estimate.matrix, np_compose(matrix, clean.matrix), points, width=size, height=size)
        row = {
            "dataset": dataset, "crop_id": crop_id, "level": level,
            "angle": angle, "scale": scale, "tx": tx, "ty": ty,
            "perturbation_px": np_mean_displacement(matrix, points),
            "converged": estimate.converged, "inliers": estimate.inliers,
            "inlier_fraction": estimate.inlier_fraction,
            "a11": estimate.matrix[0, 0], "a12": estimate.matrix[0, 1], "a13": estimate.matrix[0, 2],
            "a21": estimate.matrix[1, 0], "a22": estimate.matrix[1, 1], "a23": estimate.matrix[1, 2],
            "estimate_displacement_px": np_mean_displacement(estimate.matrix, points),
            "misaligned_probability": estimate.misaligned_probability,
            "trigger": estimate.misaligned_probability > trigger_probability,
            "ate_raw": ate_raw, "ate_diff": ate_diff,
            "seconds": seconds, "change_fraction": change_fraction,
        }
        row["success"] = is_success(row, tolerance) if level != CLEAN_LEVEL else None
        rows.append(row)
    return rows


def load_localizer(path: str, device: torch.device, *, backbone: torch.nn.Module | None = None) -> torch.nn.Module:
    """Rebuild a saved localizer: frozen backbone from its config, trainable weights from the file."""

    from rs_cd.models.change_aware_similarity_localizer import build_localizer

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_localizer(payload["model_config"], backbone=backbone)
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    if unexpected or any(not key.startswith("backbone.") for key in missing):
        raise SystemExit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return model.to(device).eval()


def selection_score(summaries: dict[str, dict[str, Any]], binding_levels: Sequence[int]) -> float:
    """Mean success over binding levels and datasets (development model selection)."""

    values = [
        summary["levels"][str(level)]["success_rate"]
        for summary in summaries.values()
        for level in binding_levels
    ]
    return float(np.mean(values)) if values else math.nan
