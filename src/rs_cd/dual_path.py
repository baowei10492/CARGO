from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ROUTER_FEATURE_NAMES = (
    "shift_probability",
    "zero_probability",
    "registration_confidence",
    "registration_margin",
    "registration_local_margin",
    "registration_reliability",
    "conditional_dx_normalized",
    "conditional_dy_normalized",
    "conditional_abs_dx_normalized",
    "conditional_abs_dy_normalized",
    "conditional_magnitude_normalized",
    "conditional_boundary",
    "raw_mean_probability",
    "aligned_mean_probability",
    "raw_change_fraction",
    "aligned_change_fraction",
    "raw_entropy",
    "aligned_entropy",
    "raw_mean_confidence",
    "aligned_mean_confidence",
    "probability_disagreement",
    "binary_disagreement_fraction",
    "raw_only_change_fraction",
    "aligned_only_change_fraction",
    "prediction_intersection_over_union",
)

REGISTRATION_ROUTER_FEATURE_NAMES = ROUTER_FEATURE_NAMES[:12]

_SIGNED_OFFSET_FEATURE_NAMES = (
    "conditional_dx_normalized",
    "conditional_dy_normalized",
)

SAFE_REGISTRATION_FEATURE_NAMES = tuple(
    name
    for name in REGISTRATION_ROUTER_FEATURE_NAMES
    if name not in _SIGNED_OFFSET_FEATURE_NAMES
)

SAFE_TASK_FEATURE_NAMES = (
    *SAFE_REGISTRATION_FEATURE_NAMES,
    *ROUTER_FEATURE_NAMES[12:],
)

# The four tile-consistency features that survived BOTH screening passes
# (pooled bar AND the per-scenario rule preregistered in main record
# 69.2: within-clean AND within-Shift-8 directional AUROC >= 0.60).
# tile_magnitude_spread passed pooled but failed within-clean (0.517)
# and is therefore ablation-only; tile_mean_offset_magnitude failed both.
TILE_CONSISTENCY_FEATURE_NAMES = (
    "tile_max_pairwise_distance",
    "tile_dispersion_radius",
    "tile_agreement_within_2px",
    "tile_agreement_within_4px",
)


def tile_consistency_features(
    tile_offsets: list[tuple[float, float]],
    whole_offset: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Reduce per-tile offset estimates to the screened features.

    The formulas MUST stay identical to the screening implementation in
    ``tools.screen_tile_consistency_features.tile_features_by_key`` (a
    test cross-checks the two on shared inputs): the screening AUROCs
    were measured on that reduction, so any drift here would silently
    invalidate the feature-inclusion evidence.
    """

    if len(tile_offsets) < 2:
        raise ValueError("Need at least two tile offsets")

    def _distance(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
        ) ** 0.5

    widest = max(
        _distance(tile_offsets[first], tile_offsets[second])
        for first in range(len(tile_offsets))
        for second in range(first + 1, len(tile_offsets))
    )
    centre = (
        sum(tile[0] for tile in tile_offsets) / len(tile_offsets),
        sum(tile[1] for tile in tile_offsets) / len(tile_offsets),
    )
    radius = sum(
        _distance(tile, centre) for tile in tile_offsets
    ) / len(tile_offsets)
    reference = whole_offset if whole_offset is not None else centre
    return {
        "tile_max_pairwise_distance": widest,
        "tile_dispersion_radius": radius,
        "tile_agreement_within_2px": sum(
            1
            for tile in tile_offsets
            if _distance(tile, reference) <= 2.0
        )
        / len(tile_offsets),
        "tile_agreement_within_4px": sum(
            1
            for tile in tile_offsets
            if _distance(tile, reference) <= 4.0
        )
        / len(tile_offsets),
    }


class SafeAlignmentClassifier(nn.Module):
    """Binary head scoring whether switching to Aligned is safe.

    Unlike :class:`TaskBenefitRouter`, this head does not regress the full
    task benefit. It returns one logit per sample for
    ``P(aligned_loss + margin < raw_loss)`` and is used exclusively as a
    veto on q-gate candidates with a Raw default. It must only consume
    direction-invariant features (no signed dx/dy).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        first, second = (int(value) for value in hidden_dims)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), first),
            nn.GELU(),
            nn.LayerNorm(first),
            nn.Dropout(float(dropout)),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(second, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class TaskBenefitRouter(nn.Module):
    """Small regressor predicting ``loss_raw - loss_aligned``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        first, second = (int(value) for value in hidden_dims)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), first),
            nn.GELU(),
            nn.LayerNorm(first),
            nn.Dropout(float(dropout)),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(second, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def _masked_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if valid_mask.shape != values.shape:
        valid_mask = torch.broadcast_to(valid_mask, values.shape)
    valid = valid_mask.to(device=values.device, dtype=values.dtype)
    denominator = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    return (values * valid).flatten(1).sum(dim=1) / denominator


def dual_path_router_features(
    raw_logits: torch.Tensor,
    aligned_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    registration: dict[str, torch.Tensor],
    *,
    maximum_offset_pixels: float,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Extract label-free registration and task-disagreement features."""

    if maximum_offset_pixels <= 0:
        raise ValueError("maximum_offset_pixels must be positive")
    raw_probability = torch.sigmoid(raw_logits.float())
    aligned_probability = torch.sigmoid(aligned_logits.float())
    valid = valid_mask.to(raw_probability)
    if valid.shape != raw_probability.shape:
        valid = torch.broadcast_to(valid, raw_probability.shape)
    raw_binary = raw_probability >= float(threshold)
    aligned_binary = aligned_probability >= float(threshold)
    valid_binary = valid >= 0.5
    raw_binary_float = raw_binary.to(raw_probability)
    aligned_binary_float = aligned_binary.to(raw_probability)
    epsilon = 1e-6
    raw_entropy_map = -(
        raw_probability.clamp(epsilon, 1.0 - epsilon)
        * raw_probability.clamp(epsilon, 1.0 - epsilon).log()
        + (1.0 - raw_probability).clamp(epsilon, 1.0 - epsilon)
        * (1.0 - raw_probability)
        .clamp(epsilon, 1.0 - epsilon)
        .log()
    )
    aligned_entropy_map = -(
        aligned_probability.clamp(epsilon, 1.0 - epsilon)
        * aligned_probability.clamp(epsilon, 1.0 - epsilon).log()
        + (1.0 - aligned_probability).clamp(epsilon, 1.0 - epsilon)
        * (1.0 - aligned_probability)
        .clamp(epsilon, 1.0 - epsilon)
        .log()
    )
    raw_confidence = (raw_probability - 0.5).abs() * 2.0
    aligned_confidence = (aligned_probability - 0.5).abs() * 2.0
    disagreement = (raw_binary != aligned_binary) & valid_binary
    raw_only = raw_binary & (~aligned_binary) & valid_binary
    aligned_only = aligned_binary & (~raw_binary) & valid_binary
    intersection = (
        (raw_binary & aligned_binary & valid_binary)
        .flatten(1)
        .sum(dim=1)
        .to(raw_probability)
    )
    union = (
        ((raw_binary | aligned_binary) & valid_binary)
        .flatten(1)
        .sum(dim=1)
        .to(raw_probability)
    )
    prediction_iou = (intersection + 1.0) / (union + 1.0)

    conditional_offset = registration[
        "conditional_hard_offset_pixels"
    ].float()
    normalized_offset = conditional_offset / float(maximum_offset_pixels)
    normalized_abs = normalized_offset.abs()
    normalized_magnitude = torch.linalg.vector_norm(
        normalized_offset,
        dim=1,
    ) / (2.0**0.5)
    boundary = (
        normalized_abs.amax(dim=1) >= 1.0 - 1e-6
    ).to(raw_probability)
    valid_count = valid_binary.flatten(1).sum(dim=1).clamp_min(1)

    columns = (
        registration["registration_shift_probability"].float(),
        registration["registration_zero_probability"].float(),
        registration["registration_confidence"].float(),
        registration["registration_margin"].float(),
        registration["registration_local_margin"].float(),
        registration["registration_reliability"].float(),
        normalized_offset[:, 0],
        normalized_offset[:, 1],
        normalized_abs[:, 0],
        normalized_abs[:, 1],
        normalized_magnitude,
        boundary,
        _masked_mean(raw_probability, valid),
        _masked_mean(aligned_probability, valid),
        _masked_mean(raw_binary_float, valid),
        _masked_mean(aligned_binary_float, valid),
        _masked_mean(raw_entropy_map, valid),
        _masked_mean(aligned_entropy_map, valid),
        _masked_mean(raw_confidence, valid),
        _masked_mean(aligned_confidence, valid),
        _masked_mean(
            (raw_probability - aligned_probability).abs(),
            valid,
        ),
        disagreement.flatten(1).sum(dim=1).to(raw_probability)
        / valid_count,
        raw_only.flatten(1).sum(dim=1).to(raw_probability)
        / valid_count,
        aligned_only.flatten(1).sum(dim=1).to(raw_probability)
        / valid_count,
        prediction_iou,
    )
    features = torch.stack(columns, dim=1)
    if features.shape[1] != len(ROUTER_FEATURE_NAMES):
        raise RuntimeError("Router feature definition is inconsistent")
    return features


def per_sample_binary_change_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: float | None = None,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return a label-aware task loss for every image in a batch.

    This is used only to measure the upper bound of a Raw/Aligned router. It
    must not be used as a deployable inference rule because it reads targets.
    """

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if valid_mask.shape != targets.shape:
        valid_mask = torch.broadcast_to(valid_mask, targets.shape)
    valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    positive_weight = (
        logits.new_tensor([pos_weight])
        if pos_weight is not None
        else None
    )
    per_pixel_bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    valid_pixels = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    bce = (per_pixel_bce * valid).flatten(1).sum(dim=1) / valid_pixels

    probabilities = torch.sigmoid(logits) * valid
    masked_targets = targets * valid
    intersection = (
        probabilities * masked_targets
    ).flatten(1).sum(dim=1)
    denominator = probabilities.flatten(1).sum(dim=1) + (
        masked_targets.flatten(1).sum(dim=1)
    )
    dice = 1.0 - (
        (2.0 * intersection + float(smooth))
        / (denominator + float(smooth))
    )
    return float(bce_weight) * bce + float(dice_weight) * dice


def select_aligned_by_task_loss(
    raw_logits: torch.Tensor,
    aligned_logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one complete prediction per image using lower target loss."""

    raw_loss = per_sample_binary_change_loss(
        raw_logits,
        targets,
        valid_mask,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
    )
    aligned_loss = per_sample_binary_change_loss(
        aligned_logits,
        targets,
        valid_mask,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
    )
    use_aligned = aligned_loss < raw_loss
    selected = torch.where(
        use_aligned[:, None, None, None],
        aligned_logits,
        raw_logits,
    )
    return selected, use_aligned, raw_loss, aligned_loss
