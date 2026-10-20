"""P25: DiffRegCD-style affine perturbation and similarity registration.

DiffRegCD (WACV 2026, Table 1) perturbs 256 px crops with a random rotation
(+-30 deg), scale (0.80-1.25) and translation (+-25 px) built with
cv2.getRotationMatrix2D about the crop centre. P25 applies the same family to
T2 at three levels (1/3, 2/3 and all of those ranges; level 3 is the
published default) and asks whether RoMa matches plus a RANSAC similarity fit
undo it. Every sample draws one set of uniform variables shared by the three
levels, so the levels are nested.

Conventions: pixel centres sit at integer coordinates (OpenCV). A matrix M
maps a point of the unperturbed T2 (co-registered with T1) to its position in
the perturbed T2, so the perturbed image is warpAffine(T2, M). An estimate A
maps a T1 point to the perturbed T2; warpAffine(perturbed, A,
WARP_INVERSE_MAP) samples the perturbed image back into the T1 frame.
Protocol: docs/P25_AFFINE_PERTURBATION_PROTOCOL.md.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from rs_cd.change_density import DEFAULT_BINS, assign_bin
from rs_cd.robustness import _stable_seed

CLEAN_LEVEL = 0


@dataclass(frozen=True)
class AffineRanges:
    """Level-3 ranges; level k uses k / 3 of each range around the identity."""

    max_rotation_deg: float = 30.0
    scale_low: float = 0.80
    scale_high: float = 1.25
    max_translation_px: float = 25.0
    levels: int = 3


def affine_parameters(
    sample_key: str, level: int, ranges: AffineRanges, seed: int
) -> tuple[float, float, float, float]:
    """(angle_deg, scale, tx, ty) for one crop and level; level 0 is the identity."""

    if level == CLEAN_LEVEL:
        return 0.0, 1.0, 0.0, 0.0
    if not 1 <= level <= ranges.levels:
        raise ValueError(f"level must be in 1..{ranges.levels}")
    rng = np.random.default_rng(_stable_seed(int(seed), str(sample_key), "affine", 0, "shared"))
    u_angle, u_scale, u_x, u_y = rng.random(4)
    fraction = level / ranges.levels
    angle = fraction * ranges.max_rotation_deg * (2.0 * u_angle - 1.0)
    scale_at_full = ranges.scale_low + (ranges.scale_high - ranges.scale_low) * u_scale
    scale = 1.0 + fraction * (scale_at_full - 1.0)
    tx = fraction * ranges.max_translation_px * (2.0 * u_x - 1.0)
    ty = fraction * ranges.max_translation_px * (2.0 * u_y - 1.0)
    return float(angle), float(scale), float(tx), float(ty)


def perturbation_matrix(
    angle: float, scale: float, tx: float, ty: float, *, width: int, height: int
) -> np.ndarray:
    """The DiffRegCD construction: rotation and scale about the centre, then translation."""

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += [tx, ty]
    return matrix.astype(np.float64)


def perturb(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Perturbed image: content at p moves to matrix @ p; uncovered pixels are zero."""

    height, width = image.shape[:2]
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def align(perturbed: np.ndarray, matrix: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    """Sample a perturbed image back into the T1 frame: out(p) = perturbed(matrix @ p)."""

    height, width = perturbed.shape[:2]
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        perturbed, matrix, (width, height),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


# Element-wise arithmetic only: on Windows the MKL build of numpy aborts in
# BLAS once torch has loaded its OpenMP runtime (see rs_cd.change_density).


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    return np.stack(
        [
            matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2],
            matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2],
        ],
        axis=1,
    )


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """2x3 matrix of p -> outer(inner(p))."""

    result = np.empty((2, 3), dtype=np.float64)
    for row in range(2):
        for column in range(3):
            value = outer[row, 0] * inner[0, column] + outer[row, 1] * inner[1, column]
            result[row, column] = value + (outer[row, 2] if column == 2 else 0.0)
    return result


def _distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.hypot(first[:, 0] - second[:, 0], first[:, 1] - second[:, 1])


IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def grid_points(width: int, height: int, count: int) -> np.ndarray:
    xs = np.linspace(0.0, width - 1.0, count)
    ys = np.linspace(0.0, height - 1.0, count)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def transfer_error(
    estimate: np.ndarray, truth: np.ndarray, points: np.ndarray, *, width: int, height: int
) -> float | None:
    """Mean |estimate(p) - truth(p)| over grid points whose true image lies inside the crop."""

    target = apply(truth, points)
    inside = (
        (target[:, 0] >= 0.0) & (target[:, 0] <= width - 1.0)
        & (target[:, 1] >= 0.0) & (target[:, 1] <= height - 1.0)
    )
    if not inside.any():
        return None
    return float(_distances(apply(estimate, points[inside]), target[inside]).mean())


def mean_displacement(matrix: np.ndarray, points: np.ndarray) -> float:
    return float(_distances(apply(matrix, points), points).mean())


@dataclass(frozen=True)
class SimilarityEstimate:
    matrix: np.ndarray = field(default_factory=lambda: IDENTITY.copy())
    converged: bool = False
    inliers: int = 0
    inlier_fraction: float | None = None


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    reprojection_px: float,
    max_iters: int,
    confidence: float,
    refine_iters: int,
    min_inliers: int,
    seed: int,
) -> SimilarityEstimate:
    """RANSAC rotation + uniform scale + translation (4 DoF), OpenCV."""

    source = np.ascontiguousarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.ascontiguousarray(target, dtype=np.float64).reshape(-1, 2)
    if len(source) < max(int(min_inliers), 2):
        return SimilarityEstimate(inliers=len(source))
    cv2.setRNGSeed(int(seed))
    matrix, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC,
        ransacReprojThreshold=float(reprojection_px), maxIters=int(max_iters),
        confidence=float(confidence), refineIters=int(refine_iters),
    )
    inliers = int(mask.sum()) if mask is not None else 0
    if matrix is None or inliers < int(min_inliers) or not np.isfinite(matrix).all():
        return SimilarityEstimate(inliers=inliers)
    return SimilarityEstimate(
        matrix=matrix.astype(np.float64),
        converged=True,
        inliers=inliers,
        inlier_fraction=inliers / len(source),
    )


class RomaSimilarityLocalizer:
    """RoMa dense matches, sampled with a per-pair seed, fitted by RANSAC similarity."""

    def __init__(
        self,
        *,
        num_matches: int,
        reprojection_px: float,
        max_iters: int,
        confidence: float,
        refine_iters: int,
        min_inliers: int,
        device: torch.device,
        seed: int,
        model_kwargs: dict[str, Any] | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if model_factory is None:
            from romatch import roma_outdoor as model_factory
        self.num_matches = int(num_matches)
        self.fit = dict(
            reprojection_px=reprojection_px, max_iters=max_iters, confidence=confidence,
            refine_iters=refine_iters, min_inliers=min_inliers,
        )
        self.device = device
        self.seed = int(seed)
        self.model = model_factory(device=device, **(model_kwargs or {}))

    @torch.inference_mode()
    def estimate(self, reference: np.ndarray, target: np.ndarray, *, sample_key: str) -> SimilarityEstimate:
        height, width = reference.shape[:2]
        warp, certainty = self.model.match(
            Image.fromarray(np.ascontiguousarray(reference)),
            Image.fromarray(np.ascontiguousarray(target)),
            device=self.device,
        )
        pair_seed = _stable_seed(self.seed, sample_key, "p25", 0, "roma")
        torch.manual_seed(pair_seed)
        matches, _ = self.model.sample(warp, certainty, num=self.num_matches)
        points_a, points_b = self.model.to_pixel_coordinates(matches, height, width, height, width)
        # RoMa pixel coordinates put pixel edges at integers; OpenCV puts centres there.
        source = points_a.reshape(-1, 2).float().cpu().numpy() - 0.5
        destination = points_b.reshape(-1, 2).float().cpu().numpy() - 0.5
        return fit_similarity(source, destination, seed=pair_seed, **self.fit)


# --------------------------------------------------------------------------
# Step 1 summaries
# --------------------------------------------------------------------------


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(1 for flag in flags if flag) / len(flags) if flags else None


def _median(values: Sequence[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return bool(value)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return None if math.isnan(number) else number


def is_success(row: dict[str, Any], tolerance: float) -> bool:
    """Converged, and within tolerance of the perturbation either directly or
    after composing with the same localizer's clean estimate (the crop's own
    residual misregistration, main record 188.2)."""

    if not _as_bool(row["converged"]):
        return False
    errors = [_as_float(row["ate_raw"]), _as_float(row["ate_diff"])]
    errors = [error for error in errors if error is not None]
    return bool(errors) and min(errors) <= tolerance + 1e-9


def dataset_summary(rows: Sequence[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(settings["success_ate_px"])
    levels: dict[str, Any] = {}
    for level in sorted({int(row["level"]) for row in rows}):
        members = [row for row in rows if int(row["level"]) == level]
        entry = {
            "n": len(members),
            "converged_rate": _rate([_as_bool(row["converged"]) for row in members]),
            "trigger_rate": _rate([_as_bool(row["trigger"]) for row in members]),
            "median_perturbation_px": _median([_as_float(row["perturbation_px"]) for row in members]),
            "median_ate_raw": _median([_as_float(row["ate_raw"]) for row in members]),
            "median_ate_diff": _median([_as_float(row["ate_diff"]) for row in members]),
            "median_seconds": _median([_as_float(row["seconds"]) for row in members]),
        }
        if level == CLEAN_LEVEL:
            entry["median_clean_displacement_px"] = _median(
                [_as_float(row["estimate_displacement_px"]) for row in members if _as_bool(row["converged"])]
            )
        else:
            entry["success_rate"] = _rate([is_success(row, tolerance) for row in members])
        levels[str(level)] = entry
    binding = [str(level) for level in settings["binding_levels"]]
    missing = [level for level in binding + [str(CLEAN_LEVEL)] if level not in levels]
    if missing:
        raise KeyError(f"step 1 rows are missing levels {missing}")
    tracks = all(levels[level]["success_rate"] >= float(settings["tracking_rate_min"]) for level in binding)
    clean_trigger = levels[str(CLEAN_LEVEL)]["trigger_rate"]
    top = str(max(int(level) for level in levels))
    bins = []
    for index, (low, high) in enumerate(DEFAULT_BINS):
        in_bin = [
            row for row in rows
            if _as_float(row.get("change_fraction")) is not None
            and assign_bin(float(row["change_fraction"]), DEFAULT_BINS) == index
        ]
        bins.append(
            {
                "range": [low, high],
                "crops": sum(1 for row in in_bin if int(row["level"]) == CLEAN_LEVEL),
                "top_level_success": _rate(
                    [is_success(row, tolerance) for row in in_bin if str(row["level"]) == top]
                ),
                "clean_trigger_rate": _rate(
                    [_as_bool(row["trigger"]) for row in in_bin if int(row["level"]) == CLEAN_LEVEL]
                ),
            }
        )
    return {
        "levels": levels,
        "tracks": tracks,
        "clean_trigger_rate": clean_trigger,
        "clean_passed": clean_trigger <= float(settings["clean_trigger_max"]),
        "verdict": "proceed_to_step2" if tracks else "stop_localizer_insufficient",
        "change_bins": bins,
    }


def selfcheck_passed(rows: Sequence[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    errors = []
    for row in rows:
        error = _as_float(row["ate_raw"])
        errors.append(error if _as_bool(row["converged"]) and error is not None else math.inf)
    worst = max(errors, default=math.inf)
    return {"n": len(rows), "max_ate_raw": worst, "tolerance_px": tolerance,
            "passed": bool(rows) and worst <= tolerance + 1e-9}


def crop_origins(tile_size: int, crop_size: int) -> list[tuple[int, int]]:
    if tile_size % crop_size:
        raise ValueError(f"tile {tile_size} is not a multiple of the crop {crop_size}")
    steps = range(0, tile_size, crop_size)
    return [(top, left) for top in steps for left in steps]
