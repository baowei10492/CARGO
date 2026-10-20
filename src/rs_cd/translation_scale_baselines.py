"""Matching baselines fitted with translation + isotropic scale (reviewer check, 2026-10-08).

The P27 baselines (rs_cd.affine_registration.RomaSimilarityLocalizer, rs_cd.p27_baseline_localizers.SiftSimilarity)
fit rotation + scale + translation (4 DoF) with RANSAC at 1 px. Reviewers asked for the transformation family of the
registration module (3 DoF, no rotation) and a fit that is not unfairly strict:

    RoMa, ransac   RANSAC over 2-point samples, 3 px inlier threshold, least-squares refinement on the inliers
    RoMa, huber    RoMa certainty as weights, weighted least squares (Eq. 5 of the paper), then Huber reweighting
                   w_i min(1, delta / r_i) for three rounds at 8 px and three at 2 px, as in the registration module
    SIFT, ransac   the P27 SIFT matches (4000 keypoints, ratio 0.75) with the same 3 px RANSAC fit

RoMa matches are sampled exactly as in RomaSimilarityLocalizer (same seed rule, same half-pixel shift) and cached per
pair key, so the second RoMa fit costs no extra RoMa pass. All matrices map reference (T1) pixels to target (T2) pixels.
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from rs_cd.affine_registration import SimilarityEstimate
from rs_cd.robustness import _stable_seed

SCALE_LIMITS = (0.25, 4.0)
HUBER_DELTAS = (8.0, 8.0, 8.0, 2.0, 2.0, 2.0)


def _matrix(scale: float, translation: np.ndarray) -> np.ndarray:
    return np.array([[scale, 0.0, translation[0]], [0.0, scale, translation[1]]], dtype=np.float64)


def weighted_scale_translation(source: np.ndarray, target: np.ndarray,
                               weights: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Closed-form weighted least squares for target = s * source + t (the registration module's Eq. 5)."""
    weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    total = weights.sum()
    if not np.isfinite(total) or total <= 1e-12:
        return None
    w = weights / total
    mean_source, mean_target = w @ source, w @ target
    centred_source, centred_target = source - mean_source, target - mean_target
    denominator = float(w @ (centred_source * centred_source).sum(1))
    if denominator <= 1e-12:
        return None
    scale = float(np.clip(w @ (centred_source * centred_target).sum(1) / denominator, *SCALE_LIMITS))
    return scale, mean_target - scale * mean_source


def _residuals(scale: float, translation: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(scale * source + translation - target, axis=1)


def fit_translation_scale_ransac(source: np.ndarray, target: np.ndarray, *, reprojection_px: float, max_iters: int,
                                 confidence: float, refine_iters: int, min_inliers: int, seed: int,
                                 batch: int = 256) -> SimilarityEstimate:
    """RANSAC for translation + isotropic scale (3 DoF) with 2-point samples and adaptive stopping."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    count = len(source)
    if count < max(int(min_inliers), 2):
        return SimilarityEstimate(inliers=count)
    rng = np.random.default_rng(int(seed))
    best_mask, best_count = None, 0
    needed, done = int(max_iters), 0
    while done < needed:
        size = min(int(batch), needed - done)
        first, second = rng.integers(0, count, size), rng.integers(0, count, size)
        step_source, step_target = source[second] - source[first], target[second] - target[first]
        norm = (step_source * step_source).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = (step_source * step_target).sum(1) / norm
        valid = (first != second) & (norm > 1e-9) & np.isfinite(scale)
        valid &= (scale >= SCALE_LIMITS[0]) & (scale <= SCALE_LIMITS[1])
        index = np.flatnonzero(valid)
        if index.size:
            centre_source = (source[first[index]] + source[second[index]]) / 2.0
            centre_target = (target[first[index]] + target[second[index]]) / 2.0
            translation = centre_target - scale[index, None] * centre_source
            predicted = scale[index, None, None] * source[None] + translation[:, None, :]
            inliers = np.linalg.norm(predicted - target[None], axis=2) <= float(reprojection_px)
            counts = inliers.sum(1)
            top = int(np.argmax(counts))
            if counts[top] > best_count:
                best_count, best_mask = int(counts[top]), inliers[top]
                share = (best_count / count) ** 2
                if share >= 1.0:
                    needed = done + size
                elif share > 0.0:
                    needed = min(int(max_iters), int(np.ceil(np.log(1.0 - confidence) / np.log(1.0 - share))))
        done += size
    if best_mask is None or best_count < int(min_inliers):
        return SimilarityEstimate(inliers=best_count)
    fit = weighted_scale_translation(source[best_mask], target[best_mask], np.ones(best_count))
    if fit is None:
        return SimilarityEstimate(inliers=best_count)
    for _ in range(int(refine_iters)):
        mask = _residuals(*fit, source, target) <= float(reprojection_px)
        if mask.sum() < int(min_inliers) or np.array_equal(mask, best_mask):
            break
        refit = weighted_scale_translation(source[mask], target[mask], np.ones(int(mask.sum())))
        if refit is None:
            break
        fit, best_mask = refit, mask
    inliers = int(best_mask.sum())
    return SimilarityEstimate(matrix=_matrix(*fit), converged=True, inliers=inliers, inlier_fraction=inliers / count)


def fit_translation_scale_huber(source: np.ndarray, target: np.ndarray, weights: np.ndarray, *,
                                deltas: Sequence[float] = HUBER_DELTAS, min_points: int = 8) -> SimilarityEstimate:
    """Weighted least squares, then Huber reweighting as in the registration module's robust fit."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    base = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    count = len(source)
    if count < max(int(min_points), 2):
        return SimilarityEstimate(inliers=count)
    fit = weighted_scale_translation(source, target, base)
    if fit is None:
        return SimilarityEstimate(inliers=count)
    for delta in deltas:
        reweighted = base * np.minimum(1.0, float(delta) / np.maximum(_residuals(*fit, source, target), 1e-6))
        refit = weighted_scale_translation(source, target, reweighted)
        if refit is None:
            break
        fit = refit
    inliers = int((_residuals(*fit, source, target) <= float(deltas[-1])).sum())
    return SimilarityEstimate(matrix=_matrix(*fit), converged=True, inliers=inliers, inlier_fraction=inliers / count)


class RomaMatchSampler:
    """RoMa matches sampled as in RomaSimilarityLocalizer.estimate, cached per pair key.

    ``base`` is a RomaSimilarityLocalizer, which carries the model, the number of matches, the seed and the device.
    """

    def __init__(self, base: Any) -> None:
        self.base = base
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

    @torch.inference_mode()
    def sample(self, reference: np.ndarray, target: np.ndarray, *, sample_key: str):
        hit = self.cache.get(sample_key)
        if hit is not None:
            return hit
        height, width = reference.shape[:2]
        model = self.base.model
        warp, certainty = model.match(
            Image.fromarray(np.ascontiguousarray(reference)),
            Image.fromarray(np.ascontiguousarray(target)),
            device=self.base.device,
        )
        pair_seed = _stable_seed(self.base.seed, sample_key, "p25", 0, "roma")
        torch.manual_seed(pair_seed)
        matches, match_certainty = model.sample(warp, certainty, num=self.base.num_matches)
        points_a, points_b = model.to_pixel_coordinates(matches, height, width, height, width)
        # RoMa pixel coordinates put pixel edges at integers; the fits use pixel centres, as OpenCV does.
        source = points_a.reshape(-1, 2).float().cpu().numpy() - 0.5
        destination = points_b.reshape(-1, 2).float().cpu().numpy() - 0.5
        weights = match_certainty.reshape(-1).float().cpu().numpy()
        entry = (source.astype(np.float32), destination.astype(np.float32), weights.astype(np.float32), int(pair_seed))
        self.cache[sample_key] = entry
        return entry


class RomaTranslationScale:
    """An estimator (reference, target, key) -> SimilarityEstimate on shared RoMa matches."""

    def __init__(self, sampler: RomaMatchSampler, *, robust: str, reprojection_px: float, max_iters: int,
                 confidence: float, refine_iters: int, min_inliers: int,
                 huber_deltas: Sequence[float] = HUBER_DELTAS) -> None:
        if robust not in ("ransac", "huber"):
            raise ValueError(f"unknown robust fit {robust!r}")
        self.sampler = sampler
        self.robust = robust
        self.ransac = dict(reprojection_px=float(reprojection_px), max_iters=int(max_iters),
                           confidence=float(confidence), refine_iters=int(refine_iters), min_inliers=int(min_inliers))
        self.huber_deltas = tuple(float(d) for d in huber_deltas)
        self.min_points = int(min_inliers)

    def __call__(self, reference: np.ndarray, target: np.ndarray, key: str) -> SimilarityEstimate:
        source, destination, weights, pair_seed = self.sampler.sample(reference, target, sample_key=key)
        if self.robust == "ransac":
            return fit_translation_scale_ransac(source, destination, seed=pair_seed, **self.ransac)
        return fit_translation_scale_huber(source, destination, weights, deltas=self.huber_deltas,
                                           min_points=self.min_points)


class SiftTranslationScale:
    """The P27 SIFT matching (rs_cd.p27_baseline_localizers.SiftSimilarity) with the 3-DoF RANSAC fit."""

    def __init__(self, *, seed: int, reprojection_px: float, max_iters: int, confidence: float, refine_iters: int,
                 min_inliers: int, max_features: int = 4000, ratio: float = 0.75) -> None:
        self.sift = cv2.SIFT_create(nfeatures=int(max_features))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = float(ratio)
        self.seed = int(seed)
        self.fit = dict(reprojection_px=float(reprojection_px), max_iters=int(max_iters), confidence=float(confidence),
                        refine_iters=int(refine_iters), min_inliers=int(min_inliers))

    def __call__(self, reference: np.ndarray, target: np.ndarray, key: str) -> SimilarityEstimate:
        first = self.sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
        second = self.sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), None)
        if first[1] is None or second[1] is None or len(first[0]) < 2 or len(second[0]) < 2:
            return SimilarityEstimate()
        good = [pair[0] for pair in self.matcher.knnMatch(first[1], second[1], k=2)
                if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance]
        # OpenCV keypoints already put pixel centres at integer coordinates.
        source = np.array([first[0][m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
        destination = np.array([second[0][m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
        return fit_translation_scale_ransac(source, destination, seed=_stable_seed(self.seed, key, "p27", 0, "sift"),
                                            **self.fit)
