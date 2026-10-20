"""Localizers compared with the P27 localizer (docs/P27_EXPERIMENTS_AMENDMENT.md).

Every entry exposes ``estimate_many(references, targets)`` returning
``LocalizerEstimate`` rows, so the F1-step harness (tools/run_p27_f1.py) runs
them unchanged.

* Matching baselines (RoMa, SIFT) fit rotation + uniform scale + translation
  with RANSAC (the P25 fit). Their "misaligned" decision is the P25 rule: the
  estimate moves the 17 x 17 grid by more than 4 px on average.
* P27 variants change the frozen localizer at inference time only: fewer fine
  iterations (0 = coarse matching alone), or no reliability weighting (the
  change head is replaced by zeros, so every correspondence keeps its
  confidence weight).
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

from rs_cd.affine_registration import SimilarityEstimate, fit_similarity, grid_points, mean_displacement
from rs_cd.localizer_training import LocalizerEstimate, TrainedLocalizer
from rs_cd.robustness import _stable_seed

Estimator = Callable[[np.ndarray, np.ndarray, str], SimilarityEstimate]


def pair_key(reference: np.ndarray, target: np.ndarray) -> str:
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(reference).tobytes())
    digest.update(np.ascontiguousarray(target).tobytes())
    return digest.hexdigest()[:24]


class SimilarityBaseline:
    """A matching estimator plus the P25 displacement trigger."""

    def __init__(self, estimator: Estimator, *, trigger_px: float, size: int = 256, grid_count: int = 17) -> None:
        self.estimator = estimator
        self.trigger_px = float(trigger_px)
        self.points = grid_points(size, size, grid_count)

    def estimate_many(self, references: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> list[LocalizerEstimate]:
        rows = []
        for reference, target in zip(references, targets):
            estimate = self.estimator(reference, target, pair_key(reference, target))
            matrix = np.asarray(estimate.matrix, dtype=np.float64)
            moved = mean_displacement(matrix, self.points) > self.trigger_px
            rows.append(LocalizerEstimate(
                matrix=matrix, converged=bool(estimate.converged), inliers=int(estimate.inliers),
                inlier_fraction=estimate.inlier_fraction,
                misaligned_probability=1.0 if estimate.converged and moved else 0.0,
            ))
        return rows


class SiftSimilarity:
    """OpenCV SIFT, Lowe ratio test, and the P25 RANSAC similarity fit."""

    def __init__(self, *, seed: int, reprojection_px: float, max_iters: int, confidence: float,
                 refine_iters: int, min_inliers: int, max_features: int = 4000, ratio: float = 0.75) -> None:
        self.sift = cv2.SIFT_create(nfeatures=int(max_features))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = float(ratio)
        self.seed = int(seed)
        self.fit = dict(reprojection_px=reprojection_px, max_iters=max_iters, confidence=confidence,
                        refine_iters=refine_iters, min_inliers=min_inliers)

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
        return fit_similarity(source, destination, seed=_stable_seed(self.seed, key, "p27", 0, "sift"), **self.fit)


def roma_estimator(roma: Any) -> Estimator:
    return lambda reference, target, key: roma.estimate(reference, target, sample_key=key)


def p27_variant(model: torch.nn.Module, device: torch.device, *, fine_iterations: int | None = None,
                reliability_weighting: bool = True) -> TrainedLocalizer:
    """The frozen P27 localizer with an inference-time change (the model is modified in place)."""

    if fine_iterations is not None:
        if not 0 <= int(fine_iterations) <= int(model.fine_iterations):
            raise ValueError("fine iterations can only be reduced")
        model.fine_iterations = int(fine_iterations)
    if not reliability_weighting:
        model.change_head.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
    return TrainedLocalizer(model, device)
