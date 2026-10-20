"""Translation + isotropic scale fits for the matching-baseline reviewer check (rs_cd.translation_scale_baselines)."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from rs_cd.affine_registration import RomaSimilarityLocalizer, apply, grid_points, perturbation_matrix, transfer_error
from rs_cd.translation_scale_baselines import (
    RomaMatchSampler,
    RomaTranslationScale,
    SiftTranslationScale,
    fit_translation_scale_huber,
    fit_translation_scale_ransac,
    weighted_scale_translation,
)

FIT = dict(reprojection_px=3.0, max_iters=2000, confidence=0.999, refine_iters=10, min_inliers=8)
TRUTH = perturbation_matrix(0.0, 0.85, 11.0, -6.0, width=256, height=256)   # no rotation: the 3-DoF family


def _matches(truth: np.ndarray, *, count: int = 2000, outliers: float = 0.3, noise: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    source = rng.random((count, 2)) * 255
    target = apply(truth, source) + rng.normal(0.0, noise, (count, 2))
    bad = rng.random(count) < outliers
    target[bad] = rng.random((int(bad.sum()), 2)) * 255
    return source, target, bad


def test_weighted_least_squares_recovers_an_exact_scale_and_translation() -> None:
    source, target, _ = _matches(TRUTH, outliers=0.0, noise=0.0)
    scale, translation = weighted_scale_translation(source, target, np.ones(len(source)))
    assert np.allclose([scale, *translation], [TRUTH[0, 0], *TRUTH[:, 2]], atol=1e-9)


def test_ransac_recovers_the_transformation_with_many_outliers_and_no_rotation() -> None:
    source, target, bad = _matches(TRUTH, outliers=0.4)
    estimate = fit_translation_scale_ransac(source, target, seed=1, **FIT)
    assert estimate.converged
    assert estimate.matrix[0, 1] == 0.0 and estimate.matrix[1, 0] == 0.0
    assert estimate.matrix[0, 0] == estimate.matrix[1, 1]
    points = grid_points(256, 256, 17)
    assert transfer_error(estimate.matrix, TRUTH, points, width=256, height=256) < 0.2
    assert abs(estimate.inlier_fraction - (1 - bad.mean())) < 0.05


def test_ransac_is_reproducible_and_reports_non_convergence() -> None:
    source, target, _ = _matches(TRUTH, outliers=0.4)
    first = fit_translation_scale_ransac(source, target, seed=7, **FIT)
    second = fit_translation_scale_ransac(source, target, seed=7, **FIT)
    assert np.array_equal(first.matrix, second.matrix)
    few = fit_translation_scale_ransac(source[:5], target[:5], seed=7, **FIT)
    assert not few.converged
    rng = np.random.default_rng(3)
    noise = fit_translation_scale_ransac(rng.random((300, 2)) * 255, rng.random((300, 2)) * 255, seed=7,
                                         **{**FIT, "reprojection_px": 0.01})
    assert not noise.converged


def test_huber_reweighting_recovers_the_transformation_with_moderate_outliers() -> None:
    source, target, bad = _matches(TRUTH, outliers=0.15)
    weights = np.where(bad, 0.2, 1.0)                     # RoMa certainty is lower on wrong matches
    estimate = fit_translation_scale_huber(source, target, weights)
    assert estimate.converged
    points = grid_points(256, 256, 17)
    assert transfer_error(estimate.matrix, TRUTH, points, width=256, height=256) < 0.5


class _FakeRoma:
    """RoMa-style edge-convention matches consistent with a known matrix (as in test_affine_registration)."""

    def __init__(self, truth: np.ndarray) -> None:
        self.truth = truth
        self.calls = 0

    def match(self, image_a, image_b, device=None):
        self.calls += 1
        width, height = image_a.size
        return torch.zeros(1), torch.tensor([float(width), float(height)])

    def sample(self, warp, certainty, num=5000):
        width, height = (float(v) for v in certainty)
        centres = torch.rand(num, 2, dtype=torch.float64) * torch.tensor([width - 1, height - 1], dtype=torch.float64)
        moved = torch.from_numpy(apply(self.truth, centres.numpy()))
        moved[: num // 10] = torch.rand(num // 10, 2, dtype=torch.float64) * 255
        edges_a, edges_b = centres + 0.5, moved + 0.5
        normalise = lambda p: torch.stack([p[:, 0] / width * 2 - 1, p[:, 1] / height * 2 - 1], 1)
        return torch.cat([normalise(edges_a), normalise(edges_b)], 1), torch.rand(num)

    def to_pixel_coordinates(self, matches, height_a, width_a, height_b, width_b):
        def pixels(coords, height, width):
            return torch.stack([width / 2 * (coords[:, 0] + 1), height / 2 * (coords[:, 1] + 1)], -1)

        return pixels(matches[:, :2], height_a, width_a), pixels(matches[:, 2:], height_b, width_b)


def test_both_fits_share_one_cached_roma_pass_and_recover_the_matrix() -> None:
    fake = _FakeRoma(TRUTH)
    base = RomaSimilarityLocalizer(num_matches=800, device=torch.device("cpu"), seed=3407,
                                   model_factory=lambda device, **kwargs: fake,
                                   reprojection_px=1.0, max_iters=2000, confidence=0.999, refine_iters=10, min_inliers=8)
    sampler = RomaMatchSampler(base)
    image = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    points = grid_points(256, 256, 17)
    for robust in ("ransac", "huber"):
        estimator = RomaTranslationScale(sampler, robust=robust, **FIT)
        first = estimator(image, image, "crop|L3")
        second = estimator(image, image, "crop|L3")
        assert first.converged and np.array_equal(first.matrix, second.matrix)
        tolerance = 1e-3 if robust == "ransac" else 3.0     # Huber keeps a bias from the 10 % uniform outliers
        assert transfer_error(first.matrix, TRUTH, points, width=256, height=256) < tolerance
    assert fake.calls == 1


def test_sift_translation_scale_recovers_a_shift_and_scale_of_a_textured_image() -> None:
    rng = np.random.default_rng(0)
    small = (rng.random((40, 40, 3)) * 255).astype(np.uint8)
    image = cv2.resize(small, (320, 320), interpolation=cv2.INTER_CUBIC)
    truth = perturbation_matrix(0.0, 1.05, 6.0, -4.0, width=256, height=256)
    # target(T(x)) = reference(x): warp the reference forward by the truth
    target = cv2.warpAffine(image, truth, (320, 320), flags=cv2.INTER_LINEAR)
    reference, target = image[32:288, 32:288].copy(), target[32:288, 32:288].copy()
    shifted = truth.copy()
    shifted[:, 2] += truth[:, :2] @ np.array([32.0, 32.0]) - np.array([32.0, 32.0])
    estimator = SiftTranslationScale(seed=3407, **FIT)
    estimate = estimator(reference, target, "crop|L3")
    assert estimate.converged
    assert estimate.matrix[0, 1] == 0.0 and estimate.matrix[1, 0] == 0.0
    points = grid_points(256, 256, 17)
    assert transfer_error(estimate.matrix, shifted, points, width=256, height=256) < 1.0
