"""P33 reviewer experiments: classical baselines, the estimate cache, the retrained-module switch, the control."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from p27_helpers import TinyBackbone
from rs_cd.affine_registration import SimilarityEstimate, affine_parameters, grid_points, perturbation_matrix
from rs_cd.estimate_cache import DiskCachedEstimator
from rs_cd.global_registration_baselines import EccScaleTranslation, FourierMellinScaleTranslation
from rs_cd.models.change_aware_similarity_localizer import build_localizer
from rs_cd.p30_geometry import BorderFreeGeometry
from rs_cd.p33_protocol import INTERP_LEVEL, INTERP_RANGES

SIZE = 256
POINTS = grid_points(SIZE, SIZE, 17)


def texture_tile(size: int = 448, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), np.float32)
    for sigma, weight in ((1.5, 0.3), (4.0, 0.5), (12.0, 1.0), (30.0, 1.0)):
        layer = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), sigma)
        image += weight * layer / layer.std()
    image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def grid_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    difference = estimate - truth
    return float(np.mean(np.linalg.norm(POINTS @ difference[:, :2].T + difference[:, 2], axis=1)))


@pytest.mark.parametrize("scale,tx,ty", [(1.08, 5.5, -7.25), (0.92, -12.0, 9.0), (1.0, 0.0, 0.0)])
@pytest.mark.parametrize("estimator,tolerance", [(FourierMellinScaleTranslation(size=SIZE), 1.0),
                                                 (EccScaleTranslation(size=SIZE), 0.3)])
def test_classical_baselines_recover_a_known_transform(estimator, tolerance, scale, tx, ty):
    tile = texture_tile()
    top = left = 96
    reference = np.ascontiguousarray(tile[top:top + SIZE, left:left + SIZE])
    truth = perturbation_matrix(0.0, scale, tx, ty, width=SIZE, height=SIZE)
    target = BorderFreeGeometry(tile, top, left, SIZE).perturb(truth)
    estimate = estimator(reference, target, "key")
    assert estimate.converged
    assert grid_error(estimate.matrix, truth) < tolerance


def test_disk_cache_memoises_and_survives_a_torn_line(tmp_path):
    calls = []

    def estimator(reference, target, key):
        calls.append(key)
        return SimilarityEstimate(matrix=np.array([[1.01, 0.0, 2.5], [0.0, 1.01, -1.25]]), converged=True,
                                  inliers=17, inlier_fraction=0.4)

    path = tmp_path / "cache.jsonl"
    cached = DiskCachedEstimator(estimator, path)
    image = np.zeros((4, 4, 3), np.uint8)
    first = cached(image, image, "a")
    again = cached(image, image, "a")
    assert calls == ["a"] and cached.hits == 1 and cached.misses == 1
    assert np.array_equal(first.matrix, again.matrix) and again.inliers == 17
    cached.close()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "torn", "matr')
    reopened = DiskCachedEstimator(estimator, path)
    assert len(reopened) == 1
    replayed = reopened(image, image, "a")
    assert calls == ["a"] and np.array_equal(replayed.matrix, first.matrix)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["key"] == "a"


def tiny_localizer(use_change_weights: bool):
    torch.manual_seed(1)
    return build_localizer(dict(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1,
                                use_change_weights=use_change_weights), backbone=TinyBackbone()).eval()


def test_confidence_only_switch_ignores_the_change_head():
    torch.manual_seed(0)
    reference, target = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    model = tiny_localizer(False)
    with torch.no_grad():
        out = model(reference, target)
        assert torch.allclose(out["coarse_weights"], out["coarse_confidence"])
        for parameter in model.change_head.parameters():
            parameter.add_(torch.randn_like(parameter))
        moved = model(reference, target)
    assert torch.allclose(out["matrix"], moved["matrix"]) and torch.allclose(out["misaligned_logit"], moved["misaligned_logit"])
    default = tiny_localizer(True)
    with torch.no_grad():
        weighted = default(reference, target)
    assert not torch.allclose(weighted["coarse_weights"], weighted["coarse_confidence"])
    assert default.use_change_weights and build_localizer(dict(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8),
                                                          backbone=TinyBackbone()).use_change_weights


def test_interpolation_control_is_sub_pixel():
    largest = 0.0
    for index in range(200):
        angle, scale, tx, ty = affine_parameters(f"crop{index}", INTERP_LEVEL, INTERP_RANGES, 3407)
        assert angle == 0.0 and abs(tx) <= 0.5 + 1e-9 and abs(ty) <= 0.5 + 1e-9 and abs(scale - 1.0) <= 0.005 + 1e-9
        matrix = perturbation_matrix(angle, scale, tx, ty, width=SIZE, height=SIZE)
        largest = max(largest, float(np.abs(POINTS @ (matrix[:, :2] - np.eye(2)).T + matrix[:, 2]).max()))
    assert largest < 1.25


def test_ablation_builds_the_classical_baselines():
    from rs_cd.p27_baseline_localizers import SimilarityBaseline
    from tools.p27_localizer_ablation import build

    evaluation = {"localizer": {"trigger_mean_displacement_px": 4.0}, "perturbation": {"crop_size": SIZE}}
    for variant in ("phasecorr_fm", "ecc_ts"):
        localizer = build(variant, None, evaluation, torch.device("cpu"))
        assert isinstance(localizer, SimilarityBaseline)
        tile = texture_tile(seed=2)
        reference = np.ascontiguousarray(tile[96:96 + SIZE, 96:96 + SIZE])
        truth = perturbation_matrix(0.0, 1.05, 6.0, 3.0, width=SIZE, height=SIZE)
        target = BorderFreeGeometry(tile, 96, 96, SIZE).perturb(truth)
        estimate = localizer.estimate_many([reference], [target])[0]
        assert estimate.converged and estimate.misaligned_probability == 1.0
        assert grid_error(estimate.matrix, truth) < 1.0
