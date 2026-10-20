"""P25: affine perturbation, similarity registration and the step 1 rules."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    IDENTITY,
    AffineRanges,
    RomaSimilarityLocalizer,
    affine_parameters,
    align,
    apply,
    compose,
    crop_origins,
    dataset_summary,
    fit_similarity,
    grid_points,
    is_success,
    mean_displacement,
    perturb,
    perturbation_matrix,
    selfcheck_passed,
    transfer_error,
)

ROOT = Path(__file__).resolve().parents[1]
RANGES = AffineRanges()
FIT = dict(reprojection_px=1.0, max_iters=2000, confidence=0.999, refine_iters=10, min_inliers=8)


def test_levels_are_nested_and_level_three_spans_the_published_ranges() -> None:
    draws = [affine_parameters(f"crop{index}", 3, RANGES, 3407) for index in range(400)]
    angles, scales, xs, ys = (np.array(values) for values in zip(*draws))
    assert angles.min() >= -30 and angles.max() <= 30 and angles.max() - angles.min() > 50
    assert scales.min() >= 0.80 and scales.max() <= 1.25 and scales.min() < 0.83 and scales.max() > 1.22
    assert np.abs(xs).max() <= 25 and np.abs(ys).max() <= 25
    full = affine_parameters("crop7", 3, RANGES, 3407)
    third = affine_parameters("crop7", 1, RANGES, 3407)
    assert third[0] == pytest.approx(full[0] / 3)
    assert third[1] - 1 == pytest.approx((full[1] - 1) / 3)
    assert third[2:] == pytest.approx((full[2] / 3, full[3] / 3))
    assert affine_parameters("crop7", CLEAN_LEVEL, RANGES, 3407) == (0.0, 1.0, 0.0, 0.0)
    assert affine_parameters("crop7", 2, RANGES, 3407) == affine_parameters("crop7", 2, RANGES, 3407)
    assert affine_parameters("crop8", 3, RANGES, 3407) != full
    with pytest.raises(ValueError):
        affine_parameters("crop7", 4, RANGES, 3407)


def test_matrix_is_the_diffregcd_construction() -> None:
    matrix = perturbation_matrix(20.0, 1.1, 5.0, -3.0, width=256, height=256)
    expected = cv2.getRotationMatrix2D((128.0, 128.0), 20.0, 1.1)
    expected[:, 2] += [5.0, -3.0]
    assert matrix == pytest.approx(expected)


def _texture(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = (rng.random((size // 8, size // 8, 3)) * 255).astype(np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def test_perturbation_moves_content_by_the_matrix_and_align_undoes_it() -> None:
    image = _texture()
    matrix = perturbation_matrix(0.0, 1.0, 7.0, -4.0, width=256, height=256)
    moved = perturb(image, matrix)
    # Content at (x=60, y=50) moves by (+7, -4) to (x=67, y=46).
    assert np.array_equal(moved[50 - 4, 60 + 7], image[50, 60])
    restored = align(moved, matrix)
    assert np.array_equal(restored[20:200, 20:200], image[20:200, 20:200])
    rotated = perturbation_matrix(25.0, 1.15, 3.0, 2.0, width=256, height=256)
    back = align(perturb(image, rotated), rotated).astype(float)
    inner = slice(90, 166)
    assert np.abs(back[inner, inner] - image[inner, inner]).mean() < 6.0


def test_compose_and_transfer_errors() -> None:
    points = grid_points(256, 256, 17)
    outer = perturbation_matrix(10.0, 0.9, 2.0, 1.0, width=256, height=256)
    inner = perturbation_matrix(0.0, 1.0, 1.5, -0.5, width=256, height=256)
    assert apply(compose(outer, inner), points) == pytest.approx(apply(outer, apply(inner, points)))
    assert transfer_error(outer, outer, points, width=256, height=256) == pytest.approx(0.0)
    shifted = outer.copy()
    shifted[:, 2] += [3.0, 4.0]
    assert transfer_error(shifted, outer, points, width=256, height=256) == pytest.approx(5.0)
    away = perturbation_matrix(0.0, 1.0, 1000.0, 0.0, width=256, height=256)
    assert transfer_error(IDENTITY, away, points, width=256, height=256) is None
    assert mean_displacement(IDENTITY, points) == 0.0


def test_similarity_fit_recovers_the_matrix_despite_outliers() -> None:
    rng = np.random.default_rng(1)
    truth = perturbation_matrix(-22.0, 1.2, -9.0, 14.0, width=256, height=256)
    source = rng.random((300, 2)) * 255
    target = apply(truth, source) + rng.normal(0.0, 0.2, (300, 2))
    target[:60] = rng.random((60, 2)) * 255
    estimate = fit_similarity(source, target, seed=5, **FIT)
    assert estimate.converged and estimate.inliers >= 230
    points = grid_points(256, 256, 17)
    assert transfer_error(estimate.matrix, truth, points, width=256, height=256) < 0.2
    again = fit_similarity(source, target, seed=5, **FIT)
    assert np.array_equal(again.matrix, estimate.matrix)
    few = fit_similarity(source[:5], target[:5], seed=5, **FIT)
    assert not few.converged and np.array_equal(few.matrix, IDENTITY)


class _FakeRoma:
    """Returns RoMa-style edge-convention matches consistent with a known matrix."""

    def __init__(self, truth: np.ndarray) -> None:
        self.truth = truth
        self.kwargs = None

    def match(self, image_a, image_b, device=None):
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


def test_roma_adapter_handles_the_half_pixel_convention_and_is_reproducible() -> None:
    truth = perturbation_matrix(28.0, 0.85, 11.0, -6.0, width=256, height=256)
    fake = _FakeRoma(truth)
    received = {}

    def factory(device, **kwargs):
        received.update(kwargs)
        return fake

    localizer = RomaSimilarityLocalizer(
        num_matches=800, device=torch.device("cpu"), seed=3407,
        model_kwargs={"use_custom_corr": True}, model_factory=factory, **{
            "reprojection_px": 1.0, "max_iters": 2000, "confidence": 0.999,
            "refine_iters": 10, "min_inliers": 8,
        },
    )
    image = _texture()
    first = localizer.estimate(image, image, sample_key="crop|L3")
    second = localizer.estimate(image, image, sample_key="crop|L3")
    assert received == {"use_custom_corr": True}
    assert first.converged
    assert np.array_equal(first.matrix, second.matrix)
    points = grid_points(256, 256, 17)
    assert transfer_error(first.matrix, truth, points, width=256, height=256) < 1e-3


def _row(level, *, converged=True, raw=None, diff=None, trigger=False, change=0.02, displacement=0.5):
    return {
        "level": level, "converged": converged, "ate_raw": raw, "ate_diff": diff,
        "trigger": trigger, "change_fraction": change, "perturbation_px": 10.0 * level,
        "seconds": 0.3, "estimate_displacement_px": displacement,
    }


def test_success_uses_the_smaller_of_raw_and_differential_error() -> None:
    assert is_success(_row(3, raw=1.5), 2.0)
    assert is_success(_row(3, raw=3.1, diff=0.4), 2.0)
    assert not is_success(_row(3, raw=3.1, diff=2.5), 2.0)
    assert not is_success(_row(3, raw=0.1, converged=False), 2.0)
    assert not is_success(_row(3, raw=None, diff=None), 2.0)
    assert is_success(_row(3, raw="1.0", diff=""), 2.0)


SETTINGS = {"success_ate_px": 2.0, "tracking_rate_min": 0.8, "binding_levels": [2, 3], "clean_trigger_max": 0.1}


def test_dataset_summary_tracks_only_if_both_binding_levels_pass() -> None:
    rows = []
    for index in range(10):
        change = 0.01 if index < 5 else 0.3
        rows.append(_row(0, raw=0.5, trigger=index == 9, change=change))
        rows.append(_row(1, raw=5.0 if index < 5 else 0.5, change=change))
        rows.append(_row(2, raw=0.5 if index < 8 else 9.0, change=change))
        rows.append(_row(3, raw=0.5 if index < 9 else 9.0, change=change))
    summary = dataset_summary(rows, SETTINGS)
    assert summary["tracks"] and summary["verdict"] == "proceed_to_step2"
    assert summary["levels"]["1"]["success_rate"] == 0.5
    assert summary["clean_trigger_rate"] == pytest.approx(0.1) and summary["clean_passed"]
    assert summary["levels"]["0"]["median_clean_displacement_px"] == 0.5
    assert summary["change_bins"][0]["crops"] == 5 and summary["change_bins"][0]["top_level_success"] == 1.0
    assert summary["change_bins"][3]["top_level_success"] == pytest.approx(0.8)
    rows[2]["ate_raw"] = 9.0  # crop 0, level 2: one more level-2 failure
    failing = dataset_summary(rows, SETTINGS)
    assert failing["levels"]["2"]["success_rate"] == pytest.approx(0.7)
    assert failing["verdict"] == "stop_localizer_insufficient"
    with pytest.raises(KeyError):
        dataset_summary([row for row in rows if row["level"] != 3], SETTINGS)


def test_selfcheck_requires_every_crop() -> None:
    good = [{"converged": True, "ate_raw": 0.0}, {"converged": True, "ate_raw": 0.9}]
    assert selfcheck_passed(good, 1.0)["passed"]
    assert not selfcheck_passed(good + [{"converged": False, "ate_raw": 0.1}], 1.0)["passed"]
    assert not selfcheck_passed(good + [{"converged": True, "ate_raw": 1.2}], 1.0)["passed"]
    assert not selfcheck_passed([], 1.0)["passed"]


def test_crop_grid() -> None:
    assert crop_origins(512, 256) == [(0, 0), (0, 256), (256, 0), (256, 256)]
    assert len(crop_origins(1024, 256)) == 16
    with pytest.raises(ValueError):
        crop_origins(500, 256)


def test_p25_config_matches_the_protocol() -> None:
    config = yaml.safe_load((ROOT / "configs/p25_affine_protocol.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "preregistered"
    assert (ROOT / config["protocol_doc"]).is_file()
    perturbation = config["perturbation"]
    assert (perturbation["crop_size"], perturbation["max_rotation_deg"], perturbation["max_translation_px"]) == (256, 30.0, 25.0)
    assert perturbation["scale_range"] == [0.80, 1.25] and perturbation["levels"] == [1, 2, 3]
    assert [spec["name"] for spec in config["datasets"]] == ["LEVIR-CD", "DSIFN-CD", "WHU-CD"]
    assert all(spec["split"] == "val" for spec in config["datasets"])
    assert all(spec["tile_size"] % perturbation["crop_size"] == 0 for spec in config["datasets"])
    assert config["localizer"]["fused_local_corr_version"] == "0.1.1"


def test_step1_tool_refuses_test_splits(tmp_path) -> None:
    from tools.run_p25_affine_step1 import load_config

    config = yaml.safe_load((ROOT / "configs/p25_affine_protocol.yaml").read_text(encoding="utf-8"))
    config["datasets"][0]["split"] = "test"
    path = tmp_path / "p25.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="validation"):
        load_config(path)


def test_selfcheck_crops_prefer_textured_crops() -> None:
    from tools.run_p25_affine_step1 import selfcheck_crops

    class Cache:
        def crop(self, item, size):
            value = np.full((size, size, 3), 100, dtype=np.uint8)
            if item["textured"]:
                value[::2] = 0
            return value, value, value[..., 0]

    crops = [{"textured": index in (4, 7, 9)} for index in range(12)]
    chosen = selfcheck_crops(crops, Cache(), 8, {"selfcheck_pool": 10, "selfcheck_crops": 3})
    assert chosen == [crops[4], crops[7], crops[9]]
