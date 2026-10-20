"""P23: published localizers behind one interface, and the screen rules.

Protocol: docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md. The kornia and RoMa
adapters are exercised with stand-in libraries: the real ones are installed
and self-checked on the server before any screening row is written.
"""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rs_cd.physical_shift import CLEAN_KEY
from rs_cd.registration_controls import phase_correlation_offset
from rs_cd.robustness import _translate_reflect
from rs_cd.strong_localizers import (
    PHYSICAL_KEYS,
    LightGlueDiskLocalizer,
    PhaseCorrelationLocalizer,
    RomaOutdoorLocalizer,
    SiftLocalizer,
    TranslationEstimate,
    anchor_offsets,
    applied_offset,
    candidate_verdict,
    change_bin_summary,
    clean_agreement,
    compare_f1_tables,
    scenario_metrics,
    select_candidate,
    selfcheck_summary,
    translation_consensus,
    triggered,
    windowed_phase_correlation,
)

ROOT = Path(__file__).resolve().parents[1]


def _texture(size: int = 96, seed: int = 0) -> torch.Tensor:
    """Unit RGB image with blobs and edges, smooth enough for sub-pixel peaks."""

    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(1, 3, size // 8, size // 8, generator=generator)
    image = torch.nn.functional.interpolate(noise, size=(size, size), mode="bicubic", align_corners=False)
    rectangles = torch.zeros(1, 1, size, size)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        w, h = rng.integers(max(3, size // 32), max(4, size // 12), size=2)
        x, y = rng.integers(0, size - max(w, h), size=2)
        rectangles[..., y : y + h, x : x + w] = float(rng.uniform(0.3, 1.0))
    return (0.5 * image + 0.5 * rectangles).clamp(0.0, 1.0)[0]


def _shift(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    return _translate_reflect(image, dx=dx, dy=dy)[0]


# --------------------------------------------------------------- consensus


def test_consensus_recovers_the_dominant_translation_despite_outliers() -> None:
    inliers = torch.tensor([[4.0, -2.0]]) + 0.1 * torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]]
    )
    outliers = torch.tensor([[30.0, 30.0], [-20.0, 5.0], [0.0, 0.0]])
    estimate = translation_consensus(
        torch.cat([outliers, inliers]), threshold=1.0, min_matches=3, max_abs_shift=None
    )
    assert estimate.converged
    assert estimate.dx == pytest.approx(4.0 + 0.1 / 6)
    assert estimate.dy == pytest.approx(-2.0 + 0.1 / 6)
    assert estimate.support == 6
    assert estimate.inlier_fraction == pytest.approx(6 / 9)


def test_consensus_breaks_ties_by_the_first_hypothesis() -> None:
    points = torch.tensor([[5.0, 5.0], [5.2, 5.0], [-5.0, -5.0], [-5.2, -5.0]])
    estimate = translation_consensus(points, threshold=0.5, min_matches=2, max_abs_shift=None)
    assert estimate.dx == pytest.approx(5.1)


def test_consensus_applies_the_search_window_before_counting() -> None:
    points = torch.tensor([[50.0, 0.0]] * 5 + [[2.0, 2.0]] * 3)
    estimate = translation_consensus(points, threshold=1.0, min_matches=3, max_abs_shift=10.0)
    assert (estimate.dx, estimate.dy) == pytest.approx((2.0, 2.0))
    failed = translation_consensus(points, threshold=1.0, min_matches=4, max_abs_shift=10.0)
    assert not failed.converged
    assert (failed.dx, failed.dy) == (0.0, 0.0)
    assert failed.support == 3


def test_consensus_caps_matches_in_the_given_order() -> None:
    points = torch.tensor([[1.0, 1.0]] * 3 + [[9.0, 9.0]] * 10)
    estimate = translation_consensus(
        points, threshold=0.5, min_matches=2, max_abs_shift=None, max_matches=4
    )
    assert (estimate.dx, estimate.dy) == pytest.approx((1.0, 1.0))


def test_triggered_prefers_the_candidates_own_gate() -> None:
    assert triggered(TranslationEstimate(0.0, 0.0, True, gate=True), 6.0)
    assert not triggered(TranslationEstimate(30.0, 0.0, True, gate=False), 6.0)
    assert not triggered(TranslationEstimate(4.0, 4.0, True), 6.0)
    assert triggered(TranslationEstimate(5.0, 4.0, True), 6.0)
    assert not triggered(TranslationEstimate(50.0, 0.0, False), 6.0)


def test_applied_offset_rounds_to_integer_pixels() -> None:
    estimate = TranslationEstimate(11.6, -0.4, True)
    assert applied_offset(estimate, rounded=True) == (12.0, -0.0)
    assert applied_offset(estimate, rounded=False) == (11.6, -0.4)


# ------------------------------------------------------- phase correlation


@pytest.mark.parametrize("shift", [(3, -2), (-5, 4), (0, 0)])
def test_windowed_phase_correlation_matches_the_reference_function(shift) -> None:
    reference = _texture(64, seed=1)
    target = _shift(reference, *shift)
    expected = phase_correlation_offset(
        reference, target, normalization="unit", max_abs_shift=6, subpixel=True
    )
    dx, dy, score = windowed_phase_correlation(reference, target, radius=6)
    assert (dx, dy) == pytest.approx((expected.dx, expected.dy))
    assert score == pytest.approx(expected.score)
    assert (round(dx), round(dy)) == shift


def test_phase_correlation_localizer_reports_native_pixels() -> None:
    reference = _texture(96, seed=2)
    target = _shift(reference, 8, -6)
    localizer = PhaseCorrelationLocalizer(working_factor=2, max_abs_shift_native=16)
    estimate = localizer.estimate(reference, target, sample_key="k")
    assert estimate.converged
    assert (round(estimate.dx), round(estimate.dy)) == (8, -6)
    assert localizer.radius == 8


def test_windowed_phase_correlation_rejects_a_radius_larger_than_the_image() -> None:
    image = _texture(16, seed=3)
    with pytest.raises(ValueError):
        windowed_phase_correlation(image, image, radius=8)


def test_sift_localizer_recovers_a_native_shift_at_working_resolution() -> None:
    pytest.importorskip("cv2")
    reference = _texture(384, seed=4)
    target = _shift(reference, 9, -6)
    localizer = SiftLocalizer(
        working_factor=3, max_features=4000, ratio=0.8,
        consensus_threshold=1.0, min_matches=4, max_abs_shift_native=24,
    )
    estimate = localizer.estimate(reference, target, sample_key="k")
    assert estimate.converged
    assert estimate.dx == pytest.approx(9.0, abs=1.5)
    assert estimate.dy == pytest.approx(-6.0, abs=1.5)


# ------------------------------------------------ stand-in matcher libraries


def _fake_kornia(shift_working: tuple[float, float]) -> types.ModuleType:
    grid = torch.stack(
        torch.meshgrid(torch.arange(4.0, 60.0, 8.0), torch.arange(4.0, 60.0, 8.0), indexing="xy"), -1
    ).reshape(-1, 2)

    class Features:
        def __init__(self, keypoints: torch.Tensor) -> None:
            self.keypoints = keypoints
            self.descriptors = torch.zeros(len(keypoints), 8)

    class Disk:
        seen_shapes: list[tuple[int, ...]] = []

        @classmethod
        def from_pretrained(cls, name: str) -> "Disk":
            assert name == "depth"
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, images, n, pad_if_not_divisible=False):
            Disk.seen_shapes.append(tuple(images.shape))
            assert pad_if_not_divisible
            moved = grid + torch.tensor(shift_working)
            moved[:3] = torch.tensor([[1.0, 1.0], [50.0, 2.0], [3.0, 40.0]])  # outliers
            return [Features(grid.clone()), Features(moved)]

    class Matcher:
        def __init__(self, name: str) -> None:
            assert name == "disk"

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, d1, d2, lafs1, lafs2, hw1=None, hw2=None):
            indices = torch.arange(len(d1))
            return torch.ones(len(d1), 1), torch.stack([indices, indices], 1)

    feature = types.ModuleType("kornia.feature")
    feature.DISK = Disk
    feature.LightGlueMatcher = Matcher
    feature.laf_from_center_scale_ori = lambda xy, scale: torch.zeros(*xy.shape[:2], 2, 3)
    kornia = types.ModuleType("kornia")
    kornia.feature = feature
    return kornia


def test_lightglue_adapter_measures_target_minus_reference(monkeypatch) -> None:
    kornia = _fake_kornia((4.0, -3.0))
    monkeypatch.setitem(sys.modules, "kornia", kornia)
    monkeypatch.setitem(sys.modules, "kornia.feature", kornia.feature)
    localizer = LightGlueDiskLocalizer(
        working_factor=3, max_keypoints=64, consensus_threshold=1.0,
        min_matches=8, max_abs_shift_native=96, device=torch.device("cpu"),
    )
    window = torch.rand(3, 192, 192)
    estimate = localizer.estimate(window, window, sample_key="k")
    assert estimate.converged
    assert (estimate.dx, estimate.dy) == pytest.approx((12.0, -9.0))
    assert estimate.support == 49 - 3
    assert kornia.feature.DISK.seen_shapes[-1] == (2, 3, 64, 64)


class _FakeRoma:
    def __init__(self, shift_pixels: tuple[float, float]) -> None:
        self.shift = shift_pixels
        self.sizes: list[tuple[int, int]] = []

    def match(self, image_a, image_b, device=None):
        self.sizes.append(image_a.size)
        width, height = image_a.size
        return torch.zeros(1), torch.tensor([float(width), float(height)])

    def sample(self, warp, certainty, num=5000):
        width, height = (float(v) for v in certainty)
        points = torch.rand(num, 2) * torch.tensor([width - 20.0, height - 20.0]) + 10.0
        moved = points + torch.tensor(self.shift)
        moved[: num // 5] += 25.0 * torch.randn(num // 5, 2)  # random outliers
        normalise = lambda p: torch.stack([p[:, 0] / width * 2 - 1, p[:, 1] / height * 2 - 1], 1)
        matches = torch.cat([normalise(points), normalise(moved)], 1)
        return matches, torch.rand(num)

    def to_pixel_coordinates(self, matches, height_a, width_a, height_b, width_b):
        def pixels(coords, height, width):
            return torch.stack([width / 2 * (coords[:, 0] + 1), height / 2 * (coords[:, 1] + 1)], -1)

        return pixels(matches[:, :2], height_a, width_a), pixels(matches[:, 2:], height_b, width_b)


def test_roma_adapter_scales_to_native_pixels_and_is_reproducible() -> None:
    fake = _FakeRoma((-5.0, 2.0))
    received = {}

    def factory(device, **kwargs):
        received.update(kwargs)
        return fake

    localizer = RomaOutdoorLocalizer(
        working_factor=3, num_matches=400, consensus_threshold=1.0, min_matches=8,
        max_abs_shift_native=96, device=torch.device("cpu"), seed=3407,
        model_kwargs={"use_custom_corr": False}, model_factory=factory,
    )
    assert received == {"use_custom_corr": False}
    window = torch.rand(3, 96, 96)
    first = localizer.estimate(window, window, sample_key="tile|scenario")
    second = localizer.estimate(window, window, sample_key="tile|scenario")
    assert first == second
    assert (first.dx, first.dy) == pytest.approx((-15.0, 6.0), abs=1e-4)
    assert fake.sizes[-1] == (32, 32)


# ---------------------------------------------------------- screen rules


def _row(candidate, scenario, gt, est, *, trigger=False, converged=True, seconds=0.1, sample="t0", change=None):
    return {
        "candidate": candidate, "scenario": scenario, "id": sample,
        "gt_dx": gt[0], "gt_dy": gt[1], "est_dx": est[0], "est_dy": est[1],
        "trigger": trigger, "converged": converged, "seconds": seconds,
        "change_fraction": change,
    }


def _by_scenario(physical_errors, clean_triggers, seconds=0.1):
    result = {}
    for key, errors in zip(PHYSICAL_KEYS, physical_errors):
        result[key] = scenario_metrics(
            [_row("c", key, (12.0, 0.0), (12.0 + e, 0.0), trigger=True, seconds=seconds) for e in errors]
        )
    result[CLEAN_KEY] = scenario_metrics(
        [_row("c", CLEAN_KEY, (0.0, 0.0), (0.0, 0.0), trigger=t, seconds=seconds) for t in clean_triggers]
    )
    return result


def test_scenario_metrics_counts_pck_triggers_and_failures() -> None:
    rows = [
        _row("c", "x", (0, 0), (1.0, 0.0), trigger=True),
        _row("c", "x", (0, 0), (0.0, 2.5)),
        _row("c", "x", (0, 0), (0.0, 0.0), converged=False, seconds=0.3),
        _row("c", "x", (0, 0), (10.0, 0.0), trigger="True"),
    ]
    metrics = scenario_metrics(rows)
    assert metrics["native_pck1"] == 0.5
    assert metrics["native_pck2"] == 0.5
    assert metrics["native_pck3"] == 0.75
    assert metrics["native_pck12"] == 1.0
    assert metrics["trigger_rate"] == 0.5
    assert metrics["nonconverged_rate"] == 0.25
    assert metrics["median_seconds"] == pytest.approx(0.1)


def test_candidate_tracks_only_if_every_physical_severity_passes() -> None:
    thresholds = {"tracking_pck2_min": 0.8, "clean_trigger_max": 0.1}
    good = candidate_verdict(_by_scenario([[0, 0, 0, 0, 3]] * 3, [False] * 10), thresholds)
    assert good["tracks"] and good["clean_passed"]
    assert good["mean_physical_pck2"] == pytest.approx(0.8)
    weak = candidate_verdict(
        _by_scenario([[0] * 5, [0] * 5, [0, 0, 0, 3, 3]], [True] * 2 + [False] * 8), thresholds
    )
    assert not weak["tracks"]
    assert not weak["clean_passed"]


def test_candidate_verdict_refuses_missing_scenarios() -> None:
    partial = _by_scenario([[0]] * 3, [False])
    del partial[CLEAN_KEY]
    with pytest.raises(KeyError):
        candidate_verdict(partial, {"tracking_pck2_min": 0.8, "clean_trigger_max": 0.1})


def _verdict(mean, tracks=True, clean=0.0, seconds=1.0):
    return {"tracks": tracks, "mean_physical_pck2": mean, "clean_trigger_rate": clean, "median_seconds": seconds}


def test_selection_excludes_the_reference_and_non_tracking_candidates() -> None:
    verdicts = {
        "p7c": _verdict(1.0),
        "a": _verdict(0.99, tracks=False),
        "b": _verdict(0.85),
    }
    roles = {"p7c": "reference", "a": "candidate", "b": "candidate"}
    assert select_candidate(verdicts, roles, margin=0.02) == "b"
    assert select_candidate({"p7c": _verdict(1.0)}, roles, margin=0.02) is None


def test_selection_breaks_margin_ties_by_clean_triggers_then_runtime() -> None:
    roles = {name: "candidate" for name in "abcd"}
    verdicts = {
        "a": _verdict(0.95, clean=0.20),
        "b": _verdict(0.94, clean=0.05, seconds=2.0),
        "c": _verdict(0.935, clean=0.05, seconds=1.0),
        "d": _verdict(0.90, clean=0.00),
    }
    assert select_candidate(verdicts, roles, margin=0.02) == "c"
    assert select_candidate(verdicts, roles, margin=0.0) == "a"


def test_offsets_anchor_requires_every_reference_row_and_gate() -> None:
    reference = [
        {"id": "t1", "scenario": "s", "est_dx": "12.0", "est_dy": "-12.0", "q_gate": "True"},
        {"id": "t2", "scenario": "s", "est_dx": "0.0", "est_dy": "12.0", "q_gate": "False"},
    ]
    observed = [
        _row("p7c", "s", (0, 0), (12.0, -12.0), trigger=True, sample="t1"),
        _row("p7c", "s", (0, 0), (0.0, 12.0), trigger=False, sample="t2"),
    ]
    assert anchor_offsets(observed, reference, tolerance=1e-4)["passed"]
    flipped = [dict(observed[0]), dict(observed[1], trigger=True)]
    result = anchor_offsets(flipped, reference, tolerance=1e-4)
    assert not result["passed"] and result["mismatches"] == 1
    moved = [dict(observed[0], est_dx=18.0), observed[1]]
    assert anchor_offsets(moved, reference, tolerance=1e-4)["max_abs_delta"] == 6.0
    missing = anchor_offsets(observed[:1], reference, tolerance=1e-4)
    assert not missing["passed"] and missing["missing"] == 1
    assert not anchor_offsets(observed, [], tolerance=1e-4)["passed"]


def test_selfcheck_fails_on_a_sign_error_or_a_failed_estimate() -> None:
    rows = [
        _row("good", "", (24, -24), (24.2, -23.9)),
        _row("good", "", (0, 0), (0.0, 0.0)),
        _row("flipped", "", (24, -24), (-24.0, 24.0)),
        _row("empty", "", (0, 0), (0.0, 0.0), converged=False),
    ]
    summary = selfcheck_summary(rows, tolerance=1.5)
    assert summary["candidates"]["good"]["passed"]
    assert not summary["candidates"]["flipped"]["passed"]
    assert not summary["candidates"]["empty"]["passed"]
    assert not summary["passed"]
    assert selfcheck_summary(rows[:2], tolerance=1.5)["passed"]


def test_clean_agreement_counts_consistent_nonzero_clean_offsets() -> None:
    rows = [
        _row("a", CLEAN_KEY, (0, 0), (0.0, -12.0), trigger=True, sample="t1"),
        _row("b", CLEAN_KEY, (0, 0), (1.0, -11.0), trigger=True, sample="t1"),
        _row("a", CLEAN_KEY, (0, 0), (0.0, -12.0), trigger=True, sample="t2"),
        _row("b", CLEAN_KEY, (0, 0), (12.0, 0.0), trigger=True, sample="t2"),
        _row("a", CLEAN_KEY, (0, 0), (0.0, 0.0), sample="t3"),
        _row("b", CLEAN_KEY, (0, 0), (0.0, 0.0), sample="t3"),
        _row("c", CLEAN_KEY, (0, 0), (0.0, -12.0), trigger=True, sample="t3"),
    ]
    result = clean_agreement(rows, ["a", "b"])
    assert result["clean_pairs"] == 3
    assert result["agreeing_nonzero"] == 1
    assert result["pairs"][0]["id"] == "t1"
    assert result["pairs"][0]["median_dy"] == pytest.approx(-11.5)


def test_change_bins_are_descriptive_per_tile() -> None:
    s3 = PHYSICAL_KEYS[-1]
    rows = [
        _row("a", CLEAN_KEY, (0, 0), (0, 0), sample="low", change=0.01),
        _row("a", s3, (48, 0), (48, 0), sample="low", change=0.01),
        _row("a", CLEAN_KEY, (0, 0), (12, 0), trigger=True, sample="high", change=0.30),
        _row("a", s3, (48, 0), (30, 0), sample="high", change=0.30),
    ]
    bins = change_bin_summary(rows)
    assert [entry["tiles"] for entry in bins] == [1, 0, 0, 1]
    assert bins[0]["physical_s3_pck2"] == 1.0 and bins[0]["clean_trigger_rate"] == 0.0
    assert bins[3]["physical_s3_pck2"] == 0.0 and bins[3]["clean_trigger_rate"] == 1.0
    assert bins[1]["physical_s3_pck2"] is None


def test_gate1_anchor_maps_p22_paths_onto_the_p7c_paths() -> None:
    reference = {
        CLEAN_KEY: {"raw": 0.9363, "oracle": 0.9363, "estimated": 0.7635, "estimated_gated": 0.8028}
    }
    observed = {
        CLEAN_KEY: {
            "raw": 0.9363, "oracle": 0.9363, "p7c_estimated": 0.7635,
            "p7c_gated": 0.80285, "estimated": 0.99, "estimated_gated": 0.99,
        }
    }
    path_map = {"raw": "raw", "oracle": "oracle", "estimated": "p7c_estimated", "estimated_gated": "p7c_gated"}
    result = compare_f1_tables(observed, reference, path_map, tolerance=1e-4)
    assert result["passed"]
    assert set(result["deltas"]) == {
        f"{path}:{CLEAN_KEY}" for path in ("raw", "oracle", "p7c_estimated", "p7c_gated")
    }
    strict = compare_f1_tables(observed, reference, path_map, tolerance=1e-5)
    assert not strict["passed"]
    del observed[CLEAN_KEY]["p7c_gated"]
    assert not compare_f1_tables(observed, reference, path_map, tolerance=1e-4)["passed"]


# ------------------------------------------------------------- config


def test_p23_config_is_frozen_against_the_p22_config() -> None:
    config = yaml.safe_load((ROOT / "configs/p23_whu_strong_localizer.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "preregistered"
    digest = hashlib.sha256((ROOT / config["p22_config"]).read_bytes()).hexdigest()
    assert digest == config["p22_config_sha256"]
    # Revised before any run: RoMa is the single pre-specified candidate.
    assert [spec["name"] for spec in config["candidates"]] == ["p7c", "roma_outdoor"]
    # Every RoMa result uses the fused kernel built for torch 2.5.1.
    assert config["candidates"][1]["use_custom_corr"] is True
    assert config["candidates"][1]["fused_local_corr_version"] == "0.1.1"
    roles = [spec.get("role", "candidate") for spec in config["candidates"]]
    assert roles.count("reference") == 1
    assert config["candidates"][roles.index("reference")]["name"] == "p7c"
    kinds = {spec["kind"] for spec in config["candidates"]}
    assert kinds <= {"p7c", "phase_correlation", "sift", "lightglue_disk", "roma_outdoor"}
    for spec in config["candidates"]:
        if spec["kind"] != "p7c":
            assert 3072 % int(spec["working_factor"]) == 0
    for dx, dy in config["selfcheck"]["shifts"]:
        assert max(abs(dx), abs(dy)) <= config["estimation"]["max_abs_shift_native_px"]
    assert (ROOT / config["protocol_doc"]).is_file()


def test_fused_kernel_must_be_the_pinned_build(monkeypatch) -> None:
    import importlib.metadata as metadata

    from tools.run_p23_whu_strong_localizer import require_fused_local_corr

    versions = {"fused-local-corr": "0.1.1"}

    def fake_version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(metadata, "version", fake_version)
    monkeypatch.setitem(sys.modules, "local_corr", types.ModuleType("local_corr"))
    require_fused_local_corr("0.1.1")
    versions["fused-local-corr"] = "0.3.211"
    with pytest.raises(SystemExit, match="0.3.211"):
        require_fused_local_corr("0.1.1")
    del versions["fused-local-corr"]
    with pytest.raises(SystemExit, match="not installed"):
        require_fused_local_corr("0.1.1")


def test_screen_items_cover_p22_blocks_and_validation_windows() -> None:
    from tools.run_p23_whu_strong_localizer import screen_items

    p22 = {
        "geometry": {
            "window_size": 64, "tile_size": 16,
            "train_region": {"x_max": 128, "y_max": 128},
            "val_columns": [8, 9], "val_rows": 2,
        },
        "gate0": {"blocks": 4},
    }
    label = np.zeros((160, 160), dtype=np.uint8)
    label[:32, 96:160] = 255
    rasters = {"t1": np.zeros((160, 160, 3), dtype=np.uint8), "label": label}
    items = screen_items(p22, rasters)
    train = [item for item in items if item["set"] == "train_blocks"]
    val = [item for item in items if item["set"] == "val_tiles"]
    assert len(train) == 4 and all(item["change_fraction"] is None for item in train)
    assert [item["id"] for item in val] == [
        "whu_val_r00c08", "whu_val_r00c09", "whu_val_r01c08", "whu_val_r01c09"
    ]
    for item in val:
        assert 0 <= item["window_left"] <= 160 - 64
        assert item["window_top"] == 0
        assert item["change_fraction"] == pytest.approx(
            float((label[0:64, item["window_left"] : item["window_left"] + 64] > 127).mean())
        )
