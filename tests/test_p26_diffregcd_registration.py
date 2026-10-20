"""P26: DiffRegCD registration head as a similarity localizer."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rs_cd.affine_crops import select_crops, select_evenly, selfcheck_crops
from rs_cd.affine_registration import (
    apply,
    fit_similarity,
    grid_points,
    perturbation_matrix,
    transfer_error,
)
from rs_cd.external_flow_registration import (
    ALL_CONVENTIONS,
    choose_convention,
    coarse_points,
    convention_key,
    correspondences,
    normalized_to_pixels,
    parse_convention,
)

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("D:/work1/external/DiffRegCD")
FIT = dict(reprojection_px=2.0, max_iters=2000, confidence=0.999, refine_iters=10, min_inliers=8)


@dataclass(frozen=True)
class _Row:
    sample_id: str


def test_crop_selection_matches_p25() -> None:
    from tools.build_whu_scale_probe import select_evenly as reference_select
    from tools.run_p25_affine_step1 import select_crops as p25_select_crops
    from tools.run_p25_affine_step1 import selfcheck_crops as p25_selfcheck_crops

    items = list(range(1234))
    assert select_evenly(items, 600) == reference_select(items, 600)
    assert select_evenly(items[:5], 600) == items[:5]

    rows = [_Row(f"tile{index}") for index in range(70)]

    class Dataset:
        def __init__(self):
            self.rows = rows

    assert select_crops(rows, 1024, 256, 600) == p25_select_crops(Dataset(), 1024, 256, 600)

    class Cache:
        def crop(self, item, size):
            value = np.full((size, size, 3), (item["row"] * 37) % 200, dtype=np.uint8)
            value[:: 1 + item["row"] % 5] = 255
            return value, value, value[..., 0]

    crops = select_crops(rows, 512, 256, 60)
    settings = {"selfcheck_pool": 20, "selfcheck_crops": 3}
    assert selfcheck_crops(crops, Cache(), 8, 20, 3) == p25_selfcheck_crops(crops, Cache(), 8, settings)


def test_coarse_points_follow_align_corners() -> None:
    points = coarse_points(3, 5, 256, 256)
    assert points.shape == (15, 2)
    assert tuple(points[0]) == (0.0, 0.0)
    assert tuple(points[-1]) == (255.0, 255.0)
    assert tuple(points[1]) == (255.0 / 4, 0.0)
    assert tuple(points[5]) == (0.0, 127.5)


def test_normalized_flow_scales_to_pixels() -> None:
    flow = np.zeros((2, 2, 2))
    flow[0, 0] = (1.0, -1.0)
    flow[1, 1] = (0.5, 0.25)
    pixels = normalized_to_pixels(flow, 256, 128)
    assert tuple(pixels[0]) == (63.5, -127.5)
    assert tuple(pixels[3]) == (31.75, 31.875)


def _field(matrix: np.ndarray, coarse: int, size: int, *, grid: str) -> np.ndarray:
    """Normalized field that a correct head would output under a convention."""

    points = coarse_points(coarse, coarse, size, size)
    if grid == "reference":
        offsets = apply(matrix, points) - points
    else:
        # Analytic inverse (no LAPACK: MKL aborts next to torch on Windows).
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        det = a * e - b * d
        inverse = np.array([[e / det, -b / det, (b * f - c * e) / det],
                            [-d / det, a / det, (c * d - a * f) / det]])
        offsets = apply(inverse, points) - points
    return (offsets / ((size - 1) / 2.0)).reshape(coarse, coarse, 2)


@pytest.mark.parametrize("grid", ["reference", "perturbed"])
def test_each_grid_convention_recovers_the_perturbation(grid) -> None:
    size = 256
    truth = perturbation_matrix(-24.0, 1.18, 9.0, -13.0, width=size, height=size)
    field = _field(truth, 32, size, grid=grid)
    points = coarse_points(32, 32, size, size)
    grid_eval = grid_points(size, size, 17)
    errors = {}
    for sign in (1, -1):
        for candidate in ("reference", "perturbed"):
            source, target = correspondences(normalized_to_pixels(field, size, size), points, sign=sign, grid=candidate)
            estimate = fit_similarity(source, target, seed=1, **FIT)
            errors[(sign, candidate)] = (
                transfer_error(estimate.matrix, truth, grid_eval, width=size, height=size)
                if estimate.converged else np.inf
            )
    assert errors[(1, grid)] < 1e-4
    assert min(value for key, value in errors.items() if key != (1, grid)) > 5.0


def test_convention_keys_round_trip() -> None:
    assert len(ALL_CONVENTIONS) == 8
    for key in ALL_CONVENTIONS:
        decoding, sign, grid = parse_convention(key)
        assert convention_key(decoding, sign, grid) == key
    with pytest.raises(ValueError):
        parse_convention("loss|*|reference")


def _errors(**overrides):
    errors = {key: 40.0 for key in ALL_CONVENTIONS}
    errors.update(overrides)
    return errors


def test_choose_convention_requires_tolerance_and_a_clear_margin() -> None:
    chosen = choose_convention(
        _errors(**{"loss|+|reference": 0.6, "model|+|reference": 0.9}), tolerance=2.0, ambiguity_ratio=0.5
    )
    assert chosen["selected"] == "loss|+|reference" and chosen["passed"]
    assert chosen["best_rival_error"] == 40.0
    ambiguous = choose_convention(
        _errors(**{"loss|+|reference": 1.0, "loss|-|perturbed": 1.5}), tolerance=2.0, ambiguity_ratio=0.5
    )
    assert not ambiguous["unambiguous"] and not ambiguous["passed"]
    loose = choose_convention(_errors(**{"model|-|perturbed": 3.0}), tolerance=2.0, ambiguity_ratio=0.5)
    assert loose["selected"] == "model|-|perturbed" and not loose["within_tolerance"]
    failed = choose_convention(
        {key: (None if index else float("inf")) for index, key in enumerate(ALL_CONVENTIONS)},
        tolerance=2.0, ambiguity_ratio=0.5,
    )
    assert not failed["passed"] and failed["errors"][ALL_CONVENTIONS[0]] is None
    with pytest.raises(KeyError):
        choose_convention({"loss|+|reference": 0.1}, tolerance=2.0, ambiguity_ratio=0.5)


def _external_root() -> Path | None:
    config = yaml.safe_load((ROOT / "configs/p26_diffregcd_registration.yaml").read_text(encoding="utf-8"))
    for candidate in (Path(config["external"]["root"]), EXTERNAL):
        if (candidate / "losses" / "roma_flow_cls_256.py").is_file():
            return candidate
    return None


def test_author_loss_decoding_uses_the_edge_codebook() -> None:
    external = _external_root()
    if external is None:
        pytest.skip("DiffRegCD checkout not available")
    sys.path.insert(0, str(external))
    try:
        losses = importlib.import_module("losses.roma_flow_cls_256")
    finally:
        sys.path.remove(str(external))
    logits = torch.full((1, 1024, 2, 2), -50.0)
    logits[0, 31, 0, 0] = 50.0            # row 0, last column: (x, y) = (+1, -1)
    logits[0, 32 * 16 + 5, 1, 1] = 50.0   # row 16, column 5
    flow = losses._softargmax_flow(logits, 32, torch.tensor(0.7)).permute(0, 2, 3, 1)[0].numpy()
    codebook = np.linspace(-1.0, 1.0, 32)
    assert flow[0, 0] == pytest.approx((1.0, -1.0), abs=1e-5)
    assert flow[1, 1] == pytest.approx((codebook[5], codebook[16]), abs=1e-5)


def test_p26_config_is_chained_to_p25() -> None:
    config = yaml.safe_load((ROOT / "configs/p26_diffregcd_registration.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "preregistered"
    assert (ROOT / config["protocol_doc"]).is_file()
    assert hashlib.sha256((ROOT / config["p25_config"]).read_bytes()).hexdigest() == config["p25_config_sha256"]
    assert config["datasets"] == ["LEVIR-CD", "DSIFN-CD"]
    assert "WHU-CD" not in config["external"]["registration_heads"]
    assert set(config["external"]["registration_heads"]) == set(config["datasets"])
    parameters = set(inspect.signature(fit_similarity).parameters) - {"source", "target", "seed"}
    assert set(config["fit"]) == parameters
    assert config["external"]["commit"] == "6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38"


def test_input_order_and_roles() -> None:
    from rs_cd.external_flow_registration import ordered_inputs
    from tools.run_p26_diffregcd_registration import roles

    assert ordered_inputs("ref", "moved", "A_perturbed") == ("moved", "ref")
    assert ordered_inputs("ref", "moved", "A_reference") == ("ref", "moved")
    with pytest.raises(ValueError):
        ordered_inputs("ref", "moved", "B_first")
    t1, t2 = np.zeros(1), np.ones(1)
    assert roles(t1, t2, "t1") == (t2, t1)   # P26: T1 perturbed, T2 reference
    assert roles(t1, t2, "t2") == (t1, t2)   # P26-A: as P25


def test_calibration_outcome_branches() -> None:
    from rs_cd.external_flow_registration import calibration_outcome

    good = choose_convention(_errors(**{"loss|+|reference": 1.0}), tolerance=2.0, ambiguity_ratio=0.5)
    imprecise = choose_convention(_errors(**{"loss|+|reference": 6.0}), tolerance=2.0, ambiguity_ratio=0.5)
    ambiguous = choose_convention(_errors(**{"loss|+|reference": 6.0, "loss|-|perturbed": 7.0}),
                                  tolerance=2.0, ambiguity_ratio=0.5)
    never = choose_convention({key: None for key in ALL_CONVENTIONS}, tolerance=2.0, ambiguity_ratio=0.5)
    assert calibration_outcome(good, False) == calibration_outcome(good, True) == "measure"
    assert calibration_outcome(imprecise, True) == "descriptive"
    assert calibration_outcome(imprecise, False) == "stop"
    assert calibration_outcome(ambiguous, True) == "stop"
    assert not never["unambiguous"] and calibration_outcome(never, True) == "stop"


def test_p26a_config_amends_p26_with_two_changes() -> None:
    p26 = yaml.safe_load((ROOT / "configs/p26_diffregcd_registration.yaml").read_text(encoding="utf-8"))
    p26a = yaml.safe_load((ROOT / "configs/p26a_diffregcd_registration.yaml").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / p26a["amends"]["config"]).read_bytes()).hexdigest()
    assert digest == p26a["amends"]["sha256"]
    assert (ROOT / p26a["protocol_doc"]).is_file()
    assert p26a["input_order"] == "A_reference" and p26a["perturbation_target"] == "t2"
    assert p26a["selfcheck"]["continue_when_imprecise"] is True
    assert "input_order" not in p26 and "perturbation_target" not in p26
    for key in ("p25_config", "p25_config_sha256", "datasets", "external", "fit", "trigger_mean_displacement_px"):
        assert p26a[key] == p26[key], key
    assert {k: v for k, v in p26a["selfcheck"].items() if k != "continue_when_imprecise"} == p26["selfcheck"]


def test_load_configs_refuses_a_changed_p25_or_a_missing_head(tmp_path, monkeypatch) -> None:
    from tools.run_p26_diffregcd_registration import load_configs

    monkeypatch.chdir(ROOT)
    p26, p25 = load_configs(ROOT / "configs/p26_diffregcd_registration.yaml")
    assert [spec["name"] for spec in p25["datasets"]][:2] == p26["datasets"]
    config = yaml.safe_load((ROOT / "configs/p26_diffregcd_registration.yaml").read_text(encoding="utf-8"))
    changed = dict(config, p25_config_sha256="0" * 64)
    path = tmp_path / "p26.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="P26 expects"):
        load_configs(path)
    extra = dict(config, datasets=["LEVIR-CD", "DSIFN-CD", "WHU-CD"])
    path.write_text(yaml.safe_dump(extra), encoding="utf-8")
    with pytest.raises(SystemExit, match="no registration head"):
        load_configs(path)
