"""DiffRegCD generator check: fidelity to the authors' generator, the views, the loop and the reading."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools.p27_diffregcd_generator_check import (
    ORDERS,
    RANGES,
    check,
    content,
    draw_seed,
    generator_matrix,
    generator_warp,
    keys,
    reading,
    views,
)

from p27_helpers import make_dataset

ROOT = Path(__file__).resolve().parents[1]


AUTHOR_RUN = """
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("prepare_gt_flow", sys.argv[1])
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except ImportError as error:
    print("IMPORT_ERROR", error)
    sys.exit(3)
angle, low, high, trans = (float(v) for v in sys.argv[5:9])
np.random.seed(11)
module.apply_affine_transform_and_generate_flow(sys.argv[2], sys.argv[3], sys.argv[4], angle_range=(-angle, angle),
                                                scale_range=(low, high), translation_range=(-trans, trans))
"""


def _author_script() -> Path:
    candidates = [os.environ.get("DIFFREGCD_ROOT"), ROOT.parent / "DiffRegCD", ROOT.parent / "external" / "DiffRegCD"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "scripts" / "prepare_gt_flow.py").is_file():
            return Path(candidate) / "scripts" / "prepare_gt_flow.py"
    pytest.skip("DiffRegCD repository not found")


@pytest.mark.parametrize("name", list(RANGES))
def test_reproduces_the_authors_generator(tmp_path, name) -> None:
    # The authors' script runs in its own process: it multiplies matrices with numpy, which aborts on
    # Windows' MKL build once torch is loaded (see rs_cd.affine_registration).
    script = _author_script()
    rng = np.random.default_rng(5)
    image = cv2.resize(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8), (256, 256), interpolation=cv2.INTER_CUBIC)
    (tmp_path / "src").mkdir()
    cv2.imwrite(str(tmp_path / "src" / "x.png"), image)
    spec = RANGES[name]
    run = subprocess.run([sys.executable, "-c", AUTHOR_RUN, str(script), str(tmp_path / "src"), str(tmp_path / "warped"),
                          str(tmp_path / "flow"), str(spec["angle"]), str(spec["scale"][0]), str(spec["scale"][1]),
                          str(spec["trans"])], capture_output=True, text=True)
    if run.returncode == 3:
        pytest.skip(f"cannot import the authors' generator here: {run.stdout.strip()}")
    assert run.returncode == 0, run.stderr
    theirs = cv2.imread(str(tmp_path / "warped" / "x.png"), cv2.IMREAD_COLOR)
    source = cv2.imread(str(tmp_path / "src" / "x.png"), cv2.IMREAD_COLOR)
    matrix = generator_matrix(np.random.RandomState(11), 256, **spec)
    assert np.array_equal(generator_warp(source, matrix), theirs)
    flow = np.load(tmp_path / "flow" / "x.npy")
    ys, xs = np.mgrid[0:256, 0:256].astype(np.float64)
    assert np.allclose(flow[0], matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2] - xs, atol=1e-3)
    assert np.allclose(flow[1], matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2] - ys, atol=1e-3)
    assert (theirs.max(axis=2) == 0).any() or name == "paper_norot"   # the constant zero border


def test_views_share_the_matrix_between_orders() -> None:
    rng = np.random.default_rng(1)
    t1 = rng.integers(1, 256, (256, 256, 3), dtype=np.uint8)
    t2 = rng.integers(1, 256, (256, 256, 3), dtype=np.uint8)
    out = {key: (a, b, inside) for key, a, b, inside in views(t1, t2, "tile|y0000x0000", 3407)}
    assert list(out) == keys()
    assert out["clean"][2].all() and out["clean"][0] is t1 and out["clean"][1] is t2
    for name in ORDERS["a_warped"]:
        matrix = generator_matrix(np.random.RandomState(draw_seed(3407, "tile|y0000x0000", name)), 256, **RANGES[name])
        assert np.array_equal(out[f"b_warped/{name}"][1], generator_warp(t2, matrix))
        assert np.array_equal(out[f"a_warped/{name}"][0], generator_warp(t1, matrix))
        assert out[f"b_warped/{name}"][0] is t1 and out[f"a_warped/{name}"][1] is t2
        assert np.array_equal(out[f"b_warped/{name}"][2], content(matrix, 256))
    norot = generator_matrix(np.random.RandomState(0), 256, **RANGES["paper_norot"])
    assert abs(norot[0, 1]) < 1e-12 and abs(norot[1, 0]) < 1e-12
    assert not out["b_warped/paper"][2].all()          # the warped view loses content at its border


def test_check_loop_and_reading(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 2, 512, split="test")
    spec = {"name": "LEVIR-CD", "dataset_root": source.dataset_root, "manifest": source.manifest, "split": "test",
            "tile_size": 512, "mask_threshold": 127}

    def infer(a, b):
        change = (a - b).abs().mean(dim=1, keepdim=True) * 8.0 - 1.0
        logits = torch.cat([torch.zeros_like(change), change], dim=1)
        return {"official_unmodified": logits, "intended_warped": logits * 0.5}

    result = check(spec, infer, size=256, seed=3407, threshold=0.5, device=torch.device("cpu"), cap=None)
    assert result["crops"] == 8
    assert all(0.0 <= value <= 1.0 for value in result["f1"].values())
    assert result["f1"]["clean/official_unmodified/all"] == result["f1"]["clean/official_unmodified/content"]
    assert len(result["f1"]) == len(keys()) * 2 * 2

    def f1(clean, b_paper, a_paper):
        values = {"clean/official_unmodified/all": clean, "b_warped/paper/official_unmodified/all": b_paper,
                  "a_warped/paper/official_unmodified/all": a_paper}
        return reading("LEVIR-CD", values)["verdict"]

    assert f1(0.87, 0.30, 0.40) == "released_not_robust"
    assert f1(0.87, 0.30, 0.84) == "released_robust"
    assert f1(0.87, 0.75, 0.60) == "intermediate"
    assert f1(0.60, 0.30, 0.30) == "sanity_failed"
    assert reading("WHU-CD", {"clean/official_unmodified/all": 0.9, "b_warped/paper/official_unmodified/all": 0.2,
                              "a_warped/paper/official_unmodified/all": 0.2})["verdict"] == "reported"
