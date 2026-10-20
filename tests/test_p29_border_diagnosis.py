"""P29b views: the border cue without misalignment, and the misalignment without the border."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from rs_cd.affine_registration import perturb, perturbation_matrix

from p27_helpers import tile
from tools.p29_border_diagnosis import VIEWS, reading, tile_matrix, views


def _case():
    big = tile(256, 3)
    top, left, size = 64, 96, 64
    crop = big[top:top + size, left:left + size]
    matrix = perturbation_matrix(0.0, 0.85, 5.0, -4.0, width=size, height=size)
    return big, crop, top, left, matrix


def test_border_views_keep_the_content_in_place() -> None:
    big, crop, top, left, matrix = _case()
    result = views(crop, big, top, left, matrix)
    assert set(result) == set(VIEWS)
    raw_zero = result["raw"].sum(axis=2) == 0
    border_zero = result["border_raw"].sum(axis=2) == 0
    # Same zero border as the raw view: every raw zero is zeroed, and the only extra zeros are the
    # faint one-pixel bilinear band (next to the raw zeros or along the crop edge); content not moved.
    frame = np.ones_like(raw_zero)
    frame[1:-1, 1:-1] = False
    near = cv2.dilate((raw_zero | frame).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert not (raw_zero & ~border_zero).any() and not (border_zero & ~raw_zero & ~near).any()
    assert border_zero.mean() > 0.1
    kept = ~border_zero
    assert np.array_equal(result["border_raw"][kept], crop[kept])
    assert np.array_equal(result["border_oracle"][result["border_oracle"].sum(axis=2) > 0],
                          crop[result["border_oracle"].sum(axis=2) > 0])


def test_context_view_is_the_raw_view_without_the_border() -> None:
    big, crop, top, left, matrix = _case()
    result = views(crop, big, top, left, matrix)
    context, raw = result["context_raw"].astype(int), result["raw"].astype(int)
    assert (result["context_raw"].sum(axis=2) == 0).mean() == 0.0
    inside = perturb(np.ones_like(crop), matrix).min(axis=2) > 0
    inner = inside & np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
    assert np.abs(context - raw)[inner].max() <= 1
    shifted = tile_matrix(matrix, top, left)
    point = np.array([left + 10.0, top + 20.0, 1.0])
    assert np.allclose(shifted @ point, matrix @ np.array([10.0, 20.0, 1.0]) + [left, top])


def test_diagnosis_loop_scores_every_view(tmp_path) -> None:
    import torch

    from test_p27_f1 import DifferenceDetector
    from test_p27_selector import _toy
    from tools.p29_border_diagnosis import LEVELS, diagnose

    _, _, evaluation, _, _ = _toy(tmp_path)
    detector = {"model": DifferenceDetector(), "threshold": 0.5}
    result = diagnose(evaluation["datasets"][0], evaluation, {"original": detector, "augmented": detector},
                      torch.device("cpu"), max_crops=2)
    assert result["crops"] == 2
    assert set(result["f1"]["original"]) == {"clean"} | {f"{level}/{view}" for level in LEVELS for view in VIEWS}
    assert result["f1"]["original"] == result["f1"]["augmented"]


def test_reading_rule() -> None:
    def result(clean, **level3):
        f1 = {"clean": clean, **{f"3/{v}": clean - level3.get(v, 0.0) for v in VIEWS}}
        return {"f1": {"original": {**f1, "3/border_raw": clean - 0.01}, "augmented": f1}}

    confirmed = {d: result(0.9, raw=0.10, border_raw=0.20, context_raw=0.25) for d in ("a", "b")}
    assert reading(confirmed)["verdict"] == "confirmed"
    ruled_out = {d: result(0.9, raw=0.10, border_raw=0.01, context_raw=0.11) for d in ("a", "b")}
    assert reading(ruled_out)["verdict"] == "ruled_out"
    assert reading({"a": result(0.9, raw=0.1, border_raw=0.08, context_raw=0.12)})["verdict"] == "partial"
    assert reading(confirmed)["context_minus_raw_augmented"] == pytest.approx(15.0)
