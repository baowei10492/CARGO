"""P30 pre-check: the localizer and detector paths with and without the zero border."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rs_cd.affine_registration import perturbation_matrix
from rs_cd.localizer_training import step1_rows

from p27_helpers import tile
from tools.p29_border_diagnosis import views
from tools.p30_precheck import border_free, check, reading


def test_border_free_target_is_the_diagnosis_context_view() -> None:
    big = tile(256, 5)
    matrix = perturbation_matrix(0.0, 1.1, -6.0, 3.0, width=64, height=64)
    crop = big[32:96, 64:128]
    assert np.array_equal(border_free(big, 32, 64, 64, matrix), views(crop, big, 32, 64, matrix)["context_raw"])


def test_step1_rows_uses_the_given_target_builder() -> None:
    from test_p27_f1 import _tiny_localizer

    from rs_cd.affine_registration import AffineRanges, grid_points

    t = tile(64, 1)
    built = []

    def maker(matrix):
        built.append(matrix)
        return t

    rows = step1_rows(_tiny_localizer(), dataset="d", crop_id="c|y0000x0000", t1=t, t2=t, change_fraction=0.0,
                      ranges=AffineRanges(0.0, 0.8, 1.25, 8.0, 3), levels=[1, 2, 3], seed=1,
                      points=grid_points(64, 64, 5), tolerance=2.0, trigger_probability=0.5, make_target=maker)
    assert len(built) == 3 and [int(r["level"]) for r in rows] == [0, 1, 2, 3]


def test_check_scores_both_modes(tmp_path) -> None:
    from test_p27_f1 import DifferenceDetector, _tiny_localizer
    from test_p27_selector import _toy

    _, _, evaluation, _, _ = _toy(tmp_path)
    result = check(evaluation["datasets"][0], evaluation, _tiny_localizer(),
                   {"model": DifferenceDetector(), "threshold": 0.5}, torch.device("cpu"), max_crops=2)
    assert result["crops"] == 2 and set(result["success"]) == {"bordered", "border_free"}
    f1 = result["f1"]
    assert f1["bordered"]["0/raw"] == f1["border_free"]["0/raw"]
    assert set(f1["bordered"]) == {f"{level}/{path}" for level in range(4)
                                   for path in ("raw", "oracle", "estimated", "gated")}


def test_reading_rule() -> None:
    def r(bordered, free):
        return {"success": {"bordered": {"3": bordered}, "border_free": {"3": free}}}

    assert reading({"a": r(0.66, 0.63), "b": r(0.50, 0.47)})["verdict"] == "no_border_dependence"
    assert reading({"a": r(0.66, 0.50)})["verdict"] == "retrain_localizer"
    assert reading({"a": r(0.66, 0.61)})["drop_points"] == pytest.approx(5.0)
