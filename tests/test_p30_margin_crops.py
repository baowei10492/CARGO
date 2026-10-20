"""P30 margin crops: no perturbation of the P27 levels reaches outside the tile."""
from __future__ import annotations

import itertools

import pytest

from rs_cd.affine_crops import margin_crop_origins, source_inside_tile
from rs_cd.affine_registration import perturbation_matrix


def test_origins_keep_the_margin_and_the_crop_count() -> None:
    levir = margin_crop_origins(1024, 256, 64)
    assert len(levir) == 16 and sorted({top for top, _ in levir}) == [64, 277, 491, 704]
    assert sorted({left for _, left in margin_crop_origins(512, 256, 64)}) == [64, 192]
    with pytest.raises(ValueError):
        margin_crop_origins(256, 256, 64)


@pytest.mark.parametrize("tile", [512, 1024])
def test_level3_extremes_stay_inside_with_a_64_px_margin(tile) -> None:
    for (top, left), scale, tx, ty in itertools.product(margin_crop_origins(tile, 256, 64), (0.80, 1.25),
                                                        (-25.0, 25.0), (-25.0, 25.0)):
        matrix = perturbation_matrix(0.0, scale, tx, ty, width=256, height=256)
        assert source_inside_tile(matrix, top, left, 256, tile)


def test_a_smaller_margin_is_caught() -> None:
    matrix = perturbation_matrix(0.0, 0.80, 25.0, 25.0, width=256, height=256)
    assert not source_inside_tile(matrix, 32, 32, 256, 512)
    assert source_inside_tile(perturbation_matrix(0.0, 1.0, 0.0, 0.0, width=256, height=256), 0, 0, 256, 256)


def test_check_no_padding_stops_the_run() -> None:
    from tools.p29_border_diagnosis import check_no_padding

    matrix = perturbation_matrix(0.0, 0.80, 25.0, 25.0, width=256, height=256)
    check_no_padding({"crop_id": "a", "top": 64, "left": 64}, matrix, 256, 512)
    with pytest.raises(SystemExit, match="margin"):
        check_no_padding({"crop_id": "b", "top": 0, "left": 0}, matrix, 256, 512)
