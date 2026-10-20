import pytest
import torch

from tools.diagnose_registration_hygiene import (
    WHOLE_IMAGE_TILE_INDEX,
    _split_tiles,
    tile_dispersion_summary,
)


def _records(offsets_by_id, whole_by_id=None):
    rows = []
    for sample_id, offsets in offsets_by_id.items():
        if whole_by_id is not None:
            whole = whole_by_id[sample_id]
            rows.append(
                {
                    "id": sample_id,
                    "scenario": "clean:s0:common",
                    "tile_index": WHOLE_IMAGE_TILE_INDEX,
                    "est_dx": whole[0],
                    "est_dy": whole[1],
                }
            )
        for index, offset in enumerate(offsets):
            rows.append(
                {
                    "id": sample_id,
                    "scenario": "clean:s0:common",
                    "tile_index": index,
                    "est_dx": offset[0],
                    "est_dy": offset[1],
                }
            )
    return rows


def test_global_translation_gives_near_zero_dispersion() -> None:
    offsets = {
        f"img_{index}": [(2.0, 0.0)] * 4 for index in range(5)
    }
    whole = {f"img_{index}": (2.0, 0.0) for index in range(5)}
    summary = tile_dispersion_summary(_records(offsets, whole))
    assert summary["images"] == 5
    assert summary["tiles_per_image"] == [4]
    assert summary["median_max_pairwise_tile_distance"] == pytest.approx(
        0.0
    )
    assert summary["median_tile_dispersion_radius"] == pytest.approx(0.0)
    assert summary["fraction_images_max_pairwise_leq_2px"] == pytest.approx(
        1.0
    )
    assert summary["mean_tile_agreement_within_1px"] == pytest.approx(1.0)


def test_varying_field_gives_large_dispersion() -> None:
    offsets = {
        f"img_{index}": [
            (0.0, 0.0),
            (4.0, 0.0),
            (0.0, 4.0),
            (-4.0, 0.0),
        ]
        for index in range(4)
    }
    whole = {f"img_{index}": (0.0, 0.0) for index in range(4)}
    summary = tile_dispersion_summary(_records(offsets, whole))
    assert summary["median_max_pairwise_tile_distance"] == pytest.approx(
        8.0
    )
    assert summary["fraction_images_max_pairwise_leq_2px"] == pytest.approx(
        0.0
    )
    # Only the (0,0) tile agrees with the whole-image estimate.
    assert summary["mean_tile_agreement_within_1px"] == pytest.approx(
        0.25
    )
    assert summary["mean_tile_agreement_within_2px"] == pytest.approx(
        0.25
    )


def test_summary_requires_at_least_two_tiles() -> None:
    with pytest.raises(RuntimeError):
        tile_dispersion_summary(_records({"a": [(0.0, 0.0)]}))
    with pytest.raises(RuntimeError):
        tile_dispersion_summary([])


def test_split_tiles_shapes_and_count() -> None:
    image = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(
        2, 3, 8, 8
    )
    tiles = _split_tiles(image, 4)
    assert len(tiles) == 4
    assert all(tuple(tile.shape) == (2, 3, 4, 4) for tile in tiles)
    torch.testing.assert_close(tiles[0], image[:, :, :4, :4])
    torch.testing.assert_close(tiles[3], image[:, :, 4:, 4:])
    with pytest.raises(ValueError):
        _split_tiles(image, 8)
    with pytest.raises(ValueError):
        _split_tiles(image, 16)


def test_error_metrics_against_injected_offset() -> None:
    # Tiles all recover the injected (8, 0) offset within 2 px.
    offsets = {
        "img_0": [(8.0, 0.0), (8.0, 2.0), (6.0, 0.0), (8.0, 0.0)],
    }
    rows = _records(offsets, {"img_0": (8.0, 0.0)})
    for row in rows:
        row["gt_dx"] = 8.0
        row["gt_dy"] = 0.0
    summary = tile_dispersion_summary(rows)
    assert summary["median_injected_offset_magnitude"] == pytest.approx(
        8.0
    )
    assert summary["forced_choice_regime"] is False
    assert summary["median_tile_error_magnitude"] == pytest.approx(1.0)
    assert summary["tile_error_pck2"] == pytest.approx(1.0)
    assert summary["median_global_error_magnitude"] == pytest.approx(0.0)


def test_forced_choice_regime_flagged_on_clean() -> None:
    offsets = {"img_0": [(2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)]}
    rows = _records(offsets, {"img_0": (2.0, 0.0)})
    for row in rows:
        row["gt_dx"] = 0.0
        row["gt_dy"] = 0.0
    summary = tile_dispersion_summary(rows)
    assert summary["forced_choice_regime"] is True
    # Every tile sits on the minimum non-zero shell, 2 px from truth.
    assert summary["median_tile_error_magnitude"] == pytest.approx(2.0)
