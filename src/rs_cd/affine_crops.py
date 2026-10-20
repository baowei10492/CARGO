"""Crop selection shared by the affine-perturbation lines (P25, P26).

These reproduce the P25 step 1 selection exactly (tools/run_p25_affine_step1.py
and tools.build_whu_scale_probe.select_evenly), without importing the
project runtime, so the DiffRegCD environment can use them. A parity test
pins the two together.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rs_cd.affine_registration import crop_origins


def select_evenly(items: list[Any], cap: int) -> list[Any]:
    """Deterministically pick <= cap items spread across the list."""

    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(items) <= cap:
        return list(items)
    step = len(items) / cap
    return [items[int(index * step)] for index in range(cap)]


def in_frame(matrix: Any, top: int, left: int) -> np.ndarray:
    """A crop-frame matrix expressed in a larger frame whose crop origin is (left, top): p -> M (p - o) + o."""

    m = np.asarray(matrix, dtype=np.float64)
    origin = np.array([left, top], dtype=np.float64)
    result = m.copy()
    result[:, 2] = m[:, 0] * -origin[0] + m[:, 1] * -origin[1] + m[:, 2] + origin
    return result


def cut_border_free(region: np.ndarray, matrix: Any, margin: int, crop_size: int) -> np.ndarray:
    """P30: warp a region (the crop plus ``margin`` pixels on every side) with a crop-frame
    matrix and cut the crop out, so the perturbed crop shows real content up to its edges."""

    import cv2

    size = region.shape[0]
    warped = cv2.warpAffine(region, in_frame(matrix, margin, margin), (region.shape[1], size),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.ascontiguousarray(warped[margin:margin + crop_size, margin:margin + crop_size])


def margin_crop_origins(tile_size: int, crop_size: int, margin: int) -> list[tuple[int, int]]:
    """As many origins per axis as the P25 grid, spread over [margin, tile - crop - margin] (P30).

    The crops keep ``margin`` real pixels to every tile edge, so a perturbation whose
    source window stays within that margin needs no padding; they may overlap.
    """
    per_axis = tile_size // crop_size
    high = tile_size - crop_size - margin
    if per_axis < 1 or high < margin:
        raise ValueError(f"a {tile_size} px tile has no {crop_size} px crop with a {margin} px margin")
    positions = sorted({int(round(v)) for v in np.linspace(margin, high, per_axis)})
    return [(top, left) for top in positions for left in positions]


def select_margin_crops(rows: Sequence[Any], tile_size: int, crop_size: int, margin: int) -> list[dict[str, Any]]:
    return [
        {"row": index, "sample_id": row.sample_id, "top": top, "left": left,
         "crop_id": f"{row.sample_id}|y{top:04d}x{left:04d}"}
        for index, row in enumerate(rows)
        for top, left in margin_crop_origins(tile_size, crop_size, margin)
    ]


def source_inside_tile(matrix: Any, top: int, left: int, crop_size: int, tile_size: int) -> bool:
    """Whether every pixel of the crop, warped by the crop-frame matrix, samples inside the tile."""

    m = np.asarray(matrix, dtype=np.float64)
    a, b = m[:, :2], m[:, 2]
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    inverse = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]]) / det
    corners = np.array([[0, 0], [crop_size - 1, 0], [0, crop_size - 1], [crop_size - 1, crop_size - 1]], float)
    # Element-wise products only (no BLAS; see rs_cd.affine_registration).
    shifted = corners - b
    source = np.stack([inverse[0, 0] * shifted[:, 0] + inverse[0, 1] * shifted[:, 1],
                       inverse[1, 0] * shifted[:, 0] + inverse[1, 1] * shifted[:, 1]], axis=1)
    source += [left, top]
    return bool(source.min() >= 0.0 and source.max() <= tile_size - 1)


def select_crops(rows: Sequence[Any], tile_size: int, crop_size: int, cap: int) -> list[dict[str, Any]]:
    crops = [
        {"row": index, "sample_id": row.sample_id, "top": top, "left": left,
         "crop_id": f"{row.sample_id}|y{top:04d}x{left:04d}"}
        for index, row in enumerate(rows)
        for top, left in crop_origins(tile_size, crop_size)
    ]
    return select_evenly(crops, cap)


class TileCache:
    """Loads one tile at a time through PairedChangeDataset and cuts crops."""

    def __init__(self, dataset: Any, tile_size: int) -> None:
        self.dataset = dataset
        self.tile_size = tile_size
        self.index: int | None = None
        self.arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def crop(self, item: dict[str, Any], size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.index != item["row"]:
            arrays = self.dataset._load_arrays(self.dataset.rows[item["row"]])
            if arrays[0].shape[:2] != (self.tile_size, self.tile_size):
                raise SystemExit(
                    f"{item['sample_id']} is {arrays[0].shape[:2]}, config says {self.tile_size}"
                )
            self.index, self.arrays = item["row"], arrays
        top, left = item["top"], item["left"]
        return tuple(
            np.ascontiguousarray(array[top : top + size, left : left + size])
            for array in self.arrays
        )


def selfcheck_crops(crops: list[dict[str, Any]], cache: Any, size: int, pool: int, count: int) -> list[dict[str, Any]]:
    """The most textured of the first ``pool`` crops (grey-level spread of T1)."""

    candidates = crops[:pool]
    spreads = [float(cache.crop(item, size)[0].astype(np.float32).mean(axis=2).std()) for item in candidates]
    order = sorted(range(len(candidates)), key=lambda index: (-spreads[index], index))
    return [candidates[index] for index in order[:count]]
