from __future__ import annotations

import numpy as np

from tools.audit_change_target_scale import mask_geometry


def test_large_target_is_less_shift_sensitive_than_small_target() -> None:
    small = np.zeros((128, 128), dtype=bool)
    small[48:80, 48:80] = True
    large = np.zeros((128, 128), dtype=bool)
    large[24:104, 24:104] = True

    small_result = mask_geometry(small)
    large_result = mask_geometry(large)

    small_iou = (
        small_result["self_shift_intersection"]["8"]
        / small_result["self_shift_union"]["8"]
    )
    large_iou = (
        large_result["self_shift_intersection"]["8"]
        / large_result["self_shift_union"]["8"]
    )
    assert large_iou > small_iou

    small_boundary = (
        small_result["boundary_band_pixels"]["8"]
        / small_result["changed_pixels"]
    )
    large_boundary = (
        large_result["boundary_band_pixels"]["8"]
        / large_result["changed_pixels"]
    )
    assert large_boundary < small_boundary
    assert (
        large_result["component_diameters"][0]
        > small_result["component_diameters"][0]
    )
