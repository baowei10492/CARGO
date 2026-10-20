"""P22 geometry and verdict logic.

The injected displacement, the window placement and the coarse-to-native
offset scaling decide what P22 measures, and none of them would fail loudly
if wrong: a sign error or an off-by-one crop only produces plausible F1.
Each is pinned here against the frozen P8 injection code or an exact
round trip.
"""
from __future__ import annotations

import pytest
import torch

from rs_cd.geometry import warp_reference_from_target
from rs_cd.physical_shift import (
    CLEAN_KEY,
    ShiftScenario,
    anchor_check,
    box_downsample,
    crop,
    gate0_summary,
    gate1_decision,
    p22_scenarios,
    recovery_fraction,
    scenario_translation,
    val_tiles,
    window_origin,
)
from rs_cd.robustness import (
    CorruptedPairDataset,
    CorruptionSpec,
    _paired_shift_valid_mask,
    _translate_reflect,
)

WHU_TRAIN_WIDTH = 21243
WHU_TRAIN_HEIGHT = 15354
IDS = ["whu_val_r00c35", "whu_val_r14c37", "whu_val_r28c40", "whu_p22_block_y00000_x00000"]


def test_scenarios_are_clean_then_control_then_physical() -> None:
    keys = [scenario.key for scenario in p22_scenarios(6)]
    assert keys == [
        CLEAN_KEY,
        "shift:s1:t2",
        "shift:s2:t2",
        "shift:s3:t2",
        "physical_shift:s1:t2",
        "physical_shift:s2:t2",
        "physical_shift:s3:t2",
    ]
    assert [scenario.pixels for scenario in p22_scenarios(6)] == [0, 2, 4, 8, 12, 24, 48]


@pytest.mark.parametrize("sample_id", IDS)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_pixel_family_is_the_p8_injection(sample_id: str, severity: int) -> None:
    """The control must be exactly what P8 Gate 1 injected."""

    p8 = CorruptedPairDataset(
        base_dataset=[], spec=CorruptionSpec("shift", severity, "t2", 3407)
    )
    expected = p8._applied_translation(sample_id, "shared")
    observed = scenario_translation(sample_id, ShiftScenario("pixel", severity))
    assert (float(observed[0]), float(observed[1])) == expected


@pytest.mark.parametrize("sample_id", IDS)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_physical_family_moves_the_same_direction_six_times_further(
    sample_id: str, severity: int
) -> None:
    pixel = scenario_translation(sample_id, ShiftScenario("pixel", severity))
    physical = scenario_translation(
        sample_id, ShiftScenario("physical", severity, scale=6)
    )
    assert physical == (pixel[0] * 6, pixel[1] * 6)


def test_clean_has_no_translation() -> None:
    assert scenario_translation("whu_val_r00c35", ShiftScenario("clean")) == (0, 0)


def test_window_is_centred_when_possible_and_clamped_at_edges() -> None:
    kwargs = dict(
        tile_size=512,
        window_size=3072,
        raster_width=WHU_TRAIN_WIDTH,
        raster_height=WHU_TRAIN_HEIGHT,
    )
    assert window_origin(8192, 7168, **kwargs) == (8192 - 1280, 7168 - 1280)
    left, top = window_origin(40 * 512, 0, **kwargs)
    assert left == WHU_TRAIN_WIDTH - 3072 and top == 0


def test_every_validation_tile_fits_inside_its_window() -> None:
    tiles = val_tiles(list(range(35, 41)), 29, 512)
    assert len(tiles) == 174
    for tile in tiles:
        left, top = window_origin(
            tile["left"],
            tile["top"],
            tile_size=512,
            window_size=3072,
            raster_width=WHU_TRAIN_WIDTH,
            raster_height=WHU_TRAIN_HEIGHT,
        )
        assert 0 <= left <= WHU_TRAIN_WIDTH - 3072
        assert 0 <= top <= WHU_TRAIN_HEIGHT - 3072
        assert left <= tile["left"] and tile["left"] + 512 <= left + 3072
        assert top <= tile["top"] and tile["top"] + 512 <= top + 3072


def test_window_larger_than_raster_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        window_origin(0, 0, tile_size=512, window_size=3072, raster_width=2048, raster_height=4096)


def test_crop_returns_the_tile_region() -> None:
    window = torch.arange(36.0).view(1, 6, 6)
    tile = crop(window, tile_left=12, tile_top=8, window_left=10, window_top=6, tile_size=3)
    torch.testing.assert_close(tile, window[:, 2:5, 2:5])


def test_downsampling_commutes_with_shifts_that_are_multiples_of_the_factor() -> None:
    """Why 12/24/48: the coarse view of the shifted window is the shifted coarse view."""

    generator = torch.Generator().manual_seed(3407)
    window = torch.rand(3, 144, 144, generator=generator)
    shifted, _ = _translate_reflect(window, dx=12, dy=-6)
    coarse_of_shifted = box_downsample(shifted.unsqueeze(0), 6)[0]
    shifted_coarse, _ = _translate_reflect(box_downsample(window.unsqueeze(0), 6)[0], dx=2, dy=-1)
    interior = (slice(None), slice(3, -3), slice(3, -3))
    torch.testing.assert_close(coarse_of_shifted[interior], shifted_coarse[interior], rtol=0, atol=1e-6)


def test_box_downsample_rejects_sizes_that_are_not_multiples() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        box_downsample(torch.zeros(1, 3, 100, 96), 6)


def test_scaled_coarse_estimate_undoes_the_native_injection_exactly() -> None:
    """Sign and scale: a perfect coarse estimate times the factor restores the window."""

    generator = torch.Generator().manual_seed(7)
    window = torch.rand(3, 240, 240, generator=generator)
    shifted, _ = _translate_reflect(window, dx=12, dy=-24)
    coarse_estimate = torch.tensor([[2.0, -4.0]])
    restored = warp_reference_from_target(shifted.unsqueeze(0), coarse_estimate * 6)[0]
    interior = (slice(None), slice(24, -24), slice(24, -24))
    # Grid construction carries ~1e-5 px of round-off at integer offsets.
    torch.testing.assert_close(restored[interior], window[interior], rtol=0, atol=1e-4)


def _records(pck_errors: dict[str, list[float]], clean_applied: list[float]) -> list[dict]:
    rows = []
    for key, errors in pck_errors.items():
        for index, error in enumerate(errors):
            rows.append({
                "id": f"b{index}", "scenario": key, "gt_dx": 0.0, "gt_dy": 0.0,
                "est_dx": error, "est_dy": 0.0, "applied_dx": 0.0, "applied_dy": 0.0,
                "q_gate": True,
            })
    for index, applied in enumerate(clean_applied):
        rows.append({
            "id": f"c{index}", "scenario": CLEAN_KEY, "gt_dx": 0.0, "gt_dy": 0.0,
            "est_dx": 12.0, "est_dy": 0.0, "applied_dx": applied, "applied_dy": 0.0,
            "q_gate": applied > 0,
        })
    return rows


THRESHOLDS0 = {"conditional_pck1_min": 0.70, "conditional_pck2_min": 0.85, "clean_gated_within_1px_min": 0.90}


def test_gate0_passes_on_accurate_coarse_estimates_even_if_clean_fails() -> None:
    records = _records(
        {"physical_shift:s2:t2": [0.0] * 9 + [6.0], "physical_shift:s3:t2": [0.0] * 8 + [12.0, 6.0]},
        clean_applied=[12.0] * 5 + [0.0] * 5,
    )
    summary = gate0_summary(records, factor=6, thresholds=THRESHOLDS0)
    assert summary["verdict"] == "proceed_to_gate1"
    assert summary["clean_passed"] is False
    assert summary["scenarios"]["physical_shift:s3:t2"]["coarse_pck1"] == pytest.approx(0.9)


def test_gate0_stops_when_one_gated_severity_is_inaccurate() -> None:
    records = _records(
        {"physical_shift:s2:t2": [0.0] * 10, "physical_shift:s3:t2": [0.0] * 6 + [18.0] * 4},
        clean_applied=[0.0] * 10,
    )
    summary = gate0_summary(records, factor=6, thresholds=THRESHOLDS0)
    assert summary["verdict"] == "stop_localizer_insufficient"


def test_gate0_reports_boundary_fractions_for_the_clamped_severity() -> None:
    """Physical s3 truth sits on the +-8 coarse boundary; est must be annotated."""

    rows = [
        {"id": "a", "scenario": "physical_shift:s3:t2", "gt_dx": 48.0, "gt_dy": 0.0,
         "est_dx": 48.0, "est_dy": 0.0, "applied_dx": 48.0, "applied_dy": 0.0, "q_gate": True},
        {"id": "b", "scenario": "physical_shift:s3:t2", "gt_dx": 0.0, "gt_dy": -48.0,
         "est_dx": 0.0, "est_dy": -36.0, "applied_dx": 0.0, "applied_dy": -36.0, "q_gate": True},
    ]
    rows += _records({"physical_shift:s2:t2": [0.0] * 10}, clean_applied=[0.0] * 10)
    summary = gate0_summary(rows, factor=6, thresholds=THRESHOLDS0, boundary_coarse_px=8.0)
    s3 = summary["scenarios"]["physical_shift:s3:t2"]
    assert s3["gt_boundary_fraction"] == pytest.approx(1.0)
    assert s3["est_boundary_fraction"] == pytest.approx(0.5)
    assert s3["excess_boundary"] == pytest.approx(-0.5)
    assert summary["boundary_coarse_px"] == 8.0
    # 36 px is two coarse px short of the truth: not a coarse PCK@1 hit.
    assert s3["coarse_pck1"] == pytest.approx(0.5)


def test_common_interior_support_covers_the_largest_shift() -> None:
    """The 48 px border leaves 416 x 416 pixels and excludes every shifted border."""

    from rs_cd.geometry import interior_valid_mask

    mask = interior_valid_mask(1, 512, 512, 48)
    assert int(mask.sum()) == 416 * 416 == 173_056
    for dx, dy in ((48, 0), (0, -48), (48, 48), (-48, 48)):
        paired = _paired_shift_valid_mask(512, 512, dx, dy)
        assert torch.all(paired[mask[0] > 0.5] > 0.5), "common support must lie inside every paired support"


def test_gate0_refuses_missing_scenarios() -> None:
    with pytest.raises(KeyError, match="missing"):
        gate0_summary(_records({"physical_shift:s2:t2": [0.0]}, [0.0]), factor=6, thresholds=THRESHOLDS0)


THRESHOLDS1 = {"raw_damage_min": 0.05, "headroom_min": 0.05, "recovery_min": 0.50}


def _f1(*, clean=0.93, raw3=0.60, oracle3=0.93, est3=0.85, raw2=0.75, oracle2=0.93, est2=0.88):
    return {
        CLEAN_KEY: {"raw": clean, "oracle": clean, "estimated": clean - 0.02, "estimated_gated": clean},
        "physical_shift:s2:t2": {"raw": raw2, "oracle": oracle2, "estimated": est2, "estimated_gated": est2},
        "physical_shift:s3:t2": {"raw": raw3, "oracle": oracle3, "estimated": est3, "estimated_gated": est3},
    }


def test_gate1_proceeds_when_the_estimate_recovers_most_of_the_headroom() -> None:
    decision = gate1_decision(_f1(), anchor_passed=True, thresholds=THRESHOLDS1)
    assert decision["verdict"] == "proceed_to_gate2"
    assert decision["recovery"]["physical_shift:s3:t2"] == pytest.approx(0.25 / 0.33)


def test_gate1_anchor_failure_overrides_everything() -> None:
    decision = gate1_decision(_f1(), anchor_passed=False, thresholds=THRESHOLDS1)
    assert decision["verdict"] == "anchor_failed"


def test_gate1_branches_in_order() -> None:
    assert gate1_decision(_f1(raw3=0.90, oracle3=0.93), anchor_passed=True, thresholds=THRESHOLDS1)["verdict"] == "no_step_needed"
    assert gate1_decision(_f1(raw3=0.80, oracle3=0.84), anchor_passed=True, thresholds=THRESHOLDS1)["verdict"] == "headroom_insufficient"
    assert gate1_decision(_f1(est3=0.70), anchor_passed=True, thresholds=THRESHOLDS1)["verdict"] == "recovery_insufficient"


def test_recovery_is_undefined_without_headroom() -> None:
    assert recovery_fraction(0.9, 0.95, 0.9) is None
    assert recovery_fraction(0.6, 0.8, 1.0) == pytest.approx(0.5)


def test_anchor_check_uses_absolute_tolerance() -> None:
    reference = {"clean": 0.93445, "raw:shift:s3:t2": 0.82565}
    passed = anchor_check({"clean": 0.934455, "raw:shift:s3:t2": 0.82566}, reference, 1e-4)
    failed = anchor_check({"clean": 0.93445, "raw:shift:s3:t2": 0.8280}, reference, 1e-4)
    assert passed["passed"] and not failed["passed"]
    with pytest.raises(KeyError):
        anchor_check({"clean": 0.93445}, reference, 1e-4)
