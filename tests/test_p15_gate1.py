from __future__ import annotations

import pytest

from tools.summarize_p8_whu_headroom import headroom_decision
from tools.summarize_p15_gate0_instrument import instrument_diagnosis


def _rows(
    clean: float,
    shift2: float,
    shift4: float,
    shift8: float,
    *,
    valid_pixels: int = 123456,
) -> list[dict[str, str]]:
    values = [
        ("clean:s0:common", clean),
        ("shift:s1:t2", shift2),
        ("shift:s2:t2", shift4),
        ("shift:s3:t2", shift8),
    ]
    return [
        {
            "scenario": scenario,
            "f1": str(f1),
            "valid_pixels": str(valid_pixels),
        }
        for scenario, f1 in values
    ]


def test_dsifn_gate1_pass_has_p15_route() -> None:
    raw = _rows(0.80, 0.70, 0.60, 0.50)
    oracle = _rows(0.80, 0.79, 0.78, 0.70)
    decision = headroom_decision(
        raw,
        oracle,
        shift8_headroom_min=0.05,
        dataset="DSIFN-CD",
    )
    assert decision["decisive_gate_passed"] is True
    assert decision["dataset"] == "DSIFN-CD"
    assert decision["next_action"] == "p15_proceed_to_gate2_full_chain"


def test_dsifn_gate1_fail_has_stop_route() -> None:
    raw = _rows(0.80, 0.70, 0.60, 0.50)
    oracle = _rows(0.80, 0.71, 0.62, 0.549)
    decision = headroom_decision(
        raw,
        oracle,
        shift8_headroom_min=0.05,
        dataset="DSIFN-CD",
    )
    assert decision["decisive_gate_passed"] is False
    assert (
        decision["next_action"]
        == "p15_stop_dsifn_input_level_not_recoverable"
    )


def test_gate1_rejects_unpaired_valid_pixel_support() -> None:
    raw = _rows(0.80, 0.70, 0.60, 0.50, valid_pixels=100)
    oracle = _rows(0.80, 0.79, 0.78, 0.70, valid_pixels=99)
    with pytest.raises(ValueError, match="different valid-pixel support"):
        headroom_decision(
            raw,
            oracle,
            shift8_headroom_min=0.05,
            dataset="DSIFN-CD",
        )


def test_gate0_instrument_flags_boundary_like_severity1() -> None:
    summary = {
        "scenarios": {
            "clean:s0:common": {},
            "shift:s1:t2": {
                "median_injected_offset_magnitude": 2.0,
                "median_max_pairwise_tile_distance": 8.0,
                "median_tile_error_magnitude": 5.0,
                "tile_error_pck2": 0.2,
                "tile_error_pck4": 0.4,
            },
            "shift:s2:t2": {
                "median_injected_offset_magnitude": 4.0,
                "median_max_pairwise_tile_distance": 8.0,
                "median_tile_error_magnitude": 4.0,
                "tile_error_pck2": 0.4,
                "tile_error_pck4": 0.6,
            },
            "shift:s3:t2": {
                "median_injected_offset_magnitude": 8.0,
                "median_max_pairwise_tile_distance": 8.0,
                "median_tile_error_magnitude": 3.0,
                "tile_error_pck2": 0.5,
                "tile_error_pck4": 0.7,
            },
        }
    }
    result = instrument_diagnosis(summary, search_radius_pixels=8.0)
    assert result["diagnostic_only_no_gate"] is True
    assert result["instrument_warning"] is True
    assert result["checks"][
        "severity1_dispersion_near_search_boundary"
    ]["flagged"] is True
