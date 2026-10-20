"""P22: WHU-CD under physically matched displacements.

LEVIR-CD's 2/4/8 px shifts are 1/2/4 m at its 0.5 m GSD. At WHU-CD's
0.075 m the same physical displacements would be 13.3/26.7/53.3 px, so the
pixel-defined protocol applies a 6.7x milder perturbation there (main record
71.5, trap F8). P22 injects 12/24/48 px (0.9/1.8/3.6 m): multiples of a 6x
analysis factor, so a localizer run on a 6x box-downsampled 3072 px context
window sees exactly the integer 2/4/8 px shifts it was frozen on (main
record 77 measured that configuration), while change detection stays at
native resolution. The native 2/4/8 px family is kept as the control.

Everything here is deterministic, device-agnostic and label-free; the P22
tool adds models, rasters and labels. Protocol:
docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from rs_cd.robustness import (
    _SEVERITY_VALUES,
    _stable_seed,
    _translation_direction,
)

BASE_SHIFT_PIXELS: tuple[float, float, float] = _SEVERITY_VALUES["shift"]
CLEAN_KEY = "clean:s0:common"


@dataclass(frozen=True)
class ShiftScenario:
    """One injected T2 translation family and severity."""

    family: str
    severity: int = 0
    scale: int = 1

    def __post_init__(self) -> None:
        if self.family not in {"clean", "pixel", "physical"}:
            raise ValueError(f"unknown scenario family {self.family!r}")
        if self.family == "clean":
            if self.severity != 0:
                raise ValueError("clean scenario must use severity 0")
        elif self.severity not in {1, 2, 3}:
            raise ValueError("severity must be 1, 2 or 3")
        if self.scale < 1:
            raise ValueError("scale must be positive")

    @property
    def key(self) -> str:
        if self.family == "clean":
            return CLEAN_KEY
        prefix = "shift" if self.family == "pixel" else "physical_shift"
        return f"{prefix}:s{self.severity}:t2"

    @property
    def pixels(self) -> int:
        if self.family == "clean":
            return 0
        return int(BASE_SHIFT_PIXELS[self.severity - 1]) * self.scale


def p22_scenarios(factor: int) -> list[ShiftScenario]:
    """Clean, the native pixel control (A) and the physical family (B)."""

    return (
        [ShiftScenario("clean")]
        + [ShiftScenario("pixel", severity) for severity in (1, 2, 3)]
        + [
            ShiftScenario("physical", severity, scale=int(factor))
            for severity in (1, 2, 3)
        ]
    )


def scenario_translation(
    sample_id: str,
    scenario: ShiftScenario,
    seed: int = 3407,
) -> tuple[int, int]:
    """T2 displacement ``(dx, dy)`` in native pixels.

    The direction comes from the P8 shift seed for this tile and severity, so
    the pixel family is identical to CorruptedPairDataset's T2 shift and the
    physical family moves the same tile in the same direction, only further.
    The two families are therefore paired per tile.
    """

    if scenario.family == "clean":
        return 0, 0
    direction_seed = _stable_seed(
        int(seed), str(sample_id), "shift", scenario.severity, "shared"
    )
    return _translation_direction(direction_seed, scenario.pixels)


def window_origin(
    tile_left: int,
    tile_top: int,
    *,
    tile_size: int,
    window_size: int,
    raster_width: int,
    raster_height: int,
) -> tuple[int, int]:
    """Top-left of a context window centred on the tile, clamped to the raster.

    Tiles near the raster edge get an off-centre window rather than padding,
    so the localizer never sees synthetic content.
    """

    if window_size < tile_size:
        raise ValueError("window must be at least as large as the tile")
    if window_size > raster_width or window_size > raster_height:
        raise ValueError("window does not fit inside the raster")
    margin = (window_size - tile_size) // 2
    left = min(max(tile_left - margin, 0), raster_width - window_size)
    top = min(max(tile_top - margin, 0), raster_height - window_size)
    inside = (
        left <= tile_left
        and tile_left + tile_size <= left + window_size
        and top <= tile_top
        and tile_top + tile_size <= top + window_size
    )
    if not inside:
        raise ValueError("tile lies outside the raster")
    return left, top


def crop(
    window: torch.Tensor,
    *,
    tile_left: int,
    tile_top: int,
    window_left: int,
    window_top: int,
    tile_size: int,
) -> torch.Tensor:
    x = tile_left - window_left
    y = tile_top - window_top
    return window[..., y : y + tile_size, x : x + tile_size]


def box_downsample(images: torch.Tensor, factor: int) -> torch.Tensor:
    """Exact block averaging, the coarse-sensor model of main record 76.

    Unlike the PIL BOX filter used by the earlier probe, no rounding to
    uint8 happens here; the difference is at most half a grey level.
    """

    if factor < 1:
        raise ValueError("factor must be positive")
    height, width = images.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(
            f"{height}x{width} is not divisible by the factor {factor}"
        )
    return F.avg_pool2d(images, kernel_size=factor, stride=factor)


def val_tiles(
    columns: list[int],
    rows: int,
    tile_size: int,
) -> list[dict[str, Any]]:
    """Validation tiles in the P8 naming ``whu_val_r<row>c<column>``."""

    return [
        {
            "sample_id": f"whu_val_r{row:02d}c{column:02d}",
            "left": column * tile_size,
            "top": row * tile_size,
        }
        for row in range(rows)
        for column in columns
    ]


def _rate(flags: list[bool]) -> float | None:
    return (sum(1 for flag in flags if flag) / len(flags)) if flags else None


def gate0_summary(
    records: list[dict[str, Any]],
    *,
    factor: int,
    thresholds: dict[str, float],
    boundary_coarse_px: float = 8.0,
) -> dict[str, Any]:
    """Label-free localizer verdict for the physical family.

    Records carry native-pixel ground truth (gt_dx, gt_dy), the conditional
    estimate (est_dx, est_dy) and the gated applied offset (applied_dx,
    applied_dy), all in native pixels, plus the q gate. PCK for the physical
    family is counted in coarse pixels (1 coarse px = ``factor`` native px),
    matching the P8 stage-A gates; the pixel control is counted in native
    pixels and never gates.

    Physical severity 3 (8 coarse px) sits on the edge of the localizer's
    search domain, where a clamped estimate can coincide with the truth
    (main record 73.2). The boundary annotation of main record 75.2 is
    therefore reported per scenario: est/gt boundary fractions and their
    excess. Clamping can only inflate that severity's PCK, so severity 2 is
    the binding gate; both must still pass.
    """

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scenario.setdefault(str(record["scenario"]), []).append(record)

    def at_boundary(dx: Any, dy: Any) -> bool:
        return max(abs(float(dx)), abs(float(dy))) / factor >= boundary_coarse_px - 1e-6

    scenarios: dict[str, Any] = {}
    for key, rows in by_scenario.items():
        errors = [
            math.hypot(
                float(row["est_dx"]) - float(row["gt_dx"]),
                float(row["est_dy"]) - float(row["gt_dy"]),
            )
            for row in rows
        ]
        est_boundary = _rate([at_boundary(row["est_dx"], row["est_dy"]) for row in rows])
        gt_boundary = _rate([at_boundary(row["gt_dx"], row["gt_dy"]) for row in rows])
        entry: dict[str, Any] = {
            "n": len(rows),
            "native_mean_epe": statistics.fmean(errors),
            "native_median_epe": statistics.median(errors),
            "q_trigger_rate": _rate([bool(row["q_gate"]) for row in rows]),
            "est_boundary_fraction": est_boundary,
            "gt_boundary_fraction": gt_boundary,
            "excess_boundary": est_boundary - gt_boundary,
        }
        if key.startswith("physical_shift"):
            entry["coarse_pck1"] = _rate([e <= factor + 1e-6 for e in errors])
            entry["coarse_pck2"] = _rate(
                [e <= 2 * factor + 1e-6 for e in errors]
            )
        entry["native_pck1"] = _rate([e <= 1.0 + 1e-6 for e in errors])
        entry["native_pck2"] = _rate([e <= 2.0 + 1e-6 for e in errors])
        if key == CLEAN_KEY:
            applied = [
                math.hypot(float(row["applied_dx"]), float(row["applied_dy"]))
                for row in rows
            ]
            entry["gated_within_1_coarse_px"] = _rate(
                [value <= factor + 1e-6 for value in applied]
            )
        scenarios[key] = entry

    gated = [f"physical_shift:s{severity}:t2" for severity in (2, 3)]
    missing = [key for key in gated + [CLEAN_KEY] if key not in scenarios]
    if missing:
        raise KeyError(f"gate 0 records are missing scenarios: {missing}")
    conditional_passed = all(
        scenarios[key]["coarse_pck1"] >= thresholds["conditional_pck1_min"]
        and scenarios[key]["coarse_pck2"] >= thresholds["conditional_pck2_min"]
        for key in gated
    )
    clean_passed = (
        scenarios[CLEAN_KEY]["gated_within_1_coarse_px"]
        >= thresholds["clean_gated_within_1px_min"]
    )
    return {
        "scenarios": scenarios,
        "conditional_passed": conditional_passed,
        "clean_passed": clean_passed,
        "verdict": (
            "proceed_to_gate1"
            if conditional_passed
            else "stop_localizer_insufficient"
        ),
        "boundary_coarse_px": float(boundary_coarse_px),
        "notes": [
            "Physical-family PCK is in coarse pixels (P8 stage-A gates).",
            "Physical severity 3 lies on the search boundary: clamping can "
            "only inflate its PCK (see excess_boundary), so severity 2 is "
            "the binding gate; both must pass.",
            "A clean failure alone does not stop the chain (P8 precedent); "
            "it is reported and handled by the router in Gate 2.",
            "The native pixel family is a control and never gates.",
        ],
    }


def anchor_check(
    observed: dict[str, float],
    reference: dict[str, float],
    tolerance: float,
) -> dict[str, Any]:
    missing = [key for key in reference if key not in observed]
    if missing:
        raise KeyError(f"anchor is missing observed values: {missing}")
    deltas = {
        key: float(observed[key]) - float(value)
        for key, value in reference.items()
    }
    return {
        "observed": {key: float(observed[key]) for key in reference},
        "reference": {key: float(value) for key, value in reference.items()},
        "deltas": deltas,
        "tolerance": float(tolerance),
        "passed": all(abs(delta) <= tolerance for delta in deltas.values()),
    }


def recovery_fraction(
    raw: float, estimated: float, oracle: float
) -> float | None:
    available = oracle - raw
    if available <= 0:
        return None
    return (estimated - raw) / available


def gate1_decision(
    f1: dict[str, dict[str, float]],
    *,
    anchor_passed: bool,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Preregistered Gate 1 verdict for the physical family.

    ``f1[scenario][path]`` with paths raw / oracle / estimated /
    estimated_gated, all on paired support.
    """

    s2 = "physical_shift:s2:t2"
    s3 = "physical_shift:s3:t2"
    clean = float(f1[CLEAN_KEY]["raw"])
    damage = clean - float(f1[s3]["raw"])
    headroom = float(f1[s3]["oracle"]) - float(f1[s3]["raw"])
    recoveries = {
        key: recovery_fraction(
            float(f1[key]["raw"]),
            float(f1[key]["estimated"]),
            float(f1[key]["oracle"]),
        )
        for key in (s2, s3)
    }
    step_needed = damage >= thresholds["raw_damage_min"]
    headroom_ok = headroom >= thresholds["headroom_min"]
    recovery_ok = all(
        value is not None and value >= thresholds["recovery_min"]
        for value in recoveries.values()
    )
    if not anchor_passed:
        verdict = "anchor_failed"
    elif not step_needed:
        verdict = "no_step_needed"
    elif not headroom_ok:
        verdict = "headroom_insufficient"
    elif not recovery_ok:
        verdict = "recovery_insufficient"
    else:
        verdict = "proceed_to_gate2"
    return {
        "verdict": verdict,
        "anchor_passed": bool(anchor_passed),
        "raw_clean_f1": clean,
        "physical_s3_damage": damage,
        "physical_s3_headroom": headroom,
        "recovery": recoveries,
        "step_needed": step_needed,
        "headroom_ok": headroom_ok,
        "recovery_ok": recovery_ok,
        "thresholds": dict(thresholds),
    }
