from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "median_injected_offset_magnitude",
    "median_max_pairwise_tile_distance",
    "median_tile_error_magnitude",
    "tile_error_pck2",
    "tile_error_pck4",
    "median_global_error_magnitude",
)


def instrument_diagnosis(
    summary: dict[str, Any],
    *,
    search_radius_pixels: float,
) -> dict[str, Any]:
    """Expose estimator diagnostics without adding a new decision gate."""

    rows: list[dict[str, Any]] = []
    for scenario, record in summary["scenarios"].items():
        if not scenario.startswith("shift:"):
            continue
        row = {"scenario": scenario}
        for field in FIELDS:
            row[field] = record.get(field)
        rows.append(row)
    rows.sort(key=lambda item: str(item["scenario"]))
    if not rows:
        raise ValueError("No injected shift scenarios in hygiene summary")

    severity1 = next(
        (
            row
            for row in rows
            if str(row["scenario"]).startswith("shift:s1:")
        ),
        None,
    )
    if severity1 is None:
        raise ValueError("Missing severity-1 injected shift scenario")
    s1_dispersion = float(
        severity1["median_max_pairwise_tile_distance"]
    )
    near_boundary = s1_dispersion >= 0.9 * float(search_radius_pixels)
    weak_s1_pck4 = (
        severity1["tile_error_pck4"] is not None
        and float(severity1["tile_error_pck4"]) <= 0.5
    )
    return {
        "status": "completed",
        "diagnostic_only_no_gate": True,
        "search_radius_pixels": float(search_radius_pixels),
        "scenarios": rows,
        "checks": {
            "severity1_dispersion_near_search_boundary": {
                "value": s1_dispersion,
                "threshold": 0.9 * float(search_radius_pixels),
                "flagged": near_boundary,
            },
            "severity1_tile_pck4_weak": {
                "value": severity1["tile_error_pck4"],
                "threshold": 0.5,
                "flagged": bool(weak_s1_pck4),
            },
        },
        "instrument_warning": bool(near_boundary or weak_s1_pck4),
        "interpretation": (
            "A warning indicates boundary clamping or estimator-domain "
            "failure remains plausible. It does not diagnose a "
            "spatially varying field and does not change the "
            "preregistered Gate 0 or Gate 1 route."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report P15 Gate 0 estimator-support diagnostics before "
            "running the decisive input-oracle headroom gate."
        )
    )
    parser.add_argument("--hygiene", required=True)
    parser.add_argument("--search-radius-pixels", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = json.loads(
        Path(args.hygiene).read_text(encoding="utf-8")
    )
    result = instrument_diagnosis(
        summary,
        search_radius_pixels=float(args.search_radius_pixels),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("scenario           injected  dispersion  tile_error   pck2   pck4")
    for row in result["scenarios"]:
        def show(key: str) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{float(value):.3f}"

        print(
            f"{row['scenario']:18s} "
            f"{show('median_injected_offset_magnitude'):>8s} "
            f"{show('median_max_pairwise_tile_distance'):>11s} "
            f"{show('median_tile_error_magnitude'):>10s} "
            f"{show('tile_error_pck2'):>6s} "
            f"{show('tile_error_pck4'):>6s}"
        )
    print(
        "instrument_warning:",
        str(result["instrument_warning"]).lower(),
    )
    print(result["interpretation"])


if __name__ == "__main__":
    main()
