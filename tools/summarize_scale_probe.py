from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


PRIMARY = "shift:s2:t2"


def _entry(summary: dict[str, Any]) -> dict[str, float]:
    shift4 = summary["per_shift"][PRIMARY]
    shift8 = summary["per_shift"]["shift:s3:t2"]
    return {
        "s4_pck1": float(shift4["raw_pck1"]),
        "s4_pck2": float(shift4["raw_pck2"]),
        "s4_mean_epe": float(shift4["raw_mean_epe"]),
        "s4_samples": int(shift4["samples"]),
        "s8_pck1": float(shift8["raw_pck1"]),
        "s8_excess_boundary": float(
            shift8.get("excess_boundary_fraction", float("nan"))
        ),
        "clean_q_rate": float(
            summary["q_trigger_rate"]["clean:s0:common"]
        ),
    }


def scale_probe_decision(
    per_scale: dict[str, dict[str, Any]],
    *,
    levir_reference: dict[str, Any] | None,
    whu_native_reference: dict[str, Any] | None,
    explain_min: float,
    not_explain_max: float,
    monotone_tolerance: float,
) -> dict[str, Any]:
    """Preregistered reading of the coverage probe.

    The probe holds pixel count, instrument, protocol, and injection
    fixed while varying only ground coverage (larger blocks box-averaged
    down to 512 px, i.e. the same scene imaged by a coarser sensor).

    Primary quantity: Shift-4 conditional PCK@1 at the LARGEST coverage,
    because Shift-4 has the lowest boundary involvement. Reading:

    * >= explain_min AND the S4 PCK@1 trend is monotone non-decreasing
      within monotone_tolerance -> the stage-A failure is a
      scale/coverage phenomenon; the applicability condition must be
      stated in GSD/coverage terms, and WHU may NOT be described as a
      "spatially varying field" negative case;
    * <= not_explain_max -> coverage does not explain the failure; the
      dataset-property attribution stands;
    * otherwise -> partial: both mechanisms contribute and neither may
      be claimed alone.

    This is an attribution instrument. It reopens no gate: the WHU
    localizer stage-A verdict remains failed in every branch.
    """

    if len(per_scale) < 2:
        raise ValueError("Need at least two scales to read a trend")
    ordered = sorted(per_scale, key=lambda name: per_scale[name]["scale_px"])
    table = {}
    for name in ordered:
        scale_info = per_scale[name]
        table[name] = {
            "scale_px": int(scale_info["scale_px"]),
            "coverage_m": float(scale_info["coverage_m"]),
            **_entry(scale_info["summary"]),
        }
    largest = table[ordered[-1]]
    smallest = table[ordered[0]]
    series = [table[name]["s4_pck1"] for name in ordered]
    monotone = all(
        series[index + 1] >= series[index] - monotone_tolerance
        for index in range(len(series) - 1)
    )
    value = largest["s4_pck1"]
    if value >= explain_min and monotone:
        verdict = "coverage_explains_the_failure"
        attribution = "applicability_condition_is_gsd_coverage_not_dataset"
    elif value <= not_explain_max:
        verdict = "coverage_does_not_explain_the_failure"
        attribution = "dataset_property_attribution_stands"
    else:
        verdict = "partial_coverage_effect"
        attribution = "neither_mechanism_may_be_claimed_alone"
    result: dict[str, Any] = {
        "status": "completed",
        "primary_scenario": PRIMARY,
        "per_scale": table,
        "s4_pck1_series": {
            name: table[name]["s4_pck1"] for name in ordered
        },
        "monotone_non_decreasing": monotone,
        "largest_coverage_s4_pck1": value,
        "smallest_coverage_s4_pck1": smallest["s4_pck1"],
        "bands": {
            "explain_min": float(explain_min),
            "not_explain_max": float(not_explain_max),
            "monotone_tolerance": float(monotone_tolerance),
        },
        "verdict": verdict,
        "attribution": attribution,
        "gates_reopened": False,
        "note": (
            "Attribution instrument on train-region content with the "
            "LEVIR P7C instrument (never trained on WHU). The WHU "
            "localizer stage-A verdict remains failed regardless."
        ),
    }
    if levir_reference is not None:
        result["levir_512_reference_s4_pck1"] = float(
            levir_reference["per_shift"][PRIMARY]["raw_pck1"]
        )
    if whu_native_reference is not None:
        result["whu_native_val_s4_pck1"] = float(
            whu_native_reference["per_shift"][PRIMARY]["raw_pck1"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read the WHU scale/coverage probe against the "
            "preregistered bands."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="SPLIT=PATH",
        help="Repeatable: scale0512=path/to/offset_consistency_summary.json",
    )
    parser.add_argument("--probe-summary", required=True)
    parser.add_argument("--levir-reference", default=None)
    parser.add_argument("--whu-native-reference", default=None)
    parser.add_argument("--explain-min", type=float, default=0.70)
    parser.add_argument("--not-explain-max", type=float, default=0.30)
    parser.add_argument(
        "--monotone-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def read(path: str) -> dict[str, Any]:
        return json.loads(
            Path(path).expanduser().read_text(encoding="utf-8")
        )

    probe = read(args.probe_summary)
    per_scale: dict[str, dict[str, Any]] = {}
    for item in args.summary:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"Expected SPLIT=PATH, got: {item}")
        if name not in probe["scales"]:
            raise ValueError(
                f"Split {name} not present in the probe summary"
            )
        per_scale[name] = {
            "scale_px": probe["scales"][name]["scale_px"],
            "coverage_m": probe["scales"][name]["coverage_m"],
            "summary": read(path),
        }
    decision = scale_probe_decision(
        per_scale,
        levir_reference=(
            read(args.levir_reference) if args.levir_reference else None
        ),
        whu_native_reference=(
            read(args.whu_native_reference)
            if args.whu_native_reference
            else None
        ),
        explain_min=float(args.explain_min),
        not_explain_max=float(args.not_explain_max),
        monotone_tolerance=float(args.monotone_tolerance),
    )
    save_json(args.output, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
