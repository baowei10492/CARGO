from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


"""Disentangle ground coverage from working GSD in the WHU failure.

The original scale probe grew the analysis window and downsampled every
block back to 512 pixels, so coverage and effective GSD moved together
and main record 77 recorded the two sub-mechanisms as entangled. This
probe holds GSD at the native 0.075 m and varies only the window, so a
recovery here isolates coverage, and a flat curve isolates resolution.

Nothing is trained and no split is consumed: the frozen LEVIR localizer
runs zero-shot on train-raster blocks, exactly as the original probe did.
"""


VERDICTS = {
    "coverage": (
        "Localization recovers at native resolution once the window is "
        "large enough, so the limiting factor is ground coverage. A "
        "deployment can keep full resolution and enlarge the analysis "
        "window."
    ),
    "gsd": (
        "Localization stays low at native resolution regardless of "
        "window size, so the limiting factor is working GSD. A "
        "deployment must coarsen before estimating the offset."
    ),
    "both": (
        "Enlarging the window at native resolution recovers part but "
        "not all of the accuracy seen when the block was also "
        "downsampled, so coverage and GSD both contribute."
    ),
    "inconclusive": (
        "The native-window curve does not separate the hypotheses at the "
        "preregistered margins."
    ),
}


def read_pck1(path: Path, scenario: str) -> float:
    summary = json.loads(path.read_text(encoding="utf-8"))
    return float(summary["per_shift"][scenario]["raw_pck1"])


def classify(
    native: dict[int, float],
    downsampled: dict[int, float],
    *,
    baseline_scale: int,
    recovery_min: float,
    flat_max: float,
) -> dict[str, Any]:
    """Route the outcome using margins fixed before the run.

    ``native`` and ``downsampled`` map window size in pixels to PCK@1.
    The baseline is the smallest window, where both probes are the same
    configuration and must therefore agree.
    """
    largest = max(native)
    baseline = native[baseline_scale]
    native_gain = native[largest] - baseline
    downsampled_gain = downsampled[largest] - downsampled[baseline_scale]

    if native[largest] >= recovery_min:
        verdict = "coverage"
    elif native_gain <= flat_max:
        verdict = "gsd"
    elif downsampled_gain > 0 and native_gain / downsampled_gain >= 0.5:
        verdict = "coverage"
    elif native_gain > flat_max:
        verdict = "both"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "interpretation": VERDICTS[verdict],
        "baseline_window_px": baseline_scale,
        "largest_window_px": largest,
        "native_gsd_pck1": dict(sorted(native.items())),
        "downsampled_reference_pck1": dict(sorted(downsampled.items())),
        "native_gain": native_gain,
        "downsampled_gain": downsampled_gain,
        "recovered_fraction_of_downsampled_gain": (
            native_gain / downsampled_gain
            if downsampled_gain > 0
            else None
        ),
        "thresholds": {
            "recovery_min": recovery_min,
            "flat_max": flat_max,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native-resolution window growth against the "
            "downsampled scale probe to separate coverage from GSD."
        )
    )
    parser.add_argument(
        "--native",
        action="append",
        required=True,
        metavar="SCALE=PATH",
        help="Window size and its offset_consistency_summary.json",
    )
    parser.add_argument(
        "--downsampled",
        action="append",
        required=True,
        metavar="SCALE=PATH",
        help="Original probe summaries, one per window size",
    )
    parser.add_argument("--scenario", default="shift:s2:t2")
    parser.add_argument("--baseline-scale", type=int, default=512)
    parser.add_argument("--recovery-min", type=float, default=0.70)
    parser.add_argument("--flat-max", type=float, default=0.10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    def parse(items: list[str]) -> dict[int, float]:
        table: dict[int, float] = {}
        for item in items:
            scale, path = item.split("=", 1)
            table[int(scale)] = read_pck1(Path(path), args.scenario)
        return table

    native = parse(args.native)
    downsampled = parse(args.downsampled)
    if args.baseline_scale not in native:
        raise SystemExit(
            f"Baseline window {args.baseline_scale} missing from native runs"
        )

    report = {
        "status": "completed",
        "analysis_type": "mechanism_probe_no_training",
        "scenario": args.scenario,
        "test_split_touched": False,
        "labels_read": False,
        **classify(
            native,
            downsampled,
            baseline_scale=int(args.baseline_scale),
            recovery_min=float(args.recovery_min),
            flat_max=float(args.flat_max),
        ),
        "notes": [
            "Both probes use the frozen LEVIR localizer zero-shot on WHU "
            "train-raster blocks; neither trains anything.",
            "The downsampled reference is the original probe from main "
            "record 77, where coverage and GSD moved together.",
        ],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{'window px':>10s} {'native GSD':>12s} {'downsampled':>12s}")
    for scale in sorted(native):
        reference = downsampled.get(scale)
        print(
            f"{scale:>10d} {native[scale]:>12.4f} "
            f"{reference if reference is None else f'{reference:12.4f}'}"
        )
    print(f"\nverdict: {report['verdict']}")
    print(report["interpretation"])


if __name__ == "__main__":
    main()
