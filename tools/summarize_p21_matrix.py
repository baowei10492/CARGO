"""Summarise one P21 cell, and assemble the cross-detector matrix.

P21 asks whether selective alignment is necessary for change detection as a
task rather than useful for one detector. That question is only answered by
comparison, so a cell is not summarised by its own gain: it is summarised by
where the routed variant sits among the alternatives a practitioner would
otherwise choose. Every cell therefore reports the same ladder:

    raw                    do nothing
    always_aligned         align unconditionally
    q_gate                 geometry confidence only, no task routing
    safe_veto_registration registration-confidence router
    safe_veto_task         task-benefit router (primary)
    loss_oracle            per-sample upper bound

Two derived quantities carry the argument and neither is a raw F1:

* **capture** -- routed gain as a fraction of the oracle gain, the quantity
  that exposed the GZ-CD failure that aggregate F1 concealed (main record
  122.6, 138). Reported per scenario and as a mean.
* **break-even p** -- the fraction of already-registered pairs above which
  routing beats unconditional alignment. Without it, "routing wins" is not
  a statement about deployment (second external review, P3).

The verdict logic mirrors the preregistered gates in
docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md and applies them per cell. A cell
whose detector is barely hurt by displacement is recorded as
``no_step_needed`` rather than as a failure: that is applicability
information, and the protocol forbids dropping it from the matrix.

Usage
-----
    python -m tools.summarize_p21_matrix --cell reports/p21/bit_levir/*.json \
        --output reports/p21/bit_levir/cell_summary.json

    python -m tools.summarize_p21_matrix --matrix reports/p21 \
        --output reports/p21/matrix.json --markdown reports/p21/matrix.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

SCENARIOS = ("clean:s0:common", "shift:s1:t2", "shift:s2:t2", "shift:s3:t2")
SHIFT_SCENARIOS = SCENARIOS[1:]
SHIFT_LABELS = {"shift:s1:t2": "2px", "shift:s2:t2": "4px", "shift:s3:t2": "8px"}
LADDER = (
    "raw",
    "always_aligned",
    "q_gate",
    "safe_veto_registration",
    "safe_veto_task",
    "loss_oracle",
)


def _f1(metrics: dict[str, Any], variant: str, scenario: str) -> float:
    try:
        return float(metrics[variant][scenario])
    except KeyError as error:  # pragma: no cover - configuration error
        raise KeyError(
            f"missing F1 for variant {variant!r} scenario {scenario!r}"
        ) from error


def _mean_shift(metrics: dict[str, Any], variant: str) -> float:
    return sum(_f1(metrics, variant, s) for s in SHIFT_SCENARIOS) / len(
        SHIFT_SCENARIOS
    )


def break_even_fraction(
    routed_clean: float,
    routed_shift: float,
    always_clean: float,
    always_shift: float,
) -> float | None:
    """Clean fraction p above which routing beats unconditional alignment.

    Solves ``routed_clean*p + routed_shift*(1-p) = always_clean*p +
    always_shift*(1-p)``. Returns None when the two lines are parallel, and
    values outside [0, 1] are reported as-is because they are meaningful:
    p <= 0 means routing always wins, p >= 1 means it never does.
    """

    denominator = (routed_clean - routed_shift) - (always_clean - always_shift)
    if abs(denominator) < 1e-12:
        return None
    return (always_shift - routed_shift) / denominator


def summarise_cell(
    metrics: dict[str, Any],
    *,
    detector: str,
    dataset: str,
    gates: dict[str, float],
) -> dict[str, Any]:
    """Apply the preregistered gates to one detector/dataset cell."""

    missing = [v for v in LADDER if v not in metrics]
    if missing:
        raise KeyError(f"cell is missing variants: {missing}")

    raw_clean = _f1(metrics, "raw", SCENARIOS[0])
    raw_shift8 = _f1(metrics, "raw", "shift:s3:t2")
    routed_clean = _f1(metrics, "safe_veto_task", SCENARIOS[0])

    per_scenario: dict[str, Any] = {}
    for scenario in SHIFT_SCENARIOS:
        raw_value = _f1(metrics, "raw", scenario)
        routed = _f1(metrics, "safe_veto_task", scenario)
        oracle = _f1(metrics, "loss_oracle", scenario)
        available = oracle - raw_value
        captured = routed - raw_value
        per_scenario[SHIFT_LABELS[scenario]] = {
            "raw_f1": raw_value,
            "routed_f1": routed,
            "oracle_f1": oracle,
            "available_gain": available,
            "captured_gain": captured,
            "capture_rate": (captured / available) if available > 0 else None,
        }

    routed_shift_mean = _mean_shift(metrics, "safe_veto_task")
    raw_shift_mean = _mean_shift(metrics, "raw")
    oracle_shift_mean = _mean_shift(metrics, "loss_oracle")
    mean_available = oracle_shift_mean - raw_shift_mean
    mean_captured = routed_shift_mean - raw_shift_mean

    clean_drop = raw_clean - routed_clean
    displacement_damage = raw_clean - raw_shift8

    # Precondition (a): if displacement barely hurts this detector, there is
    # nothing for the step to repair. Recorded, never dropped.
    step_needed = displacement_damage >= gates["raw_shift8_drop_min"]
    clean_ok = clean_drop <= gates["clean_f1_drop_max"]
    gain_ok = mean_captured >= gates["shift_mean_gain_min"]

    if not step_needed:
        verdict = "no_step_needed"
    elif clean_ok and gain_ok:
        verdict = "step_necessary_and_effective"
    elif gain_ok and not clean_ok:
        verdict = "effective_but_harms_clean"
    else:
        verdict = "step_ineffective"

    break_even = break_even_fraction(
        routed_clean,
        routed_shift_mean,
        _f1(metrics, "always_aligned", SCENARIOS[0]),
        _mean_shift(metrics, "always_aligned"),
    )

    return {
        "detector": detector,
        "dataset": dataset,
        "verdict": verdict,
        "preconditions": {
            "raw_clean_f1": raw_clean,
            "raw_shift8_f1": raw_shift8,
            "displacement_damage": displacement_damage,
            "damage_threshold": gates["raw_shift8_drop_min"],
            "step_needed": step_needed,
        },
        "gates": {
            "clean_f1_drop": {
                "value": clean_drop,
                "maximum": gates["clean_f1_drop_max"],
                "passed": clean_ok,
            },
            "shift_mean_gain": {
                "value": mean_captured,
                "minimum": gates["shift_mean_gain_min"],
                "passed": gain_ok,
            },
        },
        "ladder_mean_shift_f1": {
            variant: _mean_shift(metrics, variant) for variant in LADDER
        },
        "ladder_clean_f1": {
            variant: _f1(metrics, variant, SCENARIOS[0]) for variant in LADDER
        },
        "capture": {
            "per_scenario": per_scenario,
            "mean_available_gain": mean_available,
            "mean_captured_gain": mean_captured,
            "mean_capture_rate": (
                mean_captured / mean_available if mean_available > 0 else None
            ),
            "report_only_threshold": gates.get("mean_capture_report_only"),
        },
        "break_even_clean_fraction": break_even,
        "notes": [
            "Capture is routed gain over oracle gain; aggregate F1 can hide a "
            "router that does not rank (main record 138).",
            "Break-even p is the clean fraction above which routing beats "
            "unconditional alignment.",
            "A no_step_needed verdict is applicability information, not a "
            "failure, and must stay in the matrix.",
        ],
    }


def _markdown(cells: list[dict[str, Any]]) -> str:
    detectors: list[str] = []
    datasets: list[str] = []
    for cell in cells:
        if cell["detector"] not in detectors:
            detectors.append(cell["detector"])
        if cell["dataset"] not in datasets:
            datasets.append(cell["dataset"])
    index = {(c["detector"], c["dataset"]): c for c in cells}

    lines = ["## Necessity matrix", ""]
    lines.append("| Detector | " + " | ".join(datasets) + " |")
    lines.append("|---|" + "---|" * len(datasets))
    symbol = {
        "step_necessary_and_effective": "PASS",
        "effective_but_harms_clean": "clean",
        "step_ineffective": "FAIL",
        "no_step_needed": "n/a",
    }
    for detector in detectors:
        row = [detector]
        for dataset in datasets:
            cell = index.get((detector, dataset))
            if cell is None:
                row.append("--")
                continue
            gain = cell["gates"]["shift_mean_gain"]["value"]
            drop = cell["gates"]["clean_f1_drop"]["value"]
            row.append(
                f"{symbol[cell['verdict']]} "
                f"(+{gain:.4f} / -{drop:.4f})"
            )
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "Cell entries are the verdict with (mean shift gain / clean "
              "F1 drop).", "", "## Per-cell detail", ""]
    lines.append(
        "| Detector | Dataset | Raw damage @8px | Mean gain | Capture | "
        "Break-even p |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for cell in cells:
        capture = cell["capture"]["mean_capture_rate"]
        break_even = cell["break_even_clean_fraction"]
        lines.append(
            f"| {cell['detector']} | {cell['dataset']} "
            f"| {cell['preconditions']['displacement_damage']:.4f} "
            f"| {cell['gates']['shift_mean_gain']['value']:+.4f} "
            f"| {'n/a' if capture is None else f'{capture:.1%}'} "
            f"| {'n/a' if break_even is None else f'{break_even:.3f}'} |"
        )
    return "\n".join(lines) + "\n"


def metrics_from_scenario_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Pivot the classifier's validation_scenario_metrics.csv into a cell.

    A duplicate (method, scenario) row is an error: keeping the last one
    would let a re-run silently overwrite the reported cell.
    """

    metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        method = str(row["method"])
        scenario = str(row["scenario"])
        by_scenario = metrics.setdefault(method, {})
        if scenario in by_scenario:
            raise ValueError(f"duplicate row for {method!r} {scenario!r}")
        by_scenario[scenario] = float(row["f1"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", help="scenario metrics JSON for one cell")
    parser.add_argument(
        "--scenario-csv",
        help="validation_scenario_metrics.csv written by the classifier tool",
    )
    parser.add_argument("--detector")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--matrix", help="directory of cell summaries to assemble"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--raw-shift8-drop-min", type=float, default=0.05)
    parser.add_argument("--clean-f1-drop-max", type=float, default=0.001)
    parser.add_argument("--shift-mean-gain-min", type=float, default=0.020)
    parser.add_argument("--mean-capture-report-only", type=float, default=0.40)
    args = parser.parse_args()

    gates = {
        "raw_shift8_drop_min": args.raw_shift8_drop_min,
        "clean_f1_drop_max": args.clean_f1_drop_max,
        "shift_mean_gain_min": args.shift_mean_gain_min,
        "mean_capture_report_only": args.mean_capture_report_only,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.cell or args.scenario_csv:
        if not (args.detector and args.dataset):
            raise SystemExit(
                "--cell/--scenario-csv require --detector and --dataset"
            )
        if args.scenario_csv:
            with Path(args.scenario_csv).open(encoding="utf-8", newline="") as handle:
                metrics = metrics_from_scenario_rows(csv.DictReader(handle))
        else:
            metrics = json.loads(Path(args.cell).read_text(encoding="utf-8"))
            metrics = metrics.get("variants", metrics)
        summary = summarise_cell(
            metrics,
            detector=args.detector,
            dataset=args.dataset,
            gates=gates,
        )
        output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"{summary['detector']} x {summary['dataset']}: "
              f"{summary['verdict']}")
        print(f"  displacement damage @8px : "
              f"{summary['preconditions']['displacement_damage']:.4f}")
        print(f"  clean drop  {summary['gates']['clean_f1_drop']['value']:+.4f}"
              f"  (max {gates['clean_f1_drop_max']})")
        print(f"  mean gain   "
              f"{summary['gates']['shift_mean_gain']['value']:+.4f}"
              f"  (min {gates['shift_mean_gain_min']})")
        capture = summary["capture"]["mean_capture_rate"]
        print(f"  capture     "
              f"{'n/a' if capture is None else f'{capture:.1%}'}")
        print(f"wrote {output}")
        return

    if not args.matrix:
        raise SystemExit("provide either --cell or --matrix")

    cells = []
    for path in sorted(Path(args.matrix).rglob("cell_summary.json")):
        cells.append(json.loads(path.read_text(encoding="utf-8")))
    if not cells:
        raise SystemExit(f"no cell_summary.json found under {args.matrix}")

    verdicts = {c["verdict"] for c in cells}
    tested = [c for c in cells if c["verdict"] != "no_step_needed"]
    matrix = {
        "status": "completed",
        "cells": cells,
        "cell_count": len(cells),
        "detectors": sorted({c["detector"] for c in cells}),
        "datasets": sorted({c["dataset"] for c in cells}),
        "all_tested_cells_pass": bool(tested) and all(
            c["verdict"] == "step_necessary_and_effective" for c in tested
        ),
        "verdicts_present": sorted(verdicts),
        "claim": (
            "Selective alignment is necessary for every tested detector"
            if tested
            and all(
                c["verdict"] == "step_necessary_and_effective" for c in tested
            )
            else "Necessity does not hold uniformly; report the exceptions"
        ),
    }
    output.write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        Path(args.markdown).write_text(_markdown(cells), encoding="utf-8")
    print(_markdown(cells))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
