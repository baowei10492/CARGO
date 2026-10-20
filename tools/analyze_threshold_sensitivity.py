from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


"""Re-analyse the frozen risk--coverage sweep as a threshold sensitivity study.

Reviewers reliably ask two questions about a calibrated operating point:
how the threshold was chosen, and what would happen at a different one.
Both are answerable from artifacts that already exist. The safe-alignment
run stored a 101-point tau sweep over the same validation decisions that
produced the frozen threshold, so this tool reads that sweep and reports
adoption, harm, and captured benefit as functions of tau.

This is re-analysis only. It selects no threshold, defines no new gate,
reads no test data, and does not change the frozen operating point; the
frozen tau is marked on the curve as a reference, not re-derived.
"""


CLEAN_SCENARIO = "clean:s0:common"
SHIFT_SCENARIOS = ("shift:s1:t2", "shift:s2:t2", "shift:s3:t2")
SCENARIO_LABELS = {
    CLEAN_SCENARIO: "Clean",
    "shift:s1:t2": "Shift-2",
    "shift:s2:t2": "Shift-4",
    "shift:s3:t2": "Shift-8",
}


def read_sweep(path: Path) -> dict[float, dict[str, dict[str, float]]]:
    sweep: dict[float, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sweep[round(float(row["tau"]), 6)][row["scenario"]] = {
                "q_candidates": float(row["q_candidates"]),
                "q_benefit": float(row["q_benefit"]),
                "accepted": float(row["accepted"]),
                "accepted_harmful": float(row["accepted_harmful"]),
                "accepted_benefit": float(row["accepted_benefit"]),
            }
    if not sweep:
        raise ValueError(f"No sweep rows in {path}")
    return dict(sweep)


def summarise_tau(
    scenarios: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Aggregate one tau row into deployment-relevant quantities."""
    clean = scenarios.get(CLEAN_SCENARIO, {})
    shift_rows = [
        scenarios[key] for key in SHIFT_SCENARIOS if key in scenarios
    ]
    shift_candidates = sum(row["q_candidates"] for row in shift_rows)
    shift_accepted = sum(row["accepted"] for row in shift_rows)
    shift_available = sum(row["q_benefit"] for row in shift_rows)
    shift_captured = sum(row["accepted_benefit"] for row in shift_rows)
    total_harmful = sum(
        row["accepted_harmful"] for row in scenarios.values()
    )
    return {
        "clean_accepted": clean.get("accepted", 0.0),
        "clean_harmful": clean.get("accepted_harmful", 0.0),
        "clean_benefit": clean.get("accepted_benefit", 0.0),
        "shift_candidates": shift_candidates,
        "shift_accepted": shift_accepted,
        "shift_adoption_rate": (
            shift_accepted / shift_candidates
            if shift_candidates
            else 0.0
        ),
        "shift_available_benefit": shift_available,
        "shift_captured_benefit": shift_captured,
        "shift_captured_fraction": (
            shift_captured / shift_available
            if shift_available
            else 0.0
        ),
        "total_harmful_adoptions": total_harmful,
    }


def clean_safe_ceiling(
    sweep: dict[float, dict[str, dict[str, float]]],
) -> float | None:
    """Smallest tau at which no clean pair is adopted at all.

    The frozen rule sets tau from the worst harmful clean calibration
    sample, so the interesting comparison is where clean adoption
    disappears on validation.
    """
    safe = [
        tau
        for tau, scenarios in sweep.items()
        if scenarios.get(CLEAN_SCENARIO, {}).get("accepted", 0.0) == 0.0
    ]
    return min(safe) if safe else None


def sensitivity_table(
    sweep: dict[float, dict[str, dict[str, float]]],
    *,
    frozen_tau: float,
    reference_taus: tuple[float, ...],
) -> list[dict[str, Any]]:
    available = sorted(sweep)

    def nearest(target: float) -> float:
        return min(available, key=lambda value: abs(value - target))

    rows: list[dict[str, Any]] = []
    seen: set[float] = set()
    for target in (*reference_taus, frozen_tau):
        tau = nearest(target)
        if tau in seen:
            continue
        seen.add(tau)
        summary = summarise_tau(sweep[tau])
        rows.append(
            {
                "tau": tau,
                "is_frozen_operating_point": abs(tau - nearest(frozen_tau))
                < 1e-9,
                **summary,
            }
        )
    return sorted(rows, key=lambda row: row["tau"])


def build_report(
    task_sweep: dict[float, dict[str, dict[str, float]]],
    registration_sweep: dict[float, dict[str, dict[str, float]]] | None,
    *,
    frozen_task_tau: float,
    frozen_registration_tau: float | None,
    reference_taus: tuple[float, ...],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "completed",
        "analysis_type": "reanalysis_of_frozen_validation_sweep",
        "selects_threshold": False,
        "defines_new_gate": False,
        "test_split_touched": False,
        "note": (
            "The frozen operating point is reported as a reference on an "
            "existing sweep. Nothing here re-derives or replaces it."
        ),
        "task_router": {
            "frozen_tau": frozen_task_tau,
            "clean_adoption_disappears_at_tau": clean_safe_ceiling(
                task_sweep
            ),
            "rows": sensitivity_table(
                task_sweep,
                frozen_tau=frozen_task_tau,
                reference_taus=reference_taus,
            ),
        },
    }
    if registration_sweep is not None and frozen_registration_tau:
        report["registration_router"] = {
            "frozen_tau": frozen_registration_tau,
            "clean_adoption_disappears_at_tau": clean_safe_ceiling(
                registration_sweep
            ),
            "rows": sensitivity_table(
                registration_sweep,
                frozen_tau=frozen_registration_tau,
                reference_taus=reference_taus,
            ),
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "| tau | Shift adoption | Captured benefit | Clean adopted | "
        "Harmful adoptions |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["task_router"]["rows"]:
        marker = " (frozen)" if row["is_frozen_operating_point"] else ""
        lines.append(
            "| {tau:.4f}{marker} | {adoption:.1%} | {captured:.1%} | "
            "{clean:.0f} | {harmful:.0f} |".format(
                tau=row["tau"],
                marker=marker,
                adoption=row["shift_adoption_rate"],
                captured=row["shift_captured_fraction"],
                clean=row["clean_accepted"],
                harmful=row["total_harmful_adoptions"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report adoption, harm and captured benefit as functions of "
            "the router threshold, from the frozen validation sweep."
        )
    )
    parser.add_argument("--task-sweep", required=True)
    parser.add_argument("--registration-sweep", default=None)
    parser.add_argument("--frozen-task-tau", type=float, required=True)
    parser.add_argument(
        "--frozen-registration-tau", type=float, default=None
    )
    parser.add_argument(
        "--reference-taus",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.7, 0.8, 0.85, 0.95, 0.99],
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    task_sweep = read_sweep(Path(args.task_sweep))
    registration_sweep = (
        read_sweep(Path(args.registration_sweep))
        if args.registration_sweep
        else None
    )
    report = build_report(
        task_sweep,
        registration_sweep,
        frozen_task_tau=float(args.frozen_task_tau),
        frozen_registration_tau=args.frozen_registration_tau,
        reference_taus=tuple(args.reference_taus),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_markdown:
        Path(args.output_markdown).write_text(
            render_markdown(report), encoding="utf-8"
        )
    print(render_markdown(report))
    print(
        "clean adoption disappears at tau = "
        f"{report['task_router']['clean_adoption_disappears_at_tau']}"
    )


if __name__ == "__main__":
    main()
