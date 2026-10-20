from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def retained_benefit_curve(
    rows: list[dict[str, str]],
    *,
    scenario_prefix: str = "shift:",
) -> list[tuple[float, float]]:
    """Fraction of available shift benefit retained, as a function of tau.

    Threshold-free view of a veto classifier: at every tau it reports
    how much of the benefit available to the q gate the classifier still
    accepts. Comparing two classifiers' whole curves separates
    DISCRIMINATION (curve shape/dominance) from THRESHOLD PLACEMENT
    (which single tau the risk rule happens to pick).
    """

    totals: dict[float, list[float]] = {}
    for row in rows:
        if not str(row["scenario"]).startswith(scenario_prefix):
            continue
        tau = float(row["tau"])
        entry = totals.setdefault(tau, [0.0, 0.0])
        entry[0] += float(row["accepted_benefit"])
        entry[1] += float(row["q_benefit"])
    curve = []
    for tau in sorted(totals):
        accepted, available = totals[tau]
        curve.append(
            (tau, accepted / available if available > 1e-12 else 0.0)
        )
    return curve


def _interpolate(curve: list[tuple[float, float]], tau: float) -> float:
    if not curve:
        raise ValueError("Empty curve")
    ordered = sorted(curve)
    if tau <= ordered[0][0]:
        return ordered[0][1]
    if tau >= ordered[-1][0]:
        return ordered[-1][1]
    for left, right in zip(ordered, ordered[1:]):
        if left[0] <= tau <= right[0]:
            span = right[0] - left[0]
            if span <= 1e-12:
                return right[1]
            weight = (tau - left[0]) / span
            return left[1] + weight * (right[1] - left[1])
    raise RuntimeError("Interpolation interval was not found")


def _area(curve: list[tuple[float, float]]) -> float:
    if len(curve) < 2:
        return 0.0
    total = 0.0
    for first, second in zip(curve, curve[1:]):
        total += 0.5 * (first[1] + second[1]) * (second[0] - first[0])
    span = curve[-1][0] - curve[0][0]
    return total / span if span > 1e-12 else 0.0


def attribute_regression(
    baseline_rows: list[dict[str, str]],
    expanded_rows: list[dict[str, str]],
    *,
    baseline_tau: float,
    expanded_tau: float,
) -> dict[str, Any]:
    """Split a veto regression into discrimination vs threshold parts.

    ATTRIBUTION ONLY. The quantities below are diagnostic: reading the
    expanded classifier at the baseline's tau is NOT a deployable
    configuration, because a threshold imported from another run's
    risk-calibration split carries no risk guarantee. Nothing here
    reopens or revises any gate.
    """

    baseline = retained_benefit_curve(baseline_rows)
    expanded = retained_benefit_curve(expanded_rows)
    baseline_at_own = _interpolate(baseline, baseline_tau)
    expanded_at_own = _interpolate(expanded, expanded_tau)
    expanded_at_baseline_tau = _interpolate(expanded, baseline_tau)
    observed = expanded_at_own - baseline_at_own
    discrimination = expanded_at_baseline_tau - baseline_at_own
    threshold = expanded_at_own - expanded_at_baseline_tau
    if abs(observed) < 1e-9:
        dominant = "no_change"
    elif abs(threshold) > abs(discrimination):
        dominant = "threshold_placement"
    else:
        dominant = "discrimination"
    return {
        "status": "completed",
        "attribution_only_not_a_verdict": True,
        "baseline_tau": float(baseline_tau),
        "expanded_tau": float(expanded_tau),
        "tau_shift": float(expanded_tau) - float(baseline_tau),
        "baseline_retained_at_own_tau": baseline_at_own,
        "expanded_retained_at_own_tau": expanded_at_own,
        "expanded_retained_at_baseline_tau": expanded_at_baseline_tau,
        "observed_change": observed,
        "discrimination_component": discrimination,
        "threshold_component": threshold,
        "dominant_cause": dominant,
        "mean_retained_benefit_area": {
            "baseline": _area(baseline),
            "expanded": _area(expanded),
        },
        "curves": {
            "baseline": [
                {"tau": tau, "retained": value} for tau, value in baseline
            ],
            "expanded": [
                {"tau": tau, "retained": value} for tau, value in expanded
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute a veto-classifier gain change to discrimination "
            "versus threshold placement, using only risk-coverage CSVs "
            "already written by previous runs. Read-only, no GPU, no "
            "validation data, no gate is reopened."
        )
    )
    parser.add_argument("--baseline-coverage", required=True)
    parser.add_argument("--expanded-coverage", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--expanded-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def tau_of(path: str) -> float:
        data = json.loads(
            Path(path).expanduser().read_text(encoding="utf-8")
        )
        for key in ("probability_threshold", "tau"):
            if key in data:
                return float(data[key])
        raise KeyError(
            f"No probability_threshold in {path}; keys={list(data)}"
        )

    decision = attribute_regression(
        _read(Path(args.baseline_coverage).expanduser()),
        _read(Path(args.expanded_coverage).expanduser()),
        baseline_tau=tau_of(args.baseline_summary),
        expanded_tau=tau_of(args.expanded_summary),
    )
    decision["sources"] = {
        "baseline_coverage": str(args.baseline_coverage),
        "expanded_coverage": str(args.expanded_coverage),
    }
    save_json(args.output, decision)
    printable = {
        key: value
        for key, value in decision.items()
        if key != "curves"
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
