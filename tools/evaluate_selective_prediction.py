from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json
from tools.screen_tile_consistency_features import roc_auc


CLEAN_SCENARIO = "clean:s0:common"


def expected_calibration_error(
    probabilities: list[float],
    labels: list[bool],
    *,
    bins: int,
) -> dict[str, Any]:
    """Standard binned ECE, plus the per-bin table it is computed from.

    The table is reported alongside the scalar because a single ECE
    number hides which side of the diagonal the model sits on, and an
    over-confident and an under-confident classifier can share an ECE.
    """

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must align")
    if not probabilities:
        raise ValueError("No samples")
    table: list[dict[str, Any]] = []
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if (probability > low or (index == 0 and probability >= low))
            and probability <= high
        ]
        if not selected:
            table.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "samples": 0,
                    "mean_confidence": None,
                    "empirical_safe_rate": None,
                }
            )
            continue
        confidence = sum(item[0] for item in selected) / len(selected)
        accuracy = sum(1 for item in selected if item[1]) / len(selected)
        error += (len(selected) / total) * abs(accuracy - confidence)
        table.append(
            {
                "bin_low": low,
                "bin_high": high,
                "samples": len(selected),
                "mean_confidence": confidence,
                "empirical_safe_rate": accuracy,
            }
        )
    return {"expected_calibration_error": error, "bins": table}


def risk_coverage(
    rows: list[dict[str, Any]],
    *,
    probability_key: str,
    grid_points: int,
) -> dict[str, Any]:
    """Risk-coverage over the q-gated population, and its area.

    Coverage is measured against the q gate's candidates, not the whole
    validation set: the veto can only ever decline what the q gate first
    proposed, so the alternative denominator would credit it for samples
    it never had the option to accept.

    The oracle curve orders the same candidates by their realized
    benefit; the gap between the two areas is the part of the risk that
    better ranking could still remove.
    """

    candidates = [row for row in rows if row["q_gate"]]
    if not candidates:
        raise ValueError("No q-gated candidates")
    total = len(candidates)
    harmful_total = sum(
        1 for row in candidates if row["advantage"] <= 0.0
    )
    curve: list[dict[str, float]] = []
    for index in range(grid_points):
        tau = index / max(1, grid_points - 1)
        accepted = [
            row for row in candidates if row[probability_key] >= tau
        ]
        coverage = len(accepted) / total
        harmful = sum(1 for row in accepted if row["advantage"] <= 0.0)
        curve.append(
            {
                "tau": tau,
                "coverage": coverage,
                "risk": harmful / len(accepted) if accepted else 0.0,
                "harmful_count": harmful,
                "accepted": len(accepted),
                "mean_advantage": (
                    sum(row["advantage"] for row in accepted)
                    / len(accepted)
                    if accepted
                    else 0.0
                ),
            }
        )

    # AURC is computed from the RANKING, not from the tau sweep. A tau
    # sweep lands on whatever coverages the score distribution happens to
    # produce, while the oracle ordering has one point per k -- different
    # supports, so their trapezoid areas are not comparable and the
    # "excess" could even come out negative. Ranking both by their own
    # score gives both curves the identical support k/N, and the oracle
    # ordering then minimises cumulative harm at every k, so the excess
    # is non-negative by construction.
    def ranked_aurc(key: str) -> float:
        ordered = sorted(
            candidates,
            key=lambda row: row[key],
            reverse=True,
        )
        harmful_seen = 0
        risks: list[float] = []
        for position, row in enumerate(ordered, start=1):
            if row["advantage"] <= 0.0:
                harmful_seen += 1
            risks.append(harmful_seen / position)
        return sum(risks) / len(risks)

    measured_area = ranked_aurc(probability_key)
    oracle_area = ranked_aurc("advantage")
    return {
        "candidates": total,
        "harmful_candidates": harmful_total,
        "base_risk_at_full_coverage": harmful_total / total,
        "aurc": measured_area,
        "oracle_aurc": oracle_area,
        "excess_aurc": measured_area - oracle_area,
        "curve": curve,
    }


def selective_metrics(
    rows: list[dict[str, Any]],
    *,
    probability_key: str,
    gate_key: str,
    bins: int,
    grid_points: int,
    deployed_tau: float | None,
) -> dict[str, Any]:
    """Descriptive selective-prediction report. NOT a gate.

    Validation was consumed once, for the fixed P7-3 protocol. Everything
    here is a re-analysis of that same recorded run: no threshold is
    chosen from these numbers and no verdict depends on them.

    Per-scenario stratification is mandatory, not optional. Pooled across
    scenarios the label is largely determined by the scenario itself
    (clean pairs are mostly unsafe to align, shifted pairs mostly safe),
    so a pooled AUROC mostly measures scenario separability -- the
    confound recorded as F2 in main record 69.2.
    """

    scenarios = []
    for row in rows:
        if row["scenario"] not in scenarios:
            scenarios.append(row["scenario"])

    def auc_for(subset: list[dict[str, Any]]) -> float | None:
        if len(subset) < 2:
            return None
        labels = [row["advantage"] > 0.0 for row in subset]
        if len(set(labels)) < 2:
            return None
        return roc_auc(
            [float(row[probability_key]) for row in subset],
            labels,
        )

    per_scenario = {}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        labels = [row["advantage"] > 0.0 for row in subset]
        entry: dict[str, Any] = {
            "samples": len(subset),
            "safe_rate": (
                sum(labels) / len(labels) if labels else None
            ),
            "auroc": auc_for(subset),
        }
        if len(set(labels)) >= 2:
            entry.update(
                expected_calibration_error(
                    [float(row[probability_key]) for row in subset],
                    labels,
                    bins=bins,
                )
            )
        else:
            entry["expected_calibration_error"] = None
            entry["degenerate_reason"] = "single_class_in_scenario"
        if deployed_tau is not None:
            accepted = [
                row
                for row in subset
                if row["q_gate"]
                and float(row[probability_key]) >= deployed_tau
            ]
            harmful = sum(
                1 for row in accepted if row["advantage"] <= 0.0
            )
            entry["deployed_operating_point"] = {
                "accepted": len(accepted),
                "acceptance_rate": len(accepted) / len(subset),
                "harmful_accepted": harmful,
                "risk": (
                    harmful / len(accepted) if accepted else 0.0
                ),
            }
        per_scenario[scenario] = entry

    labels = [row["advantage"] > 0.0 for row in rows]
    result: dict[str, Any] = {
        "status": "completed",
        "reporting_only_no_gate": True,
        "probability_key": probability_key,
        "gate_key": gate_key,
        "samples": len(rows),
        "pooled_auroc_confounded_by_scenario": auc_for(rows),
        "pooled_safe_rate": sum(labels) / len(labels),
        "per_scenario": per_scenario,
        "shift_only_auroc": auc_for(
            [row for row in rows if row["scenario"] != CLEAN_SCENARIO]
        ),
        "risk_coverage": risk_coverage(
            rows,
            probability_key=probability_key,
            grid_points=grid_points,
        ),
        "deployed_tau": deployed_tau,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Selective-prediction reporting metrics (AUROC, ECE, "
            "risk-coverage, AURC) for the safe-alignment veto, computed "
            "from a recorded validation run. Read-only, no GPU, no new "
            "evaluation, no threshold selected."
        )
    )
    parser.add_argument("--sample-decisions", required=True)
    parser.add_argument(
        "--probability-column",
        default="safe_task_probability",
    )
    parser.add_argument(
        "--gate-column",
        default="safe_veto_task_gate",
    )
    parser.add_argument("--classifier-summary", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--grid-points", type=int, default=101)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.sample_decisions).expanduser().open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [
        {
            "id": str(row["id"]),
            "scenario": str(row["scenario"]),
            "advantage": float(row["aligned_loss_advantage"]),
            "q_gate": str(row["q_gate"]).lower() in {"1", "true", "yes"},
            args.probability_column: float(
                row[args.probability_column]
            ),
        }
        for row in raw_rows
    ]
    deployed_tau = None
    if args.classifier_summary:
        summary = json.loads(
            Path(args.classifier_summary)
            .expanduser()
            .read_text(encoding="utf-8")
        )
        deployed_tau = float(summary["probability_threshold"])

    report = selective_metrics(
        rows,
        probability_key=args.probability_column,
        gate_key=args.gate_column,
        bins=int(args.bins),
        grid_points=int(args.grid_points),
        deployed_tau=deployed_tau,
    )
    report["source"] = str(args.sample_decisions)
    save_json(args.output, report)
    printable = json.loads(json.dumps(report))
    printable["risk_coverage"].pop("curve", None)
    for entry in printable["per_scenario"].values():
        entry.pop("bins", None)
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
