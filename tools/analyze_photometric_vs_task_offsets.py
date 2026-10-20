from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


def _magnitude(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5


def decompose(
    records: list[dict[str, Any]],
    *,
    clean_scenario: str = "clean:s0:common",
) -> dict[str, Any]:
    """Test whether an estimator reports injection PLUS a fixed residual.

    If a method's raw error against the injected offset is roughly
    constant across severities while its DELTA (shift estimate minus its
    own clean estimate) tracks the injection accurately, then the method
    is faithfully reporting ``injection + intrinsic photometric
    displacement`` and the constant is that intrinsic displacement.

    An estimator whose raw error is already small needs no such
    correction: it is reporting the injection alone, i.e. it has learned
    to ignore a photometric displacement that carries no task meaning.
    This is the whole task-coordinates-versus-photometric-coordinates
    distinction, measured directly and without any label.
    """

    by_method: dict[str, dict[str, dict[str, tuple]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in records:
        method = str(row["method"])
        scenario = str(row["scenario"])
        by_method[method][scenario][str(row["id"])] = (
            float(row["est_dx"]),
            float(row["est_dy"]),
            float(row["gt_dx"]),
            float(row["gt_dy"]),
        )

    summary: dict[str, Any] = {}
    for method, scenarios in by_method.items():
        clean = scenarios.get(clean_scenario, {})
        if not clean:
            raise ValueError(f"{method} has no {clean_scenario} rows")
        clean_magnitudes = [
            _magnitude(value[0], value[1]) for value in clean.values()
        ]
        entry: dict[str, Any] = {
            "clean_estimate_mean_magnitude": sum(clean_magnitudes)
            / len(clean_magnitudes),
            "per_shift": {},
        }
        for scenario, rows in sorted(scenarios.items()):
            if scenario == clean_scenario:
                continue
            raw_errors: list[float] = []
            delta_errors: list[float] = []
            for key, value in rows.items():
                if key not in clean:
                    continue
                est_dx, est_dy, gt_dx, gt_dy = value
                clean_dx, clean_dy, _, _ = clean[key]
                raw_errors.append(
                    _magnitude(est_dx - gt_dx, est_dy - gt_dy)
                )
                delta_errors.append(
                    _magnitude(
                        (est_dx - clean_dx) - gt_dx,
                        (est_dy - clean_dy) - gt_dy,
                    )
                )
            if not raw_errors:
                continue
            raw_mean = sum(raw_errors) / len(raw_errors)
            delta_mean = sum(delta_errors) / len(delta_errors)
            entry["per_shift"][scenario] = {
                "samples": len(raw_errors),
                "raw_mean_epe": raw_mean,
                "delta_mean_epe": delta_mean,
                "delta_improves_by": raw_mean - delta_mean,
                "raw_pck1": sum(1 for e in raw_errors if e <= 1.0)
                / len(raw_errors),
                "delta_pck1": sum(1 for e in delta_errors if e <= 1.0)
                / len(delta_errors),
            }
        shifts = entry["per_shift"].values()
        if shifts:
            raw_values = [item["raw_mean_epe"] for item in shifts]
            entry["raw_epe_spread"] = max(raw_values) - min(raw_values)
            entry["mean_delta_improvement"] = sum(
                item["delta_improves_by"] for item in shifts
            ) / len(list(shifts))
        summary[method] = entry

    # Cross-method agreement on the clean estimate. Two algorithms with
    # nothing in common -- FFT peak correlation and gradient-based ECC --
    # agreeing per image is what separates "a real photometric
    # displacement in the data" from "an artefact of one algorithm on
    # changed content".
    clean_by_method = {
        method: scenarios.get(clean_scenario, {})
        for method, scenarios in by_method.items()
    }
    agreement: dict[str, Any] = {}
    names = sorted(clean_by_method)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            shared = [
                key
                for key in clean_by_method[first]
                if key in clean_by_method[second]
            ]
            if not shared:
                continue
            distances = [
                _magnitude(
                    clean_by_method[first][key][0]
                    - clean_by_method[second][key][0],
                    clean_by_method[first][key][1]
                    - clean_by_method[second][key][1],
                )
                for key in shared
            ]
            agreement[f"{first}|{second}"] = {
                "samples": len(shared),
                "mean_distance": sum(distances) / len(distances),
                "within_1px": sum(1 for d in distances if d <= 1.0)
                / len(distances),
                "within_2px": sum(1 for d in distances if d <= 2.0)
                / len(distances),
            }
    return {"per_method": summary, "clean_estimate_agreement": agreement}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose each estimator's offsets into injection versus a "
            "fixed photometric residual, from offset_records.csv only. "
            "Read-only: no model, no GPU, no evaluation consumed."
        )
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.records).expanduser().open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        records = list(csv.DictReader(handle))
    result = decompose(records)
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("--- raw vs delta mean EPE (lower is better) ---")
    for method, entry in result["per_method"].items():
        cells = " ".join(
            f"{key.split(':')[1]}:{item['raw_mean_epe']:.2f}->"
            f"{item['delta_mean_epe']:.2f}"
            for key, item in entry["per_shift"].items()
        )
        print(
            f"{method:8s} clean|est|={entry['clean_estimate_mean_magnitude']:.2f}"
            f"  {cells}"
        )
    print("--- clean-estimate agreement between methods ---")
    for pair, entry in result["clean_estimate_agreement"].items():
        print(
            f"{pair:16s} mean {entry['mean_distance']:.2f} px  "
            f"within1 {entry['within_1px']:.2f}  "
            f"within2 {entry['within_2px']:.2f}"
        )


if __name__ == "__main__":
    main()
