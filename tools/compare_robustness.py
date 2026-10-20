from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two robustness result files scenario by scenario."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = _load(args.reference)
    candidate = _load(args.candidate)
    if reference["protocol_sha256"] != candidate["protocol_sha256"]:
        raise ValueError("Cannot compare results from different protocols")
    reference_rows = {
        row["scenario"]: row for row in reference["results"]
    }
    candidate_rows = {
        row["scenario"]: row for row in candidate["results"]
    }
    if reference_rows.keys() != candidate_rows.keys():
        raise ValueError("Scenario sets differ between robustness reports")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, reference_row in reference_rows.items():
        candidate_row = candidate_rows[scenario]
        rows.append(
            {
                "scenario": scenario,
                "corruption": reference_row["corruption"],
                "severity": reference_row["severity"],
                "mode": reference_row["mode"],
                f"{args.reference_name}_f1": reference_row["f1"],
                f"{args.candidate_name}_f1": candidate_row["f1"],
                "candidate_minus_reference_f1": (
                    float(candidate_row["f1"])
                    - float(reference_row["f1"])
                ),
                f"{args.reference_name}_drop": reference_row["f1_drop"],
                f"{args.candidate_name}_drop": candidate_row["f1_drop"],
                "drop_reduction": (
                    float(reference_row["f1_drop"])
                    - float(candidate_row["f1_drop"])
                ),
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    corrupted = [row for row in rows if row["corruption"] != "clean"]
    clean = [row for row in rows if row["corruption"] == "clean"]
    if len(clean) != 1:
        raise ValueError(
            f"Expected exactly one clean scenario, found {len(clean)}"
        )
    clean_delta = float(clean[0]["candidate_minus_reference_f1"])
    mean_corrupted_delta = sum(
        float(row["candidate_minus_reference_f1"]) for row in corrupted
    ) / max(1, len(corrupted))
    summary = {
        "protocol_sha256": reference["protocol_sha256"],
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "scenario_count": len(rows),
        "candidate_wins": sum(
            float(row["candidate_minus_reference_f1"]) > 0
            for row in corrupted
        ),
        "clean_candidate_minus_reference_f1": clean_delta,
        "mean_corrupted_candidate_minus_reference_f1": (
            mean_corrupted_delta
        ),
        "mean_corrupted_excess_gain_over_clean": (
            mean_corrupted_delta - clean_delta
        ),
        "metric_note": (
            "Excess gain subtracts the clean delta, so detector-quality "
            "changes are not mislabeled as corruption robustness."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved scenario comparison to {output.resolve()}")


if __name__ == "__main__":
    main()
