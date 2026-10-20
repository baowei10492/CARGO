from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from rs_cd.runtime import save_json
from tools.diagnose_transfer_offset_consistency import (
    offset_consistency_summary,
)


def _read_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        for key in ("est_dx", "est_dy", "gt_dx", "gt_dy"):
            record[key] = float(row[key])
        record["q_gate"] = str(row["q_gate"]).lower() in {
            "1",
            "true",
            "yes",
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute an offset consistency summary from an existing "
            "offset_records.csv, with no GPU and no re-evaluation. Use "
            "it to backfill statistics added after a run (for example "
            "gt_boundary_fraction) onto measurements already taken."
        )
    )
    parser.add_argument(
        "--records",
        required=True,
        help="Path to offset_records.csv from a previous run.",
    )
    parser.add_argument(
        "--protocol",
        required=True,
        help="The protocol yaml that run used (for the boundary limit).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", default=None)
    args = parser.parse_args()

    protocol = yaml.safe_load(
        Path(args.protocol).expanduser().read_text(encoding="utf-8")
    )
    records = _read_records(Path(args.records).expanduser())
    scenarios = sorted({str(row["scenario"]) for row in records})
    clean = [key for key in scenarios if key.startswith("clean:")]
    shifts = [key for key in scenarios if key.startswith("shift:")]
    if len(clean) != 1:
        raise ValueError(f"Expected exactly one clean scenario: {clean}")

    summary = offset_consistency_summary(
        records,
        clean_scenario=clean[0],
        shift_scenarios=shifts,
        boundary_threshold=float(protocol["maximum_offset_pixels"]) - 0.1,
    )
    q_rates = {}
    for scenario in scenarios:
        rows = [row for row in records if row["scenario"] == scenario]
        q_rates[scenario] = sum(
            bool(row["q_gate"]) for row in rows
        ) / max(1, len(rows))
    summary["q_trigger_rate"] = q_rates
    summary["recomputed_from"] = str(args.records)
    if args.image_size is not None:
        summary["evaluation_image_size"] = args.image_size
    save_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
