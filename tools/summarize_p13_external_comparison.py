from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCENARIOS = (
    "clean:s0:common",
    "shift:s1:t2",
    "shift:s2:t2",
    "shift:s3:t2",
)


def _valid_pixels(row: dict[str, Any]) -> int:
    return sum(int(row[key]) for key in ("tp", "fp", "fn", "tn"))


def comparison_summary(
    project_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    *,
    project_method: str,
    external_method: str,
) -> dict[str, Any]:
    project = {
        (str(row["scenario"]), str(row["method"])): row
        for row in project_rows
    }
    external = {
        (str(row["scenario"]), str(row["method"])): row
        for row in external_rows
    }
    rows = []
    for scenario in SCENARIOS:
        project_row = project[(scenario, project_method)]
        external_row = external[(scenario, external_method)]
        project_support = _valid_pixels(project_row)
        external_support = _valid_pixels(external_row)
        project_f1 = float(project_row["f1"])
        external_f1 = float(external_row["change_f1"])
        rows.append(
            {
                "scenario": scenario,
                "project_method": project_method,
                "project_f1": project_f1,
                "external_method": external_method,
                "external_f1": external_f1,
                "project_minus_external_f1": (
                    project_f1 - external_f1
                ),
                "project_valid_pixels": project_support,
                "external_valid_pixels": external_support,
                "same_valid_pixel_count": (
                    project_support == external_support
                ),
            }
        )
    support_passed = all(
        bool(row["same_valid_pixel_count"]) for row in rows
    )
    shifts = rows[1:]
    return {
        "status": "completed" if support_passed else "failed",
        "reporting_only_no_selection": True,
        "same_valid_pixel_count_all_scenarios": support_passed,
        "project_method": project_method,
        "external_method": external_method,
        "scenario_comparison": rows,
        "mean_project_minus_external_shift_f1": sum(
            float(row["project_minus_external_f1"])
            for row in shifts
        )
        / len(shifts),
        "test_result_used_for_model_selection": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-results", required=True)
    parser.add_argument("--external-results", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--project-method",
        default="safe_veto_task",
    )
    parser.add_argument(
        "--external-method",
        default="official_unmodified",
    )
    args = parser.parse_args()

    project = json.loads(
        Path(args.project_results).read_text(encoding="utf-8")
    )
    external = json.loads(
        Path(args.external_results).read_text(encoding="utf-8")
    )
    summary = comparison_summary(
        project["scenario_results"],
        external["scenario_results"],
        project_method=args.project_method,
        external_method=args.external_method,
    )
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_csv, summary["scenario_comparison"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

