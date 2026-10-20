"""Summarise every P27 detector row and re-judge it with a tolerant instrument check.

For each row (the DINOv3 row and the plug-in rows) the latest F1-step,
validation and test decisions are read, and every dataset is judged again with
the same rules except that the clean oracle may differ from the clean raw path
by up to ``--oracle-tolerance`` (CD-Lamba draws per-sample random scan windows,
so two identical crops in one batch differ slightly; main record 192.13).
Writes a CSV with every F1 value and prints a compact table. Reads JSON only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from rs_cd.affine_f1 import f1_decision

ROWS = ("dinov3", "bit", "cdlamba", "cdxlstm", "bit_whu", "cdlamba_whu", "cdxlstm_whu", "dinov3aug")
STAGES = {
    "f1": ("f1_decision.json", "estimated_gated"),
    "selector_eval": ("selector_decision.json", "selector"),
    "test": ("test_decision.json", "selector"),
}


def pointer(row: str, stage: str, root: Path) -> Path:
    if row == "dinov3":
        relative = {"f1": "f1", "selector_eval": "selector/eval", "test": "test"}[stage]
        return root / relative / "LATEST"
    return root / "plugins" / row / stage / "LATEST"


def rejudge(entry: dict[str, Any], primary: str, rules: dict[str, Any], tolerance: float) -> dict[str, Any]:
    levels = [int(level) for level in entry["f1"] if level != "0"]
    return f1_decision(
        entry["f1"], levels=levels, primary=primary,
        binding_levels=[int(level) for level in rules["decision"]["binding_levels"]],
        min_gain=float(rules["decision"]["min_gain"]), max_clean_drop=float(rules["decision"]["max_clean_drop"]),
        min_clean_raw_f1=float(rules["instrument"]["min_clean_raw_f1"]), oracle_tolerance=tolerance,
    )


def collect(root: Path, rules: dict[str, Any], tolerance: float, project_root: Path = Path(".")) -> list[dict[str, Any]]:
    rows = []
    for row in ROWS:
        for stage, (filename, primary) in STAGES.items():
            latest = project_root / pointer(row, stage, root)
            if not latest.is_file():
                continue
            decision_path = project_root / latest.read_text(encoding="utf-8").strip() / "run" / filename
            if not decision_path.is_file():
                continue
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            for dataset, entry in decision["datasets"].items():
                record: dict[str, Any] = {"row": row, "stage": stage, "dataset": dataset,
                                          "original_verdict": entry["verdict"], "decision": str(decision_path)}
                if "f1" not in entry:
                    rows.append(record)
                    continue
                judged = rejudge(entry, primary, rules, tolerance)
                record.update(verdict=judged["verdict"], primary=primary, clean_drop=judged["clean_drop"],
                              clean_oracle_difference=judged["instrument"]["clean_oracle_difference"])
                for level, values in judged["levels"].items():
                    record[f"gain_{level}"] = values["gain"]
                    record[f"recovery_{level}"] = values["recovery"]
                for level, paths in entry["f1"].items():
                    for path, value in paths.items():
                        record[f"f1_{level}_{path}"] = value
                rows.append(record)
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'row':<12} {'stage':<14} {'dataset':<9} {'verdict (original)':<38} "
             f"{'clean drop':>10} {'gain L1':>8} {'gain L2':>8} {'gain L3':>8} {'rec L3':>7}"]
    for r in rows:
        if "verdict" not in r:
            lines.append(f"{r['row']:<12} {r['stage']:<14} {r['dataset']:<9} {r['original_verdict']}")
            continue
        verdict = r["verdict"] if r["verdict"] == r["original_verdict"] else f"{r['verdict']} ({r['original_verdict']})"
        recovery = "-" if r.get("recovery_3") is None else f"{r['recovery_3']:.2f}"
        lines.append(f"{r['row']:<12} {r['stage']:<14} {r['dataset'][:9]:<9} {verdict:<38} {r['clean_drop']:+10.4f} "
                     f"{r['gain_1']:+8.4f} {r['gain_2']:+8.4f} {r['gain_3']:+8.4f} {recovery:>7}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/p27_localizer")
    parser.add_argument("--rules", default="configs/p27_f1_gate.yaml", help="decision and instrument rules")
    parser.add_argument("--oracle-tolerance", type=float, default=0.001)
    parser.add_argument("--output", default=None, help="CSV with every F1 value")
    args = parser.parse_args()
    rules = yaml.safe_load(Path(args.rules).read_text(encoding="utf-8"))
    rows = collect(Path(args.root), rules, args.oracle_tolerance)
    print(render(rows))
    if args.output and rows:
        fields = sorted({key for row in rows for key in row}, key=lambda k: (not k[0].isalpha(), k))
        leading = ["row", "stage", "dataset", "verdict", "original_verdict", "primary", "clean_drop"]
        fields = leading + [f for f in fields if f not in leading]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
