"""P27 selector ablations, offline (CPU only).

For one detector row: retrain the selector on the features of a finished
selector training run with feature subsets (same P7-3 recipe, same threshold
rule), then score every variant on the per-crop records of a finished
validation or test run; also sweep the threshold of the main selector. The
frozen selectors and their results are not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from rs_cd.p27_selector import (
    FEATURE_NAMES,
    REGISTRATION_FEATURES,
    TASK_FEATURES,
    f1_table,
    feature_matrix,
    selection_rates,
    selector_probability,
)
from tools.run_p27_selector import read_records, train_dataset
from tools.run_p27_step1 import sha256

VARIANTS = {
    "full": FEATURE_NAMES,
    "registration_only": REGISTRATION_FEATURES,
    "task_only": TASK_FEATURES,
    "without_forward_backward": tuple(name for name in FEATURE_NAMES if name != "forward_backward_px"),
}
RULES = ("raw", "oracle", "estimated", "gated", "selector", "selector_veto")


def _tau(bundle: dict[str, Any]) -> float:
    return math.inf if bundle["tau"] is None else float(bundle["tau"])


def score(records: Sequence[dict[str, Any]], bundle: dict[str, Any]) -> tuple[list[float], dict[str, Any], dict[str, Any]]:
    probability = selector_probability(bundle, feature_matrix(records, bundle["feature_names"])).tolist()
    tau = _tau(bundle)
    return probability, f1_table(records, probability, tau, RULES), selection_rates(records, probability, tau)


def summarize(table: dict[str, dict[str, float]], rule: str) -> dict[str, float]:
    out = {"clean_drop": table["0"]["raw"] - table["0"][rule]}
    for level in (key for key in table if key != "0"):
        out[f"gain_{level}"] = table[level][rule] - table[level]["raw"]
    return out


def sweep(records: Sequence[dict[str, Any]], probability: Sequence[float], step: float) -> list[dict[str, Any]]:
    rows = []
    for index in range(int(round(1.0 / step)) + 2):
        tau = math.inf if index > round(1.0 / step) else round(index * step, 10)
        table = f1_table(records, probability, tau, ("raw", "selector"))
        rates = selection_rates(records, probability, tau)
        rows.append({"tau": None if math.isinf(tau) else tau, **summarize(table, "selector"),
                     **{f"selected_{level}": rates[level]["selector"] for level in rates}})
    return rows


def run(config: dict[str, Any], train_root: Path, records_path: Path, *, sweep_step: float = 0.02,
        variants: dict[str, Sequence[str]] = VARIANTS) -> dict[str, Any]:
    records = read_records(records_path)
    result: dict[str, Any] = {"records": str(records_path), "train_root": str(train_root), "datasets": {}}
    for name in sorted({record["dataset"] for record in records}):
        members = [record for record in records if record["dataset"] == name]
        main = torch.load(train_root / "bundles" / f"selector_{name}.pth", map_location="cpu", weights_only=False)
        training = read_records(train_root / "features" / f"features_{name}.csv")
        main_probability, main_table, main_rates = score(members, main)
        entry: dict[str, Any] = {
            "reference": {rule: summarize(main_table, rule) for rule in ("estimated", "gated")},
            "main": {"tau": main["tau"], **summarize(main_table, "selector"),
                     "veto": summarize(main_table, "selector_veto"), "selection": main_rates, "f1": main_table},
            "variants": {},
            "sweep": sweep(members, main_probability, sweep_step),
        }
        for variant, names in variants.items():
            bundle, _ = train_dataset(name, training, config, tuple(names))
            probability, table, rates = score(members, bundle)
            entry["variants"][variant] = {"features": len(names), "tau": bundle["tau"],
                                          **summarize(table, "selector"), "selection": rates, "f1": table}
            if variant == "full":
                # The retrained full selector must be the frozen one (deterministic CPU training).
                entry["full_reproduces_main"] = {
                    "tau_equal": bundle["tau"] == main["tau"],
                    "max_probability_difference": max(abs(a - b) for a, b in zip(probability, main_probability)),
                }
        result["datasets"][name] = entry
    return result


def render(result: dict[str, Any]) -> str:
    lines = [f"records: {result['records']}"]
    for name, entry in result["datasets"].items():
        levels = [key for key in entry["main"] if key.startswith("gain_")]
        lines.append(f"\n[{name}] full selector reproduces the frozen one: {entry.get('full_reproduces_main')}")
        lines.append(f"  {'variant':<26} {'tau':>6} {'clean drop':>10} " + " ".join(f"{level:>8}" for level in levels))

        def row(label, tau, values):
            tau_text = "never" if tau is None else f"{tau:.2f}"
            gains = " ".join(f"{values[level]:+8.4f}" for level in levels)
            lines.append(f"  {label:<26} {tau_text:>6} {values['clean_drop']:+10.4f} {gains}")

        row("always correct", 0.0, entry["reference"]["estimated"])
        row("localizer gate", 0.0, entry["reference"]["gated"])
        row("selector (frozen)", entry["main"]["tau"], entry["main"])
        row("selector veto (frozen)", entry["main"]["tau"], entry["main"]["veto"])
        for variant, values in entry["variants"].items():
            row(f"retrained {variant} ({values['features']})", values["tau"], values)
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the selector config of this detector row")
    parser.add_argument("--train-root", required=True, help="selector training report root (features/, bundles/)")
    parser.add_argument("--records", required=True, help="validation_records.csv or test_records.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sweep-step", type=float, default=0.02)
    args = parser.parse_args()
    config_path, train_root = Path(args.config), Path(args.train_root)
    summary = json.loads((train_root / "bundles" / "train_summary.json").read_text(encoding="utf-8"))
    if summary["config_sha256"] != sha256(config_path):
        raise SystemExit("the training run used a different selector config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = run(config, train_root, Path(args.records), sweep_step=args.sweep_step)
    (output_dir / "selector_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(output_dir / "tau_sweep.csv", [{"dataset": name, **row} for name, entry in result["datasets"].items()
                                             for row in entry["sweep"]])
    print(render(result))


if __name__ == "__main__":
    main()
