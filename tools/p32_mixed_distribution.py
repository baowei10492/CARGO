"""Mixed-distribution evaluation (docs/P32_PLUGIN_EXTENSION.md, section 3). CPU only.

    python -m tools.p32_mixed_distribution --config configs/p32_mixed_whu.yaml

Every table so far reports one misalignment level at a time, which tells a reader how
each method behaves once the state of the input is known. Deployment does not know it.
This evaluation therefore builds one test set whose crops are split into equal thirds --
aligned, small misalignment, large misalignment -- by a deterministic hash of the crop
identifier, and scores each method once over the whole set. It is where a fixed policy
has to pay for its assumption: "never correct" pays on the misaligned thirds, "always
correct" pays on the aligned third, and the selector is the only row that can avoid both.

Nothing is run on images: each crop's confusion counts under its assigned state are
already in the frozen per-crop records, and the records are checked against the frozen
test tables (tools.p31_statistics.check_cell) before a single number is computed.
The two P32 detectors join the table as soon as their test records exist; the state
assignment does not change when they do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from rs_cd.robustness import _stable_seed
from tools.p31_paper_tables import latex, markdown, pct, signed, write_csv
from tools.p31_selector_ablation import DATASETS, sources
from tools.p31_statistics import Cell, augmentation_counts, check_augmentation, check_cell, f1

LABELS = {"dinov3": "DINOv3 detector", "bit": "BIT", "cdlamba": "CD-Lamba", "cdxlstm": "CDxLSTM",
          "snunet": "SNUNet", "bifa": "BiFA"}
REPORT_ROOTS = (Path("reports/p31"), Path("reports/p32"))
RULES = (("raw", "Without correction"), ("always", "Always corrected"),
         ("selector", "Ours (localizer + selector)"))


# --------------------------------------------------------------------------------- state assignment
def state_of(crop_id: str, seed: int, states: int) -> int:
    """The third a crop falls in. Deterministic in the crop identifier, as the P27 split is."""
    return _stable_seed(int(seed), str(crop_id), "p32-mixed", 0, "state") % states


# --------------------------------------------------------------------------------- loading
def row_sources(row: str, dataset: str) -> tuple[Path, Path, Path]:
    """(selector bundle directory, test records, test table) of a border-free row."""
    if row == "dinov3":
        train_root, records = sources("dinov3")
        test = Path(records[DATASETS.index(dataset)]).parent
        return train_root / "bundles", records[DATASETS.index(dataset)], test / "p30_test.json"
    for root in REPORT_ROOTS:
        table = root / row / "test" / "p31_test.json"
        if table.is_file():
            return root / row / "bundles", root / row / "test" / f"records_{dataset}.csv", table
    raise SystemExit(f"{row}: no finished test run under {[str(p) for p in REPORT_ROOTS]}")


def anchor_of(row: str, dataset: str, table: Path) -> dict[str, dict[str, float]]:
    data = json.loads(table.read_text(encoding="utf-8"))["datasets"][dataset]
    return data["proposed"]["f1"] if row == "dinov3" else data["f1"]


def load_cell(row: str, dataset: str) -> Cell:
    bundles, records, table = row_sources(row, dataset)
    bundle = torch.load(bundles / f"selector_{dataset}.pth", map_location="cpu", weights_only=False)
    cell = Cell(row, LABELS.get(row, row), dataset, records, bundle)
    check_cell(cell, anchor_of(row, dataset, table))
    return cell


# --------------------------------------------------------------------------------- mixing
def assignment(cell: Cell, seed: int, levels: Sequence[int]) -> np.ndarray:
    """Per record: True when its level is the one its crop was assigned."""
    chosen = {crop: levels[state_of(crop, seed, len(levels))] for crop in set(cell.crop.tolist())}
    return np.array([chosen[crop] == level for crop, level in zip(cell.crop, cell.level)])


def counts_of(cell: Cell, rule: str) -> np.ndarray:
    if rule == "raw":
        return cell.counts["raw"]
    if rule == "always":
        return cell.counts["estimated"]
    if rule == "oracle":
        return cell.counts["oracle"]
    if rule == "selector":
        return cell.selected()
    raise ValueError(f"unknown rule {rule}")


def scores(cell: Cell, keep: np.ndarray, rule_counts: dict[str, np.ndarray],
           states: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Pooled F1 over the mixed set, and the same rule restricted to each state."""
    out: dict[str, Any] = {"mixed": {}, "by_state": {}}
    for rule, counts in rule_counts.items():
        out["mixed"][rule] = float(f1(counts[keep].sum(axis=0)))
    levels = [int(state["level"]) for state in states]
    chosen = np.array([levels[state_of(crop, seed, len(levels))] for crop in cell.crop])
    for state in states:
        level = int(state["level"])
        inside = keep & (chosen == level)
        out["by_state"][state["name"]] = {
            "level": level,
            "crops": int(inside.sum()),
            **{rule: float(f1(counts[inside].sum(axis=0))) for rule, counts in rule_counts.items()},
        }
    return out


def run(config: dict[str, Any], rows: Sequence[str]) -> dict[str, Any]:
    dataset, seed, states = config["dataset"], int(config["seed"]), config["states"]
    levels = [int(state["level"]) for state in states]
    result: dict[str, Any] = {"dataset": dataset, "seed": seed,
                              "states": [{**state} for state in states], "rows": {}}
    for row in rows:
        cell = load_cell(row, dataset)
        keep = assignment(cell, seed, levels)
        rule_counts = {rule: counts_of(cell, rule) for rule in ("raw", "always", "selector", "oracle")}
        if row == config.get("augmentation_row"):
            augmented = augmentation_counts(cell)
            check_augmentation(cell, augmented)
            rule_counts["augmentation"] = augmented
        entry = scores(cell, keep, rule_counts, states, seed)
        entry["label"] = cell.label
        entry["crops"] = int(keep.sum())
        entry["tau"] = None if not np.isfinite(cell.tau) else float(cell.tau)
        entry["corrected_fraction"] = {
            state["name"]: float(
                (cell.probability[keep & (cell.level == int(state["level"]))] >= cell.tau).mean()
            )
            for state in states
        }
        result["rows"][row] = entry
        print(render_row(row, entry), flush=True)
    counted = {row: entry["crops"] for row, entry in result["rows"].items()}
    if len(set(counted.values())) != 1:
        raise SystemExit(f"the rows do not share one mixed set: {counted}")
    result["crops"] = next(iter(counted.values()))
    result["ours_is_best_fixed_policy"] = all(
        entry["mixed"]["selector"] >= max(entry["mixed"]["raw"], entry["mixed"]["always"])
        for entry in result["rows"].values())
    return result


def render_row(row: str, entry: dict[str, Any]) -> str:
    states = ", ".join(f"{name}: {values['selector'] * 100:.1f}"
                       for name, values in entry["by_state"].items())
    mixed = entry["mixed"]
    return (f"[{row}] mixed F1 % raw {mixed['raw'] * 100:.1f} / always {mixed['always'] * 100:.1f} / "
            f"ours {mixed['selector'] * 100:.1f} (oracle {mixed['oracle'] * 100:.1f}); ours by state: {states}")


# --------------------------------------------------------------------------------- tables
def table_rows(result: dict[str, Any]) -> tuple[list[str], list[list[str]], list[list[Any]]]:
    header = ["Detector", "Setting", "Mixed set", *[state["name"].capitalize() for state in result["states"]],
              "Change"]
    text, numbers = [], []
    for row, entry in result["rows"].items():
        rules = list(RULES)
        if "augmentation" in entry["mixed"]:
            rules.insert(1, ("augmentation", "Augmentation"))
        for rule, label in rules:
            values = [entry["mixed"][rule], *[entry["by_state"][state["name"]][rule]
                                              for state in result["states"]]]
            change = entry["mixed"][rule] - entry["mixed"]["raw"]
            text.append([entry["label"] if rule == rules[0][0] else "", label,
                         *[pct(value) for value in values],
                         "-" if rule == "raw" else signed(change)])
            numbers.append([entry["label"], label, *values, None if rule == "raw" else change])
    return header, text, numbers


def caption(result: dict[str, Any]) -> str:
    states = ", ".join(f"{state['name']} (level {state['level']})" for state in result["states"])
    return (f"F1 (%) on the mixed {result['dataset']} test set: each of the {result['crops']} crops is "
            f"assigned one of three states -- {states} -- in equal thirds by a deterministic hash, and no "
            "method is told which. The three state columns score the same rule on that third alone; "
            "``Change'' is the mixed-set F1 relative to the same detector without correction, in points.")


def write(result: dict[str, Any], output_dir: Path, stem: str = "table_p32_mixed") -> None:
    header, text, numbers = table_rows(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = [index for index, row in enumerate(text) if index and row[0]]
    align = "ll" + "r" * (len(header) - 2)
    (output_dir / f"{stem}.md").write_text(markdown(caption(result), header, text), encoding="utf-8")
    (output_dir / f"{stem}.tex").write_text(
        latex(caption(result), "tab:mixed", header, text, align=align, groups=groups), encoding="utf-8")
    write_csv(output_dir / f"{stem}.csv", header, numbers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p32_mixed_whu.yaml")
    parser.add_argument("--rows", nargs="+", default=None, help="overrides the rows of the config")
    parser.add_argument("--output-root", default="reports/p32/mixed")
    parser.add_argument("--paper-dir", default="paper/data/p32")
    args = parser.parse_args()

    path = Path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit(f"{path} is not marked preregistered")
    result = run(config, args.rows or config["rows"])
    result["config"] = {"path": path.as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mixed.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write(result, Path(args.paper_dir))
    print(json.dumps({"output": str(output_dir), "crops": result["crops"],
                      "ours_is_best_fixed_policy": result["ours_is_best_fixed_policy"]}, indent=2))


if __name__ == "__main__":
    main()
