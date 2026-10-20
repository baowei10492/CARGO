"""P33 statistics from the frozen border-free test records (CPU only; docs/P33_REVIEWER_EXPERIMENTS.md).

    versus_always   per cell and level: F1 of CARGO (registration module + selector) minus always correcting and minus
                    never correcting, point value and 95% tile-bootstrap interval; and the mean over the 18 cells, the
                    tiles of a dataset resampled once per repetition and shared by its six detectors
    mixed           the WHU-CD mixed test (configs/p32_mixed_whu.yaml): overall F1 of CARGO minus never and minus always
                    correcting, per detector, tile bootstrap

The records must reproduce the frozen test tables (tools.p31_statistics.check_cell through load_cell) before any
number is written. Output: reports/p33/statistics/statistics.json and statistics.txt.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from rs_cd.robustness import _stable_seed
from tools.p31_selector_ablation import DATASETS, sources
from tools.p31_statistics import LEVELS, Cell, check_cell, draws, f1, interval, resampled

ROWS = ("dinov3", "bit", "cdlamba", "cdxlstm", "snunet", "bifa")
MIXED_CONFIG = Path("configs/p32_mixed_whu.yaml")
REPORT_ROOTS = (Path("reports/p31"), Path("reports/p32"))


# The next three follow tools.p32_mixed_distribution (state_of, assignment, load_cell) without its table imports.
def state_of(crop_id: str, seed: int, states: int) -> int:
    return _stable_seed(int(seed), str(crop_id), "p32-mixed", 0, "state") % states


def assignment(cell: Cell, seed: int, levels: Sequence[int]) -> np.ndarray:
    """Per record: True when its level is the one its crop was assigned."""
    chosen = {crop: levels[state_of(crop, seed, len(levels))] for crop in set(cell.crop.tolist())}
    return np.array([chosen[crop] == level for crop, level in zip(cell.crop, cell.level)])


def load_cell(row: str, dataset: str) -> Cell:
    """The frozen test records of a row on a dataset, checked against its test table."""
    if row == "dinov3":
        train_root, records = sources("dinov3")
        path = Path(records[DATASETS.index(dataset)])
        bundles = train_root / "bundles"
        anchor = json.loads((path.parent / "p30_test.json").read_text(encoding="utf-8"))["datasets"][dataset]["proposed"]["f1"]
    else:
        roots = [root / row for root in REPORT_ROOTS if (root / row / "test" / "p31_test.json").is_file()]
        if not roots:
            raise SystemExit(f"{row}: no finished test run under {[str(r) for r in REPORT_ROOTS]}")
        path, bundles = roots[0] / "test" / f"records_{dataset}.csv", roots[0] / "bundles"
        anchor = json.loads((roots[0] / "test" / "p31_test.json").read_text(encoding="utf-8"))["datasets"][dataset]["f1"]
    bundle = torch.load(bundles / f"selector_{dataset}.pth", map_location="cpu", weights_only=False)
    cell = Cell(row, row, dataset, path, bundle)
    check_cell(cell, anchor)
    return cell


def level_samples(cell: Cell, first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> dict[int, tuple[float, np.ndarray]]:
    a, b = cell.tile_sums(first), cell.tile_sums(second)
    out = {}
    for i, level in enumerate(LEVELS):
        point = float(f1(a[i].sum(axis=0)) - f1(b[i].sum(axis=0)))
        out[level] = (point, f1(resampled(weights, a[i])) - f1(resampled(weights, b[i])))
    return out


def shared_weights(cells: Sequence[Cell], repeats: int, seed: int) -> dict[str, np.ndarray]:
    """One tile draw per dataset; every detector of that dataset sees the same tiles (same crops, same tile index)."""
    crops: dict[str, set] = {}
    weights = {}
    for cell in cells:
        if cell.dataset not in crops:
            crops[cell.dataset] = set(cell.crop.tolist())
            weights[cell.dataset] = draws(cell.tiles, repeats, seed + zlib.crc32(cell.dataset.encode()) % 100000)
        elif set(cell.crop.tolist()) != crops[cell.dataset] or weights[cell.dataset].shape[1] != cell.tiles:
            raise SystemExit(f"{cell.name}: its crops differ from the other rows of {cell.dataset}")
    return weights


def run_versus(cells: Sequence[Cell], repeats: int, seed: int) -> dict[str, Any]:
    weights = shared_weights(cells, repeats, seed)
    result: dict[str, Any] = {"repeats": repeats, "seed": seed, "unit": "tile", "cells": {}, "mean_18": {}}
    pooled: dict[str, dict[int, list]] = {key: {level: [] for level in LEVELS} for key in ("always", "never")}
    for cell in cells:
        w = weights[cell.dataset]
        entry = {}
        for key, other in (("always", cell.counts["estimated"]), ("never", cell.counts["raw"])):
            samples = level_samples(cell, cell.selected(), other, w)
            entry[f"selector_minus_{key}"] = {str(level): interval(s, p) for level, (p, s) in samples.items()}
            for level, value in samples.items():
                pooled[key][level].append(value)
        result["cells"][cell.name] = entry
    for key, levels in pooled.items():
        result["mean_18"][f"selector_minus_{key}"] = {
            str(level): interval(np.mean([s for _, s in values], axis=0), float(np.mean([p for p, _ in values])))
            for level, values in levels.items()}
    always = [entry["selector_minus_always"] for entry in result["cells"].values()]
    result["counts"] = {
        f"level_{level}_selector_vs_always": {
            "significantly_higher": sum(e[str(level)]["low"] > 0 for e in always),
            "significantly_lower": sum(e[str(level)]["high"] < 0 for e in always),
            "cells": len(always)} for level in LEVELS}
    return result


def tile_totals(cell: Cell, counts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.zeros((cell.tiles, 3))
    np.add.at(out, cell.tile[keep], counts[keep])
    return out


def run_mixed(rows: Sequence[str], repeats: int, seed: int) -> dict[str, Any]:
    config = yaml.safe_load(MIXED_CONFIG.read_text(encoding="utf-8"))
    dataset, mix_seed = config["dataset"], int(config["seed"])
    levels = [int(state["level"]) for state in config["states"]]
    result: dict[str, Any] = {"dataset": dataset, "levels": levels, "rows": {}}
    weights = None
    for row in rows:
        cell = load_cell(row, dataset)
        keep = assignment(cell, mix_seed, levels)
        if weights is None:
            weights = draws(cell.tiles, repeats, seed + zlib.crc32(f"mixed/{dataset}".encode()) % 100000)
        sums = {rule: tile_totals(cell, counts, keep) for rule, counts in
                (("never", cell.counts["raw"]), ("always", cell.counts["estimated"]), ("selector", cell.selected()))}
        scores = {rule: float(f1(total.sum(axis=0))) for rule, total in sums.items()}
        boot = {rule: f1(resampled(weights, total)) for rule, total in sums.items()}
        result["rows"][row] = {"f1": scores, **{
            f"selector_minus_{rule}": interval(boot["selector"] - boot[rule], scores["selector"] - scores[rule])
            for rule in ("never", "always")}}
    return result


def pts(value: float) -> str:
    return f"{value * 100:+.2f}"


def render(versus: dict[str, Any], mixed: dict[str, Any]) -> str:
    lines = ["[versus always] CARGO minus always correcting, points, 95% tile bootstrap"]
    for name, entry in versus["cells"].items():
        cells = [f"L{level} {pts(entry['selector_minus_always'][str(level)]['value'])} "
                 f"[{pts(entry['selector_minus_always'][str(level)]['low'])}, "
                 f"{pts(entry['selector_minus_always'][str(level)]['high'])}]" for level in (0, 3)]
        lines.append(f"  {name:<22} " + "   ".join(cells))
    for key in ("selector_minus_always", "selector_minus_never"):
        mean = versus["mean_18"][key]
        lines.append(f"  mean of 18, {key}: " + "   ".join(
            f"L{level} {pts(mean[str(level)]['value'])} [{pts(mean[str(level)]['low'])}, {pts(mean[str(level)]['high'])}]"
            for level in LEVELS))
    lines.append(f"  counts: {json.dumps(versus['counts'])}")
    lines.append(f"[mixed {mixed['dataset']}, levels {mixed['levels']}] overall F1 differences")
    for row, entry in mixed["rows"].items():
        lines.append(f"  {row:<8} " + "   ".join(
            f"minus {rule} {pts(entry[f'selector_minus_{rule}']['value'])} "
            f"[{pts(entry[f'selector_minus_{rule}']['low'])}, {pts(entry[f'selector_minus_{rule}']['high'])}]"
            for rule in ("never", "always")))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="reports/p33/statistics")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    cells = [load_cell(row, name) for row in ROWS for name in DATASETS]
    versus = run_versus(cells, args.repeats, args.seed)
    mixed = run_mixed(ROWS, args.repeats, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistics.json").write_text(json.dumps({"versus": versus, "mixed": mixed}, indent=2), encoding="utf-8")
    text = render(versus, mixed)
    (output / "statistics.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
