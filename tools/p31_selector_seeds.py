"""Selector training seeds (CPU; main record 192.71).

Every border-free selector (4 detectors x 3 datasets) is retrained on its own training records with the frozen
recipe (configs/p27_selector.yaml) and only the classifier seed changed, then scored on its test records. The
recipe's own seed must reproduce the frozen selector, so run this on the machine that trained the selectors.
Reports the spread of the clean change and the level gains over seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tools.p27_selector_ablation import score, summarize
from tools.p31_selector_ablation import BORDER_FREE_ROWS, DATASETS, SELECTOR_CONFIG, check_bundles, sources
from tools.run_p27_selector import read_records, train_dataset
from tools.run_p27_step1 import sha256

SEEDS = (3407, 1, 2, 3, 4)
UNSAFE = -0.005
KEYS = ("clean_drop", "gain_1", "gain_2", "gain_3")


def cell_seeds(name: str, training: list[dict[str, Any]], members: list[dict[str, Any]], frozen: dict[str, Any],
               config: dict[str, Any], seeds=SEEDS) -> dict[str, Any]:
    out: dict[str, Any] = {"seeds": {}}
    frozen_probability, _, _ = score(members, frozen)
    for seed in seeds:
        bundle, _ = train_dataset(name, training, {**config, "seed": int(seed)})
        probability, table, _ = score(members, bundle)
        out["seeds"][str(seed)] = {"tau": bundle["tau"], **summarize(table, "selector")}
        if int(seed) == int(config["seed"]):
            out["reproduces_frozen"] = {
                "tau_equal": bundle["tau"] == frozen["tau"],
                "max_probability_difference": max(abs(a - b) for a, b in zip(probability, frozen_probability))}
    return out


def spread(cells: dict[str, dict[str, Any]], seeds=SEEDS) -> dict[str, Any]:
    """Per seed the 12-cell summary, then mean and standard deviation over seeds; per cell the spread."""
    per_seed = {}
    for seed in (str(s) for s in seeds):
        values = [cell["seeds"][seed] for cell in cells.values()]
        changes = [-v["clean_drop"] for v in values]
        per_seed[seed] = {"clean_change_mean": float(np.mean(changes)), "clean_change_worst": float(min(changes)),
                          "unsafe": sum(c < UNSAFE for c in changes),
                          **{f"{k}_mean": float(np.mean([v[k] for v in values])) for k in KEYS[1:]}}
    over = {key: {"mean": float(np.mean([s[key] for s in per_seed.values()])),
                  "std": float(np.std([s[key] for s in per_seed.values()], ddof=1))}
            for key in next(iter(per_seed.values()))}
    per_cell = {name: {k: {"mean": float(np.mean([cell["seeds"][s][k] for s in cell["seeds"]])),
                           "std": float(np.std([cell["seeds"][s][k] for s in cell["seeds"]], ddof=1))} for k in KEYS}
                for name, cell in cells.items()}
    return {"per_seed": per_seed, "over_seeds": over, "per_cell": per_cell}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(SELECTOR_CONFIG.read_text(encoding="utf-8"))
    config_sha = sha256(SELECTOR_CONFIG)
    cells = {}
    for row, label, names in BORDER_FREE_ROWS:
        train_root, records = sources(row)
        check_bundles(train_root, names, config_sha)
        for name, path in zip(DATASETS, records):
            members = [r for r in read_records(path) if r["dataset"] == name]
            training = read_records(train_root / "features" / f"features_{name}.csv")
            frozen = torch.load(train_root / "bundles" / f"selector_{name}.pth", map_location="cpu", weights_only=False)
            cells[f"{row}/{name}"] = cell_seeds(name, training, members, frozen, config)
            seeds = cells[f"{row}/{name}"]["seeds"]
            print(f"{row}/{name}: L3 gain " + " ".join(f"{100 * seeds[s]['gain_3']:+.1f}" for s in seeds)
                  + " | clean change " + " ".join(f"{-100 * seeds[s]['clean_drop']:+.1f}" for s in seeds), flush=True)
    result = {"seeds": list(SEEDS), "selector_config_sha256": config_sha, "cells": cells, **spread(cells)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["over_seeds"], indent=1))
    reproduced = all(c["reproduces_frozen"]["tau_equal"] and c["reproduces_frozen"]["max_probability_difference"] < 1e-4
                     for c in cells.values())
    if not reproduced:
        raise SystemExit("the recipe seed does not reproduce a frozen selector; run on the training machine")


if __name__ == "__main__":
    main()
