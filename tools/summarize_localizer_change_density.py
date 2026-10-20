"""A2: is P21 localizer error associated with within-tile change fraction?

docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md. Reads the per-sample decisions the
P21 BIT cells already wrote (conditional_epe_pixels, q_gate) and the
validation masks those cells already consumed. No model is run. The
localizer output does not depend on the detector, so one detector row is
enough.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from rs_cd.change_density import assign_bin, association_verdict, overall_association, spearman


def tile_change_fractions(
    dataset_root: Path, manifest: Path, *, split: str, threshold: int
) -> dict[str, float]:
    fractions: dict[str, float] = {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            mask = np.asarray(Image.open(dataset_root / row["mask_path"]).convert("L"), dtype=np.uint8)
            fractions[row["sample_id"]] = float((mask > threshold).mean())
    return fractions


def load_decisions(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_scenario.setdefault(row["scenario"], {})[row["id"]] = row
    return by_scenario


def analyse_cell(
    *, dataset: str, report_root: Path, dataset_root: Path, manifest: Path,
    threshold: int, bins: list[tuple[float, float]], thresholds: dict[str, float],
) -> dict[str, Any]:
    decisions = load_decisions(report_root / "safe_alignment_experiment" / "validation_sample_decisions.csv")
    fractions = tile_change_fractions(dataset_root, manifest, split="val", threshold=threshold)
    missing = [sid for sid in decisions["shift:s3:t2"] if sid not in fractions]
    if missing:
        raise SystemExit(f"{dataset}: {len(missing)} decision ids have no validation mask, e.g. {missing[:3]}")

    ids = sorted(decisions["shift:s3:t2"])
    change = [fractions[sid] for sid in ids]
    s3 = [float(decisions["shift:s3:t2"][sid]["conditional_epe_pixels"]) for sid in ids]
    s2 = [float(decisions["shift:s2:t2"][sid]["conditional_epe_pixels"]) for sid in ids]
    clean_q = [1.0 if decisions["clean:s0:common"][sid]["q_gate"] == "True" else 0.0 for sid in ids]

    table = []
    for index, (low, high) in enumerate(bins):
        members = [i for i, value in enumerate(change) if assign_bin(value, bins) == index]
        table.append(
            {
                "range": [low, high],
                "n": len(members),
                "mean_s2_epe": statistics.fmean(s2[i] for i in members) if members else None,
                "mean_s3_epe": statistics.fmean(s3[i] for i in members) if members else None,
                "clean_q_rate": statistics.fmean(clean_q[i] for i in members) if members else None,
            }
        )
    verdict = association_verdict(change, s3, thresholds=thresholds, bins=bins)
    return {
        "dataset": dataset,
        "report": str(report_root),
        "n": len(ids),
        "mean_change_fraction": statistics.fmean(change),
        "verdict": verdict["verdict"],
        "association_s3": verdict,
        "spearman_change_vs_s2_epe": spearman(change, s2),
        "spearman_change_vs_clean_q": spearman(change, clean_q),
        "bins": table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p22_change_density_probe.yaml")
    parser.add_argument("--matrix", default="configs/p21_matrix.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    probe = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    bins = [tuple(pair) for pair in probe["bins"]]
    detector_slug = matrix["detectors"][probe["a2_detector"]]["slug"]
    cells = {}
    for dataset in probe["a2_datasets"]:
        entry = matrix["datasets"][dataset]
        stage1 = yaml.safe_load(
            Path(matrix["stage1_config_pattern"].format(detector=detector_slug, dataset=entry["slug"])).read_text(encoding="utf-8")
        )
        report_root = Path(Path(f"reports/p21/cells/{detector_slug}_{entry['slug']}/LATEST").read_text().strip())
        cells[dataset] = analyse_cell(
            dataset=dataset, report_root=report_root, dataset_root=Path(entry["dataset_root"]),
            manifest=Path(entry["manifest"]), threshold=int(stage1["data"]["mask_threshold"]),
            bins=bins, thresholds=probe["a2_thresholds"],
        )
        cell = cells[dataset]
        print(
            f"{dataset:9s} n={cell['n']:4d} mean_change={cell['mean_change_fraction']:.3f} "
            f"rho(change,S3 EPE)={cell['association_s3']['spearman']} verdict={cell['verdict']}"
        )
        for row in cell["bins"]:
            print(f"    {row['range']} n={row['n']:4d} S2 EPE={row['mean_s2_epe']} S3 EPE={row['mean_s3_epe']} clean q={row['clean_q_rate']}")

    result = {
        "diagnostic": "p22_a2_cross_dataset_change_density",
        "overall": overall_association(cells["DSIFN-CD"]["verdict"], cells["GZ-CD"]["verdict"]),
        "reference_dataset": "LEVIR-CD (not part of the overall verdict)",
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("A2 overall:", result["overall"])


if __name__ == "__main__":
    main()
