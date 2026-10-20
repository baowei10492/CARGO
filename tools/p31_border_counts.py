"""Same-crop comparison of the zero border, with per-crop counts (final measurements, record 192.66).

For the DINOv3 detector and the two augmentation baselines (zero-border and border-free training), on the
P30 test crops and draws: the pair as is (clean), the border-free perturbation (the whole T2 tile warped,
as in P30) and the per-crop perturbation with a zero border (as in P27), levels 1-3, all scored on the
P27 scoring pixels of the crop. Writes per-crop confusion counts (for bootstrap intervals) and pooled F1.
The clean and border-free F1 must reproduce the P30 test run (<= 0.001), else the run stops.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import affine_parameters, perturb, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p30_geometry import BorderFreeGeometry
from tools import run_p30
from tools.run_p27_selector import harness, open_dataset, with_split

GROUPS = ("original", "p29_augmented", "p30_augmented")
VIEWS = ("clean", "border_free", "zero_border")
FIELDS = ("dataset", "group", "tile_id", "crop_id", "view", "level", "tp", "fp", "fn", "tn")
TOLERANCE = 0.001


def crop_views(t2: np.ndarray, geometry: BorderFreeGeometry, crop_id: str, h: dict[str, Any]
               ) -> tuple[list[tuple[str, int, np.ndarray]], np.ndarray]:
    """(view, level, T2 image) for one crop, and its scoring pixels (the P27 rule, as in run_p30.raw_scores)."""
    size = h["size"]
    matrices = {level: perturbation_matrix(*affine_parameters(crop_id, level, h["ranges"], h["seed"]),
                                           width=size, height=size) for level in h["levels"]}
    views = [("clean", 0, t2)]
    for level, matrix in matrices.items():
        views.append(("border_free", level, geometry.perturb(matrix)))
        views.append(("zero_border", level, perturb(t2, matrix)))
    return views, scoring_mask(list(matrices.values()), size)


@torch.no_grad()
def run_dataset(spec: dict[str, Any], evaluation: dict[str, Any], detectors: dict[str, dict[str, Any]], *,
                margin: int, device: torch.device, max_crops: int | None = None
                ) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    h = harness(evaluation)
    size = h["size"]
    dataset = open_dataset(spec, spec["split"])
    crops = run_p30.margin_crops(dataset, spec, size, margin)
    crops = crops[:max_crops] if max_crops is not None else crops
    cache = TileCache(dataset, int(spec["tile_size"]))
    totals = {group: {f"{view}/{level}": BinaryChangeMetrics(threshold=detectors[group]["threshold"])
                      for view in VIEWS for level in ([0] if view == "clean" else h["levels"])}
              for group in detectors}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], size)
        views, valid = crop_views(t2, geometry, item["crop_id"], h)
        t2_batch = torch.stack([normalize(image) for _, _, image in views]).to(device)
        t1_batch = normalize(t1).unsqueeze(0).repeat(len(views), 1, 1, 1).to(device)
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        for group, detector in detectors.items():
            logits = detector["model"](t1_batch, t2_batch)["change_logits"]
            for (view, level, _), counts in zip(views, confusion_counts(logits, target, keep, detector["threshold"])):
                add_counts(totals[group][f"{view}/{level}"], counts)
                rows.append(dict(zip(FIELDS, (spec["name"], group, item["sample_id"], item["crop_id"], view, level,
                                             *(int(c) for c in counts)))))
        if index % 500 == 0 or index == len(crops):
            print(f"[p31 border counts {spec['name']} {index}/{len(crops)}]", flush=True)
    f1 = {group: {key: float(metrics.compute()["f1"]) for key, metrics in values.items()}
          for group, values in totals.items()}
    return rows, f1


def anchor(f1: dict[str, dict[str, float]], p30_entry: dict[str, Any], levels: list[int]) -> dict[str, Any]:
    """Clean and border-free F1 against the P30 test run's rows of the same detectors."""
    differences = {}
    for group in GROUPS:
        if group not in f1:
            continue
        reference = p30_entry[group]
        differences[f"{group}/clean"] = abs(f1[group]["clean/0"] - float(reference["0"]))
        for level in levels:
            differences[f"{group}/{level}"] = abs(f1[group][f"border_free/{level}"] - float(reference[str(level)]))
    largest = max(differences.values())
    return {"max_abs_difference": largest, "tolerance": TOLERANCE, "passed": largest <= TOLERANCE,
            "differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p30_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    p30, evaluation, _ = run_p30.load(Path(args.config))
    evaluation = with_split(evaluation, "test")
    p30_test = json.loads((run_p30.pointed("reports/p30/test/LATEST") / "p30_test.json").read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    levels = harness(evaluation)["levels"]
    summary: dict[str, Any] = {"crop_margin": int(p30["crop_margin"]), "datasets": {}}
    failed = []
    for spec in evaluation["datasets"]:
        name = spec["name"]
        detectors = {group: run_p30.open_detector(p30, group, name, device) for group in GROUPS}
        rows, f1 = run_dataset(spec, evaluation, detectors, margin=int(p30["crop_margin"]), device=device)
        with (output_dir / f"counts_{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        check = anchor(f1, p30_test["datasets"][name], levels)
        summary["datasets"][name] = {"f1": f1, "anchor": check,
                                     "detectors": {group: d["sha256"] for group, d in detectors.items()}}
        print(f"[{name}] level 3 F1 %: " + "; ".join(
            f"{group} border-free {f1[group]['border_free/3'] * 100:.1f} zero-border {f1[group]['zero_border/3'] * 100:.1f}"
            for group in GROUPS) + f"; anchor max diff {check['max_abs_difference']:.5f}", flush=True)
        if not check["passed"]:
            failed.append(name)
        del detectors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"clean / border-free F1 do not reproduce the P30 test run on {failed}: instrument error")


if __name__ == "__main__":
    main()
