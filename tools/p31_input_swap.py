"""Does a frozen detector compare the two dates? Replace the earlier image (main record 192.71).

The counterpart, for the four frozen detectors, of the DiffRegCD mechanism check (tools/p27_diffregcd_rerun.py,
part "mechanism"): on the aligned P30 test crops the detector sees (T1, T2) as usual, then the earlier image is
replaced by the later one (same_date), by the T1 of another tile (unrelated) and by zeros (black), T2 unchanged.
A detector that compares the dates loses almost everything once T1 is gone. P27 scoring pixels, as the test run;
the normal view must reproduce the test run's clean raw F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import affine_parameters, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.metrics.binary import BinaryChangeMetrics
from tools import run_p30
from tools.p31_dose_response import other_index
from tools.p31_selector_ablation import BORDER_FREE_ROWS
from tools.p31_statistics import anchor_table
from tools.run_p27_selector import harness, open_dataset, with_split
from tools.run_p31_plugins import load, release, row_detector

VIEWS = ("normal", "same_date", "unrelated", "black")
TOLERANCE = 0.001


@torch.no_grad()
def swap_scores(spec: dict[str, Any], evaluation: dict[str, Any], detector: dict[str, Any], device: torch.device, *,
                margin: int, max_crops: int | None = None) -> dict[str, dict[str, float]]:
    h = harness(evaluation)
    size = h["size"]
    dataset = open_dataset(spec, spec["split"])
    crops = run_p30.margin_crops(dataset, spec, size, margin)
    crops = crops[:max_crops] if max_crops is not None else crops
    cache, other_cache = TileCache(dataset, int(spec["tile_size"])), TileCache(dataset, int(spec["tile_size"]))
    totals = {view: BinaryChangeMetrics(threshold=detector["threshold"]) for view in VIEWS}
    for index, item in enumerate(crops):
        t1, t2, mask = cache.crop(item, size)
        other_t1 = other_cache.crop(crops[other_index(crops, index)], size)[0]
        matrices = [perturbation_matrix(*affine_parameters(item["crop_id"], level, h["ranges"], h["seed"]),
                                        width=size, height=size) for level in h["levels"]]
        valid = scoring_mask(matrices, size)
        earlier = {"normal": t1, "same_date": t2, "unrelated": other_t1, "black": np.zeros_like(t1)}
        t1_batch = torch.stack([normalize(earlier[view]) for view in VIEWS]).to(device)
        t2_batch = normalize(t2).unsqueeze(0).repeat(len(VIEWS), 1, 1, 1).to(device)
        logits = detector["model"](t1_batch, t2_batch)["change_logits"]
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        for view, counts in zip(VIEWS, confusion_counts(logits, target, keep, detector["threshold"])):
            add_counts(totals[view], counts)
    computed = {view: metrics.compute() for view, metrics in totals.items()}
    return {key: {view: float(values[key]) for view, values in computed.items()} for key in ("f1", "precision", "recall")}


def open_row_detector(row: str, name: str, config: dict[str, Any], p30: dict[str, Any], device: torch.device):
    return run_p30.open_detector(p30, "original", name, device) if row == "dinov3" else row_detector(config, row, name, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p31_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, p30, evaluation, _ = load(Path(args.config))
    evaluation = with_split(evaluation, "test")
    device = torch.device(args.device)
    result: dict[str, Any] = {"views": VIEWS, "rows": {}}
    failed = []
    for row, label, names in BORDER_FREE_ROWS:
        anchors = anchor_table(row)
        result["rows"][row] = {}
        for spec in evaluation["datasets"]:
            name = spec["name"]
            if name not in names:
                continue
            detector = open_row_detector(row, name, config, p30, device)
            scores = swap_scores(spec, evaluation, detector, device, margin=int(p30["crop_margin"]))
            difference = abs(scores["f1"]["normal"] - float(anchors[name]["0"]["raw"]))
            scores["anchor_difference"] = difference
            result["rows"][row][name] = scores
            if difference > TOLERANCE:
                failed.append(f"{row}/{name}")
            print(f"{label:<9} {name:<9} " + "  ".join(f"{v} {100 * scores['f1'][v]:5.1f}" for v in VIEWS)
                  + f"  (anchor {difference:.4f})", flush=True)
            del detector
            release(device)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"normal view does not reproduce the clean raw F1: {failed}; instrument error")


if __name__ == "__main__":
    main()
