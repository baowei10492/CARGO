"""DiffRegCD's input convention for the frozen detectors and the proposed modules (main record 192.72).

DiffRegCD perturbs the earlier image and keeps the later one, whose frame is the label's. Here the P30/P31 rows are
evaluated the same way: the whole T1 tile is warped by the crop's draw (rs_cd.p30_geometry), T2 stays; the localizer
registers the perturbed T1 to T2 and corrects T1 on the whole tile; the detectors still receive (earlier, later).
The selectors are the frozen P30/P31 ones with their tau, applied unchanged (trained under the other convention).
Same test crops, draws and scoring pixels as the P30 test run and as DiffRegCD's border-free part, so all rows see
identical inputs. Also the border-free augmentation baseline without correction. Reporting only; the clean raw path
must reproduce the test run. Resumable per row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import affine_parameters, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p27_selector import crop_records
from rs_cd.p30_geometry import BorderFreeGeometry
from tools import run_p30
from tools.p31_input_swap import open_row_detector
from tools.p31_selector_ablation import BORDER_FREE_ROWS, sources
from tools.p31_statistics import anchor_table
from tools.run_p27_selector import harness, open_dataset, with_split, write_csv
from tools.run_p31_plugins import load, release

TOLERANCE = 0.001
AUGMENTATION = "p30_augmented"


class EarlierFirst:
    """crop_records calls detector(reference, moved); here the reference is T2 and the moved image T1."""

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, reference: torch.Tensor, moved: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(moved, reference)


def swapped_records(dataset: Any, spec: dict[str, Any], crops: Sequence[dict[str, Any]], localizer, detector: dict[str, Any],
                    h: dict[str, Any], device: torch.device, label: str) -> list[dict[str, Any]]:
    """rs_cd.p27_selector.crop_records with the roles of the dates exchanged (T1 moved, T2 the reference)."""
    cache = TileCache(dataset, int(spec["tile_size"]))
    records = []
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, h["size"])
        geometry = BorderFreeGeometry(cache.arrays[0], item["top"], item["left"], h["size"])
        _, crop = crop_records(
            localizer, EarlierFirst(detector["model"]), threshold=detector["threshold"], loss_weights=detector["loss"],
            dataset=spec["name"], item=item, t1=t2, t2=t1, mask=mask, mask_threshold=int(spec["mask_threshold"]),
            ranges=h["ranges"], levels=h["levels"], seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
            trigger_probability=h["trigger_probability"], device=device, geometry=geometry)
        for record in crop:
            record["split"] = "test"
        records.extend(crop)
        if index % 500 == 0 or index == len(crops):
            print(f"[swapped {label} {spec['name']} {index}/{len(crops)}]", flush=True)
    return records


@torch.no_grad()
def swapped_raw(dataset: Any, spec: dict[str, Any], crops: Sequence[dict[str, Any]], detector: dict[str, Any],
                h: dict[str, Any], device: torch.device) -> dict[str, float]:
    """F1 per level without correction: (perturbed T1, T2), as tools.run_p30.raw_scores with the dates exchanged."""
    cache = TileCache(dataset, int(spec["tile_size"]))
    levels = [0, *h["levels"]]
    totals = {str(level): BinaryChangeMetrics(threshold=detector["threshold"]) for level in levels}
    for item in crops:
        t1, t2, mask = cache.crop(item, h["size"])
        geometry = BorderFreeGeometry(cache.arrays[0], item["top"], item["left"], h["size"])
        matrices = [perturbation_matrix(*affine_parameters(item["crop_id"], level, h["ranges"], h["seed"]),
                                        width=h["size"], height=h["size"]) for level in h["levels"]]
        valid = scoring_mask(matrices, h["size"])
        images = [t1, *(geometry.perturb(m) for m in matrices)]
        t1_batch = torch.stack([normalize(image) for image in images]).to(device)
        t2_batch = normalize(t2).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
        logits = detector["model"](t1_batch, t2_batch)["change_logits"]
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        for level, counts in zip(levels, confusion_counts(logits, target, keep, detector["threshold"])):
            add_counts(totals[str(level)], counts)
    return {level: float(m.compute()["f1"]) for level, m in totals.items()}


def render(label: str, name: str, entry: dict[str, Any]) -> str:
    f1 = entry["f1"]
    row = lambda rule: " / ".join(f"{100 * f1[level][rule]:.1f}" for level in ("0", "1", "2", "3"))
    return (f"{label:<9} {name:<9} raw {row('raw')} | oracle {row('oracle')} | always {row('estimated')} | "
            f"selector {row('selector')} | success L3 {100 * entry['localization_success']['3']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p31_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", nargs="+", default=[row for row, _, _ in BORDER_FREE_ROWS])
    parser.add_argument("--output-dir", required=True, help="resumable: finished rows are kept")
    args = parser.parse_args()
    config, p30, evaluation, _ = load(Path(args.config))
    evaluation = with_split(evaluation, "test")
    h = harness(evaluation)
    margin = int(p30["crop_margin"])
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    localizer, _ = run_p30.open_localizer(p30, "p30", device)
    labels = {row: label for row, label, _ in BORDER_FREE_ROWS}
    failed = []
    for row in args.rows:
        path = output_dir / f"{row}.json"
        if path.is_file():
            continue
        train_root, _ = sources(row)
        anchors = anchor_table(row)
        result: dict[str, Any] = {"row": row, "convention": "T1 perturbed (whole tile), T2 reference, label in T2 frame",
                                  "selectors": str(train_root / "bundles"), "datasets": {}}
        for spec in evaluation["datasets"]:
            name = spec["name"]
            dataset = open_dataset(spec, "test")
            crops = run_p30.margin_crops(dataset, spec, h["size"], margin)
            detector = open_row_detector(row, name, config, p30, device)
            bundle = torch.load(train_root / "bundles" / f"selector_{name}.pth", map_location="cpu", weights_only=False)
            records = swapped_records(dataset, spec, crops, localizer, detector, h, device, row)
            entry = {"crops": len(crops), **run_p30.system_table(records, bundle)}
            entry["clean_raw_anchor_difference"] = abs(entry["f1"]["0"]["raw"] - float(anchors[name]["0"]["raw"]))
            if entry["clean_raw_anchor_difference"] > TOLERANCE:
                failed.append(f"{row}/{name}")
            if row == "dinov3":
                augmented = run_p30.open_detector(p30, AUGMENTATION, name, device)
                entry[f"{AUGMENTATION}_raw"] = swapped_raw(dataset, spec, crops, augmented, h, device)
                del augmented
            write_csv(output_dir / f"records_{row}_{name}.csv", records)
            result["datasets"][name] = entry
            print(render(labels[row], name, entry), flush=True)
            if row == "dinov3":
                print(f"{'':<9} {name:<9} {AUGMENTATION} raw " + " / ".join(
                    f"{100 * entry[f'{AUGMENTATION}_raw'][k]:.1f}" for k in ("0", "1", "2", "3")), flush=True)
            del detector
            release(device)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failed:
        raise SystemExit(f"clean raw path does not reproduce the test run: {failed}; instrument error")


if __name__ == "__main__":
    main()
