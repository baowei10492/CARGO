"""Qualitative panels under the border-free protocol (the P30 test run, DINOv3 row).

As tools/p27_qualitative.py (same categories, colours and tiles), but the crops are the P30 test crops and
both the perturbation and the correction act on the whole T2 tile (rs_cd.p30_geometry), with localizer v2
and the P30 selectors. Candidates of every category are rendered with their tiles; the paper figure is
assembled from the chosen tiles (tools/p27_figure_qualitative.py). Each index entry records the rendered
crop F1 next to the test record's, as a drift check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import scoring_mask
from rs_cd.affine_registration import perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.localizer_training import TrainedLocalizer, step1_rows
from rs_cd.p27_selector import matrix_of
from rs_cd.p30_geometry import BorderFreeGeometry
from tools import run_p30
from tools.p27_qualitative import VIEWS, crop_f1, error_map, find_record, masked_f1, panel, pick
from tools.run_p27_selector import harness, open_dataset, read_records, with_split


def parse_crop(crop_id: str) -> tuple[str, int, int]:
    tile, position = crop_id.rsplit("|", 1)
    top, left = position[1:].split("x")
    return tile, int(top), int(left)


@torch.no_grad()
def example_views(record: dict[str, Any], spec: dict[str, Any], evaluation: dict[str, Any], localizer: TrainedLocalizer,
                  detector: dict[str, Any], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    h = harness(evaluation)
    size = h["size"]
    dataset = open_dataset(spec, spec["split"])
    tile, top, left = parse_crop(record["crop_id"])
    row = next(i for i, r in enumerate(dataset.rows) if r.sample_id == tile)
    item = {"row": row, "sample_id": tile, "top": top, "left": left, "crop_id": record["crop_id"]}
    cache = TileCache(dataset, int(spec["tile_size"]))
    t1, t2, mask = cache.crop(item, size)
    geometry = BorderFreeGeometry(cache.arrays[1], top, left, size)
    rows = step1_rows(localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2, change_fraction=0.0,
                      ranges=h["ranges"], levels=h["levels"], seed=h["seed"], points=h["points"],
                      tolerance=h["tolerance"], trigger_probability=h["trigger_probability"],
                      make_target=geometry.perturb)
    truths = {int(r["level"]): perturbation_matrix(float(r["angle"]), float(r["scale"]), float(r["tx"]), float(r["ty"]),
                                                   width=size, height=size) for r in rows}
    level = int(record["level"])
    estimate = matrix_of(next(r for r in rows if int(r["level"]) == level))
    views = geometry.path_images(None, truths[level], estimate)
    batch = torch.stack([normalize(views["raw"]), normalize(views["estimated"])]).to(device)
    logits = detector["model"](normalize(t1)[None].repeat(2, 1, 1, 1).to(device), batch)["change_logits"]
    predictions = (torch.sigmoid(logits.float()) >= detector["threshold"]).cpu().numpy()[:, 0]
    truth = mask > int(spec["mask_threshold"])
    valid = scoring_mask([truths[k] for k in h["levels"]], size)
    selected = predictions[1] if float(record["selector_probability"]) >= float(record["tau"]) else predictions[0]
    reference = np.repeat((truth * 255).astype(np.uint8)[..., None], 3, axis=2)
    outputs = {"without_correction": predictions[0], "localizer_only": predictions[1], "with_selector": selected}
    images = {"t1": t1, "t2_input": views["raw"], "t2_corrected": views["estimated"], "reference": reference,
              **{name: error_map(prediction, truth, valid) for name, prediction in outputs.items()}}
    return images, {name: masked_f1(prediction, truth, valid) for name, prediction in outputs.items()}


def load_members(test_root: Path, bundle_dir: Path, names: list[str]) -> dict[str, list[dict[str, Any]]]:
    members = {}
    for name in names:
        bundle = torch.load(bundle_dir / f"selector_{name}.pth", map_location="cpu", weights_only=False)
        tau = 2.0 if bundle["tau"] is None else float(bundle["tau"])
        members[name] = [{**r, "tau": tau} for r in read_records(test_root / f"records_proposed_{name}.csv")]
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p30_border_free.yaml")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--select", default=None, help="JSON list of {dataset, crop_id, level}: render only these")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    p30, evaluation, _ = run_p30.load(Path(args.config))
    evaluation = with_split(evaluation, "test")
    specs = {spec["name"]: spec for spec in evaluation["datasets"]}
    members = load_members(run_p30.pointed("reports/p30/test/LATEST"),
                           run_p30.pointed("reports/p30/selector/LATEST") / "bundles", list(specs))
    if args.select:
        wanted = json.loads(Path(args.select).read_text(encoding="utf-8"))
        jobs = [(w["dataset"], "selected", number, find_record(members[w["dataset"]], w["crop_id"], w["level"]))
                for number, w in enumerate(wanted, start=1)]
    else:
        jobs = [(name, category, number, record) for name in specs
                for category, examples in pick(members[name], args.per_category).items()
                for number, record in enumerate(examples, start=1)]
    device = torch.device(args.device)
    localizer, _ = run_p30.open_localizer(p30, "p30", device)
    detectors = {name: run_p30.open_detector(p30, "original", name, device) for name in specs}
    output_dir = Path(args.output_dir)
    (output_dir / "tiles").mkdir(parents=True, exist_ok=True)
    index = []
    for name, category, number, record in jobs:
        images, scores = example_views(record, specs[name], evaluation, localizer, detectors[name], device)
        stem = f"{name}_{category}_{number}"
        title = (f"{name} {record['crop_id']} level {record['level']}  "
                 f"P(safe)={float(record['selector_probability']):.2f} tau={float(record['tau']):.2f}  "
                 f"crop F1 raw {crop_f1(record, 'raw'):.2f} / corrected {crop_f1(record, 'estimated'):.2f}")
        panel([images[view] for view in VIEWS], title).save(output_dir / f"{stem}.png")
        tiles = {}
        for view in VIEWS:
            tiles[view] = f"tiles/{stem}_{view}.png"
            Image.fromarray(images[view]).save(output_dir / tiles[view])
        drift = max(abs(scores["without_correction"] - crop_f1(record, "raw")),
                    abs(scores["localizer_only"] - crop_f1(record, "estimated")))
        index.append({"file": f"{stem}.png", "dataset": name, "category": category, "order": len(index) + 1,
                      "crop_id": record["crop_id"], "level": int(record["level"]),
                      "selector_probability": float(record["selector_probability"]), "tau": float(record["tau"]),
                      "crop_f1_raw": crop_f1(record, "raw"), "crop_f1_corrected": crop_f1(record, "estimated"),
                      "change_fraction": float(record["change_fraction"]),
                      "rendered_f1": scores, "rendered_drift": drift, "tiles": tiles})
        print(output_dir / f"{stem}.png", f"(rendered F1 drift {drift:.4f})", flush=True)
    (output_dir / "qualitative_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
