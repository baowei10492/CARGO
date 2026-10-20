"""P27 qualitative panels from a finished test run (DINOv3 row by default).

Examples are picked from the per-crop test records, per dataset:
  corrected   level 3, the selector corrected and the crop F1 rose the most
  kept_clean  clean crops the localizer gate would have moved, the selector kept,
              and moving them would have cost the most
  failure     level 3, the selector's choice was the worse of the two paths
With --select, only the listed crops are rendered, in the listed order (the
paper figure; tools/p27_figure_qualitative.py assembles it from the tiles).
Each panel: T1 | T2 as given | T2 corrected | reference | error of the
uncorrected prediction | error of the always-corrected prediction | error of
the selector's output. Error colours: true positive white, false positive red,
false negative blue; pixels outside the scoring region are grey. Every view is
also saved as its own tile, with the crop F1 of each error view.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import path_images, scoring_mask
from rs_cd.affine_registration import AffineRanges, perturb, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.localizer_training import TrainedLocalizer, load_localizer, step1_rows
from rs_cd.p27_selector import matrix_of
from tools.run_p27_f1 import open_detectors
from tools.run_p27_selector import load_config, open_dataset, read_records, with_split

VIEWS = ("t1", "t2_input", "t2_corrected", "reference", "without_correction", "localizer_only", "with_selector")
LABELS = ("T1", "T2 (input)", "T2 corrected", "Reference", "Without correction", "Localizer only", "With selector")
ERROR_VIEWS = ("without_correction", "localizer_only", "with_selector")


def f1_of(tp: int, fp: int, fn: int) -> float:
    return 1.0 if tp + fp + fn == 0 else 2.0 * tp / (2.0 * tp + fp + fn)


def crop_f1(record: dict[str, Any], path: str) -> float:
    return f1_of(*(int(record[f"{path}_{key}"]) for key in ("tp", "fp", "fn")))


def masked_f1(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    return f1_of(int((prediction & truth & valid).sum()), int((prediction & ~truth & valid).sum()),
                 int((~prediction & truth & valid).sum()))


def pick(records: Sequence[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    def used(r: dict[str, Any]) -> bool:
        return float(r["selector_probability"]) >= float(r["tau"])

    def benefit(r: dict[str, Any]) -> float:
        return crop_f1(r, "estimated") - crop_f1(r, "raw")

    level3 = [r for r in records if int(r["level"]) == 3 and float(r["change_fraction"]) >= 0.02]
    corrected = sorted((r for r in level3 if used(r)), key=benefit, reverse=True)
    kept = sorted((r for r in records if int(r["level"]) == 0 and r["trigger"] and not used(r)), key=benefit)
    failures = sorted((r for r in level3 if used(r) != (benefit(r) > 0) and benefit(r) != 0),
                      key=lambda r: abs(benefit(r)), reverse=True)
    return {"corrected": corrected[:count], "kept_clean": kept[:count], "failure": failures[:count]}


def error_map(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    image = np.zeros(truth.shape + (3,), dtype=np.uint8)
    image[prediction & truth] = (255, 255, 255)
    image[prediction & ~truth] = (220, 40, 40)
    image[~prediction & truth] = (40, 90, 230)
    image[~valid] = (128, 128, 128)
    return image


def panel(views: Sequence[np.ndarray], title: str) -> Image.Image:
    size, gap, header = views[0].shape[0], 6, 34
    canvas = Image.new("RGB", (len(views) * (size + gap) - gap, size + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (view, label) in enumerate(zip(views, LABELS)):
        left = index * (size + gap)
        canvas.paste(Image.fromarray(view), (left, header))
        draw.text((left + 2, 2), label, fill="black")
    draw.text((2, 18), title, fill="black")
    return canvas


@torch.no_grad()
def example_views(record: dict[str, Any], spec: dict[str, Any], evaluation: dict[str, Any], localizer: TrainedLocalizer,
                  detector: dict[str, Any], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    size = int(evaluation["perturbation"]["crop_size"])
    levels = [int(level) for level in evaluation["perturbation"]["levels"]]
    low, high = evaluation["perturbation"]["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(evaluation["perturbation"]["max_translation_px"]), len(levels))
    dataset = open_dataset(spec, spec["split"])
    tile, top, left = record["crop_id"].rsplit("|", 1)[0], *record["crop_id"].rsplit("|", 1)[1][1:].split("x")
    row = next(i for i, r in enumerate(dataset.rows) if r.sample_id == tile)
    item = {"row": row, "sample_id": tile, "top": int(top), "left": int(left), "crop_id": record["crop_id"]}
    t1, t2, mask = TileCache(dataset, int(spec["tile_size"])).crop(item, size)
    rows = step1_rows(localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2, change_fraction=0.0,
                      ranges=ranges, levels=levels, seed=int(evaluation["seed"]),
                      points=np.zeros((1, 2)), tolerance=2.0,
                      trigger_probability=float(evaluation["p27"]["trigger_probability"]))
    truths = {int(r["level"]): perturbation_matrix(float(r["angle"]), float(r["scale"]), float(r["tx"]),
                                                   float(r["ty"]), width=size, height=size) for r in rows}
    level = int(record["level"])
    estimate = matrix_of(next(r for r in rows if int(r["level"]) == level))
    perturbed = t2 if level == 0 else perturb(t2, truths[level])
    views = path_images(perturbed, truths[level], estimate)
    batch = torch.stack([normalize(views["raw"]), normalize(views["estimated"])]).to(device)
    logits = detector["model"](normalize(t1)[None].repeat(2, 1, 1, 1).to(device), batch)["change_logits"]
    predictions = (torch.sigmoid(logits.float()) >= detector["threshold"]).cpu().numpy()[:, 0]
    truth = mask > int(spec["mask_threshold"])
    valid = scoring_mask([truths[k] for k in levels], size)
    selected = predictions[1] if float(record["selector_probability"]) >= float(record["tau"]) else predictions[0]
    reference = np.repeat((truth * 255).astype(np.uint8)[..., None], 3, axis=2)
    outputs = {"without_correction": predictions[0], "localizer_only": predictions[1], "with_selector": selected}
    images = {"t1": t1, "t2_input": perturbed, "t2_corrected": views["estimated"], "reference": reference,
              **{name: error_map(prediction, truth, valid) for name, prediction in outputs.items()}}
    return images, {name: masked_f1(prediction, truth, valid) for name, prediction in outputs.items()}


def render_example(record: dict[str, Any], spec: dict[str, Any], evaluation: dict[str, Any], localizer: TrainedLocalizer,
                   detector: dict[str, Any], device: torch.device) -> Image.Image:
    images, _ = example_views(record, spec, evaluation, localizer, detector, device)
    title = (f"{spec['name']} {record['crop_id']} level {record['level']}  "
             f"P(safe)={float(record['selector_probability']):.2f} tau={float(record['tau']):.2f}  "
             f"crop F1 raw {crop_f1(record, 'raw'):.2f} / corrected {crop_f1(record, 'estimated'):.2f}")
    return panel([images[name] for name in VIEWS], title)


def find_record(records: Sequence[dict[str, Any]], crop_id: str, level: int) -> dict[str, Any]:
    found = [r for r in records if r["crop_id"] == crop_id and int(r["level"]) == int(level)]
    if len(found) != 1:
        raise SystemExit(f"{len(found)} test records for {crop_id} level {level}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_selector.yaml")
    parser.add_argument("--test-root", required=True, help="a finished test report root (run/test_records.csv)")
    parser.add_argument("--train-root", required=True, help="the selector training root of that test")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--select", default=None,
                        help="JSON list of {dataset, crop_id, level}: render only these, in this order")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, f1_config, evaluation = load_config(Path(args.config))
    evaluation = with_split(evaluation, "test")
    specs = {spec["name"]: spec for spec in evaluation["datasets"]}
    records = read_records(Path(args.test_root) / "run" / "test_records.csv")
    members = {}
    for name in specs:
        bundle = torch.load(Path(args.train_root) / "bundles" / f"selector_{name}.pth", map_location="cpu",
                            weights_only=False)
        tau = 2.0 if bundle["tau"] is None else float(bundle["tau"])
        members[name] = [{**r, "tau": tau} for r in records if r["dataset"] == name]
    if args.select:
        wanted = json.loads(Path(args.select).read_text(encoding="utf-8"))
        jobs = [(w["dataset"], "selected", number, find_record(members[w["dataset"]], w["crop_id"], w["level"]))
                for number, w in enumerate(wanted, start=1)]
    else:
        jobs = [(name, category, number, record) for name in specs
                for category, examples in pick(members[name], args.per_category).items()
                for number, record in enumerate(examples, start=1)]

    device = torch.device(args.device)
    run = config["localizer_run"]
    localizer = TrainedLocalizer(load_localizer(str(Path(run) / "checkpoints" / config["localizer_checkpoint"]), device), device)
    detectors = open_detectors(f1_config, device)
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
        # The records hold the scores of the test run; the rendered ones are what the tiles show.
        drift = max(abs(scores["without_correction"] - crop_f1(record, "raw")),
                    abs(scores["localizer_only"] - crop_f1(record, "estimated")))
        index.append({"file": f"{stem}.png", "dataset": name, "category": category, "order": len(index) + 1,
                      "crop_id": record["crop_id"], "level": int(record["level"]),
                      "selector_probability": float(record["selector_probability"]), "tau": float(record["tau"]),
                      "crop_f1_raw": crop_f1(record, "raw"), "crop_f1_corrected": crop_f1(record, "estimated"),
                      "rendered_f1": scores, "rendered_drift": drift, "tiles": tiles})
        print(output_dir / f"{stem}.png", f"(rendered F1 drift {drift:.4f})", flush=True)
    (output_dir / "qualitative_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
