"""P30 border-free comparison (docs/P30_BORDER_FREE_PROTOCOL.md, configs/p30_border_free.yaml).

Commands:
  features   selector-training records on border-free training crops (P30 localizer, original detectors)
  train      the P30 selectors (configs/p27_selector.yaml recipe, unchanged)
  evaluate   rows 1-5 on validation (--split val), then on the test split once (--split test)

Every crop keeps ``crop_margin`` real pixels to the tile edge; the perturbation and
the correction act on the whole tile (rs_cd.p30_geometry), so no image a detector
or a localizer sees carries a zero border.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_evenly, select_margin_crops
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import affine_parameters, perturbation_matrix
from rs_cd.datasets.localizer_pairs import in_holdout, normalize
from rs_cd.localizer_training import TrainedLocalizer, load_localizer
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p27_selector import crop_records, f1_table, feature_matrix, selection_rates, selector_probability, selector_split
from rs_cd.p30_geometry import BorderFreeGeometry
from tools.run_p27_f1 import open_detectors
from tools.run_p27_selector import (
    SPLITS,
    check_sources,
    harness,
    load_bundles,
    open_dataset,
    read_records,
    render_calibration,
    train_dataset,
    with_split,
    write_csv,
)
from tools.run_p27_step1 import load_config as load_evaluation_config
from tools.run_p27_step1 import sha256

MARKER_DIR = Path("reports/p30/markers")
LEVELS = ("0", "1", "2", "3")


# ---------------------------------------------------------------- config
def load(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(P30 config, evaluation config, selector recipe), with the chained hashes checked."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("the P30 config is not marked preregistered")
    for key in ("evaluation_config", "selector_config"):
        if sha256(Path(config[key])) != config[f"{key}_sha256"]:
            raise SystemExit(f"{config[key]} was changed")
    evaluation = load_evaluation_config(Path(config["evaluation_config"]))
    selector = yaml.safe_load(Path(config["selector_config"]).read_text(encoding="utf-8"))
    return config, evaluation, selector


def pointed(path: str) -> Path:
    pointer = Path(path)
    if not pointer.is_file():
        raise SystemExit(f"{pointer} does not exist yet")
    return Path(pointer.read_text(encoding="utf-8").strip())


def localizer_checkpoint(config: dict[str, Any], which: str) -> Path:
    entry = config["localizers"][which]
    run = pointed(entry["pointer"]) if "pointer" in entry else Path(entry["run"])
    return run / "checkpoints" / "best.pth"


def open_localizer(config: dict[str, Any], which: str, device: torch.device) -> tuple[TrainedLocalizer, dict[str, Any]]:
    checkpoint = localizer_checkpoint(config, which)
    run_config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text(encoding="utf-8"))
    return TrainedLocalizer(load_localizer(str(checkpoint), device), device), run_config


def detector_spec(config: dict[str, Any], group: str, name: str) -> dict[str, Any]:
    entry = config["detectors"][group]
    if "config" in entry:
        path = Path(entry["config"])
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"{path} was changed")
        entry = yaml.safe_load(path.read_text(encoding="utf-8"))["detectors"]
    return entry[name]


def open_detector(config: dict[str, Any], group: str, name: str, device: torch.device) -> dict[str, Any]:
    return open_detectors({"detectors": {name: detector_spec(config, group, name)}}, device)[name]


# ----------------------------------------------------------------- crops
def margin_crops(dataset: Any, spec: dict[str, Any], size: int, margin: int) -> list[dict[str, Any]]:
    return select_margin_crops(dataset.rows, int(spec["tile_size"]), size, margin)


def training_items(dataset: Any, spec: dict[str, Any], selector: dict[str, Any], localizer_config: dict[str, Any],
                   size: int, margin: int) -> list[tuple[dict[str, Any], str]]:
    """As tools.run_p27_selector.training_items, on margin crops."""
    fraction, seed = float(localizer_config["holdout"]["fraction"]), int(localizer_config["seed"])
    held_rows = [in_holdout(row.sample_id, fraction, seed) for row in dataset.rows]
    crops = margin_crops(dataset, spec, size, margin)
    held = [crop for crop in crops if held_rows[crop["row"]]]
    rest = [crop for crop in crops if not held_rows[crop["row"]]]
    rest = select_evenly(rest, int(selector["training_crops"]["max_fit_crops_per_dataset"])) if rest else []
    early = float(selector["training_crops"]["early_stop_fraction"])
    return [(crop, selector_split(crop["sample_id"], holdout=held_rows[crop["row"]], seed=int(selector["seed"]),
                                  early_stop_fraction=early)) for crop in held + rest]


def records_for(dataset: Any, spec: dict[str, Any], items: Sequence[tuple[dict[str, Any], str]],
                localizer: TrainedLocalizer, detector: dict[str, Any], h: dict[str, Any], device: torch.device,
                label: str) -> list[dict[str, Any]]:
    cache = TileCache(dataset, int(spec["tile_size"]))
    records = []
    for index, (item, split) in enumerate(items, start=1):
        t1, t2, mask = cache.crop(item, h["size"])
        geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], h["size"])
        _, crop = crop_records(
            localizer, detector["model"], threshold=detector["threshold"], loss_weights=detector["loss"],
            dataset=spec["name"], item=item, t1=t1, t2=t2, mask=mask, mask_threshold=int(spec["mask_threshold"]),
            ranges=h["ranges"], levels=h["levels"], seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
            trigger_probability=h["trigger_probability"], device=device, geometry=geometry,
        )
        for record in crop:
            record["split"] = split
        records.extend(crop)
        if index % 200 == 0 or index == len(items):
            print(f"[p30 {label} {spec['name']} {index}/{len(items)}]", flush=True)
    return records


@torch.no_grad()
def raw_scores(dataset: Any, spec: dict[str, Any], crops: Sequence[dict[str, Any]], detector: dict[str, Any],
               h: dict[str, Any], device: torch.device) -> dict[str, float]:
    """F1 per level of a detector given the border-free misaligned pair as it is (no correction).

    Same crops, draws and scoring pixels as crop_records.
    """
    cache = TileCache(dataset, int(spec["tile_size"]))
    levels = [0, *h["levels"]]
    totals = {str(level): BinaryChangeMetrics(threshold=detector["threshold"]) for level in levels}
    for item in crops:
        t1, t2, mask = cache.crop(item, h["size"])
        geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], h["size"])
        matrices = [perturbation_matrix(*affine_parameters(item["crop_id"], level, h["ranges"], h["seed"]),
                                        width=h["size"], height=h["size"]) for level in h["levels"]]
        valid = scoring_mask(matrices, h["size"])
        images = [t2, *(geometry.perturb(m) for m in matrices)]
        t2_batch = torch.stack([normalize(image) for image in images]).to(device)
        t1_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
        logits = detector["model"](t1_batch, t2_batch)["change_logits"]
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        for level, counts in zip(levels, confusion_counts(logits, target, keep, detector["threshold"])):
            add_counts(totals[str(level)], counts)
    return {level: float(m.compute()["f1"]) for level, m in totals.items()}


# ------------------------------------------------------------- features
def run_features(config: dict[str, Any], evaluation: dict[str, Any], selector: dict[str, Any], device: torch.device,
                 output_dir: Path, *, max_crops: int | None = None) -> dict[str, Any]:
    localizer, localizer_config = open_localizer(config, "p30", device)
    check_sources(evaluation, localizer_config)
    h = harness(evaluation)
    summary: dict[str, Any] = {"localizer_sha256": sha256(localizer_checkpoint(config, "p30")),
                               "selector_config_sha256": config["selector_config_sha256"],
                               "crop_margin": int(config["crop_margin"]), "datasets": {}}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        dataset = open_dataset(spec, selector["training_crops"]["split"])
        items = training_items(dataset, spec, selector, localizer_config, h["size"], int(config["crop_margin"]))
        items = items[:max_crops] if max_crops is not None else items
        detector = open_detector(config, "original", name, device)
        records = records_for(dataset, spec, items, localizer, detector, h, device, "features")
        write_csv(output_dir / f"features_{name}.csv", records)
        summary["datasets"][name] = {"tiles": len(dataset.rows),
                                     "crops": {split: sum(1 for _, s in items if s == split) for split in SPLITS},
                                     "records": len(records), "detector_sha256": detector["sha256"]}
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------- train
def run_train(config: dict[str, Any], evaluation: dict[str, Any], selector: dict[str, Any], features_dir: Path,
              output_dir: Path) -> dict[str, Any]:
    features = json.loads((features_dir / "features_summary.json").read_text(encoding="utf-8"))
    if features["selector_config_sha256"] != config["selector_config_sha256"]:
        raise SystemExit("the features were extracted under a different selector recipe")
    summaries = {}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        bundle, summaries[name] = train_dataset(name, read_records(features_dir / f"features_{name}.csv"), selector)
        bundle.update(protocol="p30", localizer_sha256=features["localizer_sha256"],
                      config_sha256=config["selector_config_sha256"])
        torch.save(bundle, output_dir / f"selector_{name}.pth")
    payload = {"localizer_sha256": features["localizer_sha256"], "datasets": summaries}
    (output_dir / "train_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(render_calibration(summaries))
    return payload


# ------------------------------------------------------------- evaluate
def p27_bundle_dir(config: dict[str, Any]) -> Path:
    return pointed(config["p27_selector_bundles"]["pointer"]) / "bundles"


def components(config: dict[str, Any], names: Sequence[str], bundle_dir: Path) -> dict[str, Any]:
    """The sha256 of the P30 config and of every file the evaluation adds; the markers are keyed by their digest."""
    return {
        "p30_config_sha256": config["_sha256"],
        "localizers": {which: sha256(localizer_checkpoint(config, which)) for which in ("p30", "p27")},
        "selectors": {"p30": {name: sha256(bundle_dir / f"selector_{name}.pth") for name in names},
                      "p27": {name: sha256(p27_bundle_dir(config) / f"selector_{name}.pth") for name in names}},
        "p30_augmented": {name: sha256(pointed(detector_spec(config, "p30_augmented", name)["checkpoint_pointer"]))
                          for name in names},
    }


def digest_of(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def claim(split: str, digest: str, record: dict[str, Any], marker_dir: Path | None = None) -> Path:
    """Validation once per set of components; the test once, and only for components seen on validation."""
    marker_dir = MARKER_DIR if marker_dir is None else marker_dir
    marker_dir.mkdir(parents=True, exist_ok=True)
    if split == "test":
        if any(marker_dir.glob("test_*.json")):
            raise SystemExit("the P30 test split has been used already")
        if not (marker_dir / f"val_{digest}.json").is_file():
            raise SystemExit("these components have not been evaluated on validation; the test follows that step")
    marker = marker_dir / f"{split}_{digest}.json"
    if marker.exists():
        raise SystemExit(f"these components have met the {split} crops already ({marker})")
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return marker


def system_table(records: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    tau = math.inf if bundle["tau"] is None else float(bundle["tau"])
    probability = selector_probability(bundle, feature_matrix(records, bundle["feature_names"])).tolist()
    for record, p in zip(records, probability):
        record["selector_probability"] = p
    success = {str(level): float(np.mean([r["localizer_success"] in (True, "True") for r in records
                                          if int(r["level"]) == level]))
               for level in sorted({int(r["level"]) for r in records}) if level > 0}
    missing = [int(r["estimated_missing_px"]) for r in records]
    return {"tau": bundle["tau"], "f1": f1_table(records, probability, tau),
            "selection_rate": selection_rates(records, probability, tau), "localization_success": success,
            "estimated_missing_px": {"records_with_any": int(sum(m > 0 for m in missing)),
                                     "records": len(missing), "max": int(max(missing))}}


def run_evaluate(config: dict[str, Any], evaluation: dict[str, Any], split: str, bundle_dir: Path,
                 device: torch.device, output_dir: Path, *, max_crops: int | None = None) -> dict[str, Any]:
    evaluation = with_split(evaluation, split)
    h = harness(evaluation)
    margin = int(config["crop_margin"])
    names = [spec["name"] for spec in evaluation["datasets"]]
    p30_localizer, _ = open_localizer(config, "p30", device)
    p27_localizer, _ = open_localizer(config, "p27", device)
    p30_bundles, _ = load_bundles(bundle_dir, names)
    p27_bundles, _ = load_bundles(p27_bundle_dir(config), names)
    expected = {which: sha256(localizer_checkpoint(config, which)) for which in ("p30", "p27")}
    for name in names:
        if p30_bundles[name].get("protocol") != "p30" or p30_bundles[name]["localizer_sha256"] != expected["p30"]:
            raise SystemExit(f"{name}: the P30 selector was trained for another localizer or protocol")
        if p27_bundles[name]["localizer_sha256"] != expected["p27"]:
            raise SystemExit(f"{name}: the P27 selector does not belong to the P27 localizer")
    decision: dict[str, Any] = {"split": split, "crop_margin": margin, "datasets": {}}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        dataset = open_dataset(spec, split)
        crops = margin_crops(dataset, spec, h["size"], margin)
        crops = crops[:max_crops] if max_crops is not None else crops
        items = [(crop, split) for crop in crops]
        original = open_detector(config, "original", name, device)
        proposed = records_for(dataset, spec, items, p30_localizer, original, h, device, "proposed")
        reference = records_for(dataset, spec, items, p27_localizer, original, h, device, "reference")
        entry: dict[str, Any] = {"crops": len(crops), "detectors": {"original": original["sha256"]},
                                 "proposed": system_table(proposed, p30_bundles[name]),
                                 "reference_p27": system_table(reference, p27_bundles[name])}
        entry["original"] = {level: row["raw"] for level, row in entry["proposed"]["f1"].items()}
        del original
        for group in ("p29_augmented", "p30_augmented"):
            detector = open_detector(config, group, name, device)
            entry[group] = raw_scores(dataset, spec, crops, detector, h, device)
            entry["detectors"][group] = detector["sha256"]
            del detector
            if device.type == "cuda":
                torch.cuda.empty_cache()
        decision["datasets"][name] = entry
        write_csv(output_dir / f"records_proposed_{name}.csv", proposed)
        write_csv(output_dir / f"records_reference_{name}.csv", reference)
        print(render(name, entry), flush=True)
    return decision


def rows_of(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {"1 original detector, no correction": entry["original"],
            "2 P29 augmented detector": entry["p29_augmented"],
            "3 P30 augmented detector": entry["p30_augmented"],
            "4 proposed (localizer v2 + P30 selector)": {k: v["selector"] for k, v in entry["proposed"]["f1"].items()},
            "5 reference (localizer v1 + P27 selector)":
                {k: v["selector"] for k, v in entry["reference_p27"]["f1"].items()}}


def render(name: str, entry: dict[str, Any]) -> str:
    lines = [f"\n[{name}] {entry['crops']} crops; F1 % at clean / L1 / L2 / L3"]
    for label, values in rows_of(entry).items():
        lines.append(f"  {label:<44}" + " / ".join(f"{values[level] * 100:5.1f}" for level in LEVELS))
    for key in ("proposed", "reference_p27"):
        system = entry[key]
        success = system["localization_success"]
        missing = system["estimated_missing_px"]
        lines.append(f"  {key}: tau {system['tau']}; localization success L1-L3 "
                     + " / ".join(f"{success[level] * 100:.1f}" for level in LEVELS[1:])
                     + "; L3 " + ", ".join(f"{rule} {system['f1']['3'][rule] * 100:.1f}"
                                          for rule in ("oracle", "estimated", "gated", "selector"))
                     + f"; corrected L0/L3 {system['selection_rate']['0']['selector']:.3f}/"
                       f"{system['selection_rate']['3']['selector']:.3f}"
                     + f"; estimated view with uncovered pixels {missing['records_with_any']}/{missing['records']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "train", "evaluate"))
    parser.add_argument("--config", default="configs/p30_border_free.yaml")
    parser.add_argument("--features-dir", help="train: output of the features command")
    parser.add_argument("--bundle-dir", help="evaluate: output of the train command")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, evaluation, selector = load(Path(args.config))
    config["_sha256"] = sha256(Path(args.config))
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    if args.command == "features":
        output_dir.mkdir(parents=True, exist_ok=False)
        summary = run_features(config, evaluation, selector, device, output_dir)
        (output_dir / "features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    if args.command == "train":
        output_dir.mkdir(parents=True, exist_ok=False)
        run_train(config, evaluation, selector, Path(args.features_dir), output_dir)
        return

    bundle_dir = Path(args.bundle_dir)
    names = [spec["name"] for spec in evaluation["datasets"]]
    identity = components(config, names, bundle_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    marker = claim(args.split, digest_of(identity),
                   {"split": args.split, "output_dir": str(output_dir), "bundle_dir": str(bundle_dir), **identity})
    decision = run_evaluate(config, evaluation, args.split, bundle_dir, device, output_dir)
    decision.update(components=identity, marker=str(marker), test_split_touched=args.split == "test")
    (output_dir / f"p30_{args.split}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
