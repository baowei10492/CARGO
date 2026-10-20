"""P27 F1 step (docs/P27_F1_AMENDMENT.md): the frozen P27 localizer in front of
frozen detectors (the DINOv3 rows, or P21 detectors under
docs/P27_PLUGIN_AMENDMENT.md), on every 256 px validation crop.

Perturbations, crops and localizer rows are those of the P27 evaluation
config (P25 without rotation). For each dataset the detector scores the raw,
oracle, estimated and gated paths on shared pixels; the verdict is
"positive_gain" when the gated path beats raw at the binding levels and costs
at most ``max_clean_drop`` F1 on clean crops. The run counts as this
checkpoint's single use of the validation crops (markers of run_p27_step1).
Validation labels are read; test splits are never opened.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_crops
from rs_cd.affine_f1 import (
    DETECTOR_PATHS,
    PATHS,
    add_counts,
    confusion_counts,
    f1_decision,
    gated_source,
    path_images,
    scoring_mask,
)
from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    AffineRanges,
    crop_origins,
    dataset_summary,
    grid_points,
    perturb,
    perturbation_matrix,
)
from rs_cd.change_density import DEFAULT_BINS, assign_bin
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.localizer_training import TrainedLocalizer, load_localizer, step1_rows
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p27_detectors import open_detector
from tools.run_p27_step1 import MARKER_DIR, claim, sha256
from tools.run_p27_step1 import load_config as load_evaluation_config


PLUGIN_MARKER_ROOT = Path("reports/p27_localizer/plugins")


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("P27 F1 config is not marked preregistered")
    evaluation_path = Path(config["evaluation_config"])
    digest = sha256(evaluation_path)
    if digest != config["evaluation_config_sha256"]:
        raise SystemExit(f"{evaluation_path} has sha256 {digest}, the F1 config expects "
                         f"{config['evaluation_config_sha256']}")
    evaluation = load_evaluation_config(evaluation_path)
    if "datasets" in config:
        # Plug-in rows may cover a subset of the evaluation datasets.
        wanted = list(config["datasets"])
        known = [spec["name"] for spec in evaluation["datasets"]]
        if not wanted or any(name not in known for name in wanted):
            raise SystemExit(f"datasets {wanted} are not a subset of {known}")
        evaluation["datasets"] = [spec for spec in evaluation["datasets"] if spec["name"] in wanted]
    names = [spec["name"] for spec in evaluation["datasets"]]
    if sorted(names) != sorted(config["detectors"]):
        raise SystemExit(f"detectors {sorted(config['detectors'])} do not match datasets {sorted(names)}")
    if config["decision"]["primary_path"] not in PATHS:
        raise SystemExit(f"unknown primary path {config['decision']['primary_path']}")
    return config, evaluation


def marker_dir(config: dict[str, Any]) -> Path:
    """The DINOv3 row keeps the E1 markers; each plug-in detector has its own."""

    namespace = config.get("marker_namespace")
    return MARKER_DIR if namespace is None else PLUGIN_MARKER_ROOT / str(namespace) / "frozen"


def open_detectors(config: dict[str, Any], device: torch.device) -> dict[str, dict[str, Any]]:
    manifest = config.get("p21_freeze_manifest")
    detectors = {}
    for name, spec in config["detectors"].items():
        model, run_config, digest, source = open_detector(
            spec, device, manifest_path=None if manifest is None else Path(manifest))
        loss = run_config.get("loss", {})
        detectors[name] = {
            "model": model,
            "threshold": float(run_config["evaluation"].get("threshold", 0.5)),
            "loss": {key: (None if loss.get(key) is None else float(loss[key]))
                     for key in ("bce_weight", "dice_weight", "pos_weight") if key in loss},
            "sha256": digest,
            "source": source,
        }
    return detectors


def all_crops(dataset: PairedChangeDataset, tile_size: int, size: int) -> list[dict[str, Any]]:
    total = len(dataset.rows) * len(crop_origins(tile_size, size))
    return select_crops(dataset.rows, tile_size, size, max(total, 1))


def _metrics(threshold: float, keys) -> dict[str, dict[str, BinaryChangeMetrics]]:
    return {key: {path: BinaryChangeMetrics(threshold=threshold) for path in PATHS} for key in keys}


@torch.no_grad()
def evaluate_dataset(
    spec: dict[str, Any],
    evaluation: dict[str, Any],
    localizer: TrainedLocalizer,
    detector: torch.nn.Module,
    threshold: float,
    device: torch.device,
    *,
    max_crops: int | None = None,
) -> dict[str, Any]:
    perturbation = evaluation["perturbation"]
    settings = evaluation["step1"]
    size = int(perturbation["crop_size"])
    seed = int(evaluation["seed"])
    levels = [int(level) for level in perturbation["levels"]]
    all_levels = [CLEAN_LEVEL, *levels]
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels))
    points = grid_points(size, size, int(settings["grid_points"]))
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    crops = all_crops(dataset, int(spec["tile_size"]), size)
    if max_crops is not None:
        crops = crops[:max_crops]
    cache = TileCache(dataset, int(spec["tile_size"]))
    level_keys = [str(level) for level in all_levels]
    totals = _metrics(threshold, level_keys)
    by_bin = {index: _metrics(threshold, level_keys) for index in range(len(DEFAULT_BINS))}
    bin_crops = {index: 0 for index in range(len(DEFAULT_BINS))}
    localizer_rows, count_rows = [], []
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        truth_mask = mask > int(spec["mask_threshold"])
        change_fraction = float(truth_mask.mean())
        rows = step1_rows(
            localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2,
            change_fraction=change_fraction, ranges=ranges, levels=levels, seed=seed, points=points,
            tolerance=float(settings["success_ate_px"]),
            trigger_probability=float(evaluation["p27"]["trigger_probability"]),
        )
        localizer_rows.extend(rows)
        truths = {int(row["level"]): _truth(row, size) for row in rows}
        valid = scoring_mask([truths[level] for level in levels], size)
        valid_pixels = int(valid.sum())
        images, sources = [], []
        for row in rows:
            level = int(row["level"])
            perturbed = t2 if level == CLEAN_LEVEL else perturb(t2, truths[level])
            estimate = np.array([[row["a11"], row["a12"], row["a13"]], [row["a21"], row["a22"], row["a23"]]])
            for path, image in path_images(perturbed, truths[level], estimate).items():
                images.append(normalize(image))
                sources.append((level, path))
        t2_batch = torch.stack(images).to(device)
        t1_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
        logits = detector(t1_batch, t2_batch)["change_logits"]
        target = torch.from_numpy(truth_mask.astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        counts = dict(zip(sources, confusion_counts(logits, target, keep, threshold)))
        crop_bin = assign_bin(change_fraction, DEFAULT_BINS)
        bin_crops[crop_bin] += 1
        for row in rows:
            level = int(row["level"])
            chosen = {path: counts[(level, path)] for path in DETECTOR_PATHS}
            chosen["estimated_gated"] = counts[(level, gated_source(bool(row["trigger"])))]
            for path, values in chosen.items():
                add_counts(totals[str(level)][path], values)
                add_counts(by_bin[crop_bin][str(level)][path], values)
                count_rows.append({"dataset": spec["name"], "crop_id": item["crop_id"], "level": level,
                                   "path": path, "tp": values[0], "fp": values[1], "fn": values[2],
                                   "tn": values[3], "trigger": bool(row["trigger"]),
                                   "change_fraction": change_fraction, "valid_pixels": valid_pixels})
        if index % 200 == 0 or index == len(crops):
            print(f"[p27 f1 {spec['name']} {index}/{len(crops)}]", flush=True)
    table = {key: {path: totals[key][path].compute() for path in PATHS} for key in level_keys}
    bins = []
    for bin_index, (low_edge, high_edge) in enumerate(DEFAULT_BINS):
        bins.append({"range": [low_edge, high_edge], "crops": bin_crops[bin_index],
                     "f1": {key: {path: by_bin[bin_index][key][path].compute()["f1"] for path in PATHS}
                            for key in level_keys}})
    return {
        "crops": len(crops),
        "tiles": len(dataset.rows),
        "table": table,
        "f1": {key: {path: float(values["f1"]) for path, values in by_path.items()} for key, by_path in table.items()},
        "change_bins": bins,
        "localizer": dataset_summary(localizer_rows, settings),
        "localizer_rows": localizer_rows,
        "count_rows": count_rows,
    }


def _truth(row: dict[str, Any], size: int) -> np.ndarray:
    return perturbation_matrix(float(row["angle"]), float(row["scale"]), float(row["tx"]), float(row["ty"]),
                               width=size, height=size)


def run(config: dict[str, Any], evaluation: dict[str, Any], localizer: TrainedLocalizer, detectors: dict[str, Any],
        device: torch.device, output_dir: Path, *, max_crops: int | None = None) -> dict[str, Any]:
    decision_spec = config["decision"]
    levels = [int(level) for level in evaluation["perturbation"]["levels"]]
    decision: dict[str, Any] = {"gate": "p27_f1", "datasets": {}, "test_split_touched": False}
    localizer_rows, count_rows, table_rows = [], [], []
    for spec in evaluation["datasets"]:
        name = spec["name"]
        detector, threshold, digest = detectors[name]
        result = evaluate_dataset(spec, evaluation, localizer, detector, threshold, device, max_crops=max_crops)
        verdict = f1_decision(
            result["f1"], levels=levels, primary=decision_spec["primary_path"],
            binding_levels=[int(level) for level in decision_spec["binding_levels"]],
            min_gain=float(decision_spec["min_gain"]), max_clean_drop=float(decision_spec["max_clean_drop"]),
            min_clean_raw_f1=float(config["instrument"]["min_clean_raw_f1"]),
        )
        localizer_summary = result["localizer"]
        decision["datasets"][name] = {
            **verdict,
            "f1": result["f1"],
            "crops": result["crops"],
            "tiles": result["tiles"],
            "detector": {**config["detectors"][name], "sha256": digest, "threshold": threshold},
            "trigger_rate": {key: value["trigger_rate"] for key, value in localizer_summary["levels"].items()},
            "localizer_success": {key: value.get("success_rate") for key, value in localizer_summary["levels"].items()},
            "change_bins": result["change_bins"],
        }
        localizer_rows.extend(result["localizer_rows"])
        count_rows.extend(result["count_rows"])
        table_rows.extend({"dataset": name, "level": key, "path": path, **values}
                          for key, by_path in result["table"].items() for path, values in by_path.items())
    decision["verdicts"] = {name: entry["verdict"] for name, entry in decision["datasets"].items()}
    for filename, rows in (("f1_table.csv", table_rows), ("localizer_rows.csv", localizer_rows),
                           ("crop_counts.csv", count_rows)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return decision


def render(decision: dict[str, Any]) -> str:
    fmt = lambda value: "-" if value is None else f"{value:.4f}"
    lines = [f"verdicts: {decision['verdicts']}"]
    for name, entry in decision["datasets"].items():
        lines.append(f"\n[{name}] {entry['verdict']}  ({entry['crops']} crops from {entry['tiles']} tiles)")
        lines.append("  level  raw     oracle  estim.  gated   trigger  loc.success")
        for key, row in entry["f1"].items():
            lines.append(f"  {key:>5}  {fmt(row['raw'])}  {fmt(row['oracle'])}  {fmt(row['estimated'])}  "
                         f"{fmt(row['estimated_gated'])}  {fmt(entry['trigger_rate'].get(key))}   "
                         f"{fmt(entry['localizer_success'].get(key))}")
        lines.append("  level  harm     ceiling  gain     recovery")
        for key, row in entry["levels"].items():
            lines.append(f"  {key:>5}  {row['harm']:+.4f}  {row['ceiling']:+.4f}  {row['gain']:+.4f}  "
                         f"{fmt(row['recovery'])}")
        lines.append(f"  clean drop {entry['clean_drop']:+.4f} (ungated {entry['clean_drop_estimated']:+.4f}); "
                     f"checks {entry['checks']}; instrument passed {entry['instrument']['passed']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_f1_gate.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", required=True, help="the config.yaml saved with the training run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true",
                        help="load detectors, localizer and datasets, print their sources, and stop")
    args = parser.parse_args()
    config, evaluation = load_config(Path(args.config))
    device = torch.device(args.device)

    # Everything that can fail without looking at a validation crop is done
    # before the one-use marker is written.
    checkpoint = Path(args.checkpoint)
    opened = open_detectors(config, device)
    detectors = {name: (entry["model"], entry["threshold"], entry["sha256"]) for name, entry in opened.items()}
    digests = {name: entry["sha256"] for name, entry in opened.items()}
    for spec in evaluation["datasets"]:
        if not PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                   normalization="unit", mask_threshold=int(spec["mask_threshold"])).rows:
            raise SystemExit(f"{spec['name']}: no {spec['split']} rows")
    localizer = TrainedLocalizer(load_localizer(str(checkpoint), device), device)
    if args.check_only:
        print(json.dumps({"localizer_sha256": sha256(checkpoint),
                          "detectors": {name: {"sha256": entry["sha256"], **entry["source"]}
                                        for name, entry in opened.items()},
                          "marker_dir": str(marker_dir(config))}, indent=2))
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    record = {
        "step": "p27_f1",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "training_config": args.training_config, "training_config_sha256": sha256(Path(args.training_config)),
        "f1_config_sha256": sha256(Path(args.config)),
        "evaluation_config_sha256": config["evaluation_config_sha256"],
        "detector_sha256": digests, "detector_sources": {name: entry["source"] for name, entry in opened.items()},
        "output_dir": args.output_dir,
    }
    marker = claim(record["checkpoint_sha256"], evaluation, marker_dir(config), record)
    start = time.perf_counter()
    decision = run(config, evaluation, localizer, detectors, device, output_dir)
    decision.update(frozen=record, marker=str(marker), seconds=time.perf_counter() - start)
    (output_dir / "f1_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(render(decision))


if __name__ == "__main__":
    main()
