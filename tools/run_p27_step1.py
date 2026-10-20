"""P27 E1: a frozen P27 localizer in the P25 step 1 harness, rotation fixed
at 0 (docs/P27_LOCALIZER_EVALUATION_PROTOCOL.md).

Crops, per-crop perturbations, success rule, summaries and verdicts are
those of P25 (configs/p27_eval_perturbation.yaml). The clean / misaligned
decision is the localizer's own probability. Each frozen checkpoint may meet
the validation crops once, and at most ``p27.max_frozen_versions`` distinct
checkpoints ever may; markers under reports/p27_localizer/step1/ enforce it.
Validation images only; masks feed the descriptive change-fraction column.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.affine_crops import TileCache, select_crops
from rs_cd.affine_registration import AffineRanges, dataset_summary, grid_points
from rs_cd.datasets import PairedChangeDataset
from rs_cd.localizer_training import TrainedLocalizer, load_localizer, step1_rows

MARKER_DIR = Path("reports/p27_localizer/step1/frozen")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("P27 evaluation config is not marked preregistered")
    if float(config["perturbation"]["max_rotation_deg"]) != 0.0:
        raise SystemExit("P27 version 1 is evaluated without rotation")
    if any(spec["split"] != "val" for spec in config["datasets"]):
        raise SystemExit("E1 reads validation splits only")
    return config


def claim(checkpoint_sha: str, config: dict[str, Any], marker_dir: Path, record: dict[str, Any]) -> Path:
    """One use per frozen checkpoint, and a cap on distinct checkpoints."""

    marker_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(marker_dir.glob("*.json"))
    marker = marker_dir / f"{checkpoint_sha}.json"
    if marker.exists():
        raise SystemExit(f"checkpoint {checkpoint_sha[:12]} already met the validation crops ({marker})")
    if len(existing) >= int(config["p27"]["max_frozen_versions"]):
        raise SystemExit(f"{len(existing)} frozen versions already evaluated; the protocol allows "
                         f"{config['p27']['max_frozen_versions']}")
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return marker


def run(config: dict[str, Any], localizer: TrainedLocalizer, output_dir: Path) -> dict[str, Any]:
    perturbation = config["perturbation"]
    settings = config["step1"]
    size = int(perturbation["crop_size"])
    seed = int(config["seed"])
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]),
                          len(perturbation["levels"]))
    levels = [int(level) for level in perturbation["levels"]]
    points = grid_points(size, size, int(settings["grid_points"]))
    decision: dict[str, Any] = {"gate": "p27_step1", "datasets": {}, "test_split_touched": False,
                                "labels_read": "validation masks, change_fraction column only (descriptive)"}
    all_rows = []
    for spec in config["datasets"]:
        dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                      normalization="unit", mask_threshold=int(spec["mask_threshold"]))
        crops = select_crops(dataset.rows, int(spec["tile_size"]), size, int(settings["max_crops_per_dataset"]))
        cache = TileCache(dataset, int(spec["tile_size"]))
        rows = []
        for index, item in enumerate(crops, start=1):
            t1, t2, mask = cache.crop(item, size)
            rows.extend(step1_rows(
                localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2,
                change_fraction=float((mask > int(spec["mask_threshold"])).mean()),
                ranges=ranges, levels=levels, seed=seed, points=points,
                tolerance=float(settings["success_ate_px"]),
                trigger_probability=float(config["p27"]["trigger_probability"]),
            ))
            if index % 100 == 0 or index == len(crops):
                print(f"[p27 step1 {spec['name']} {index}/{len(crops)}]")
        summary = dataset_summary(rows, settings)
        summary.update(crops=len(crops), tiles=len(dataset.rows))
        decision["datasets"][spec["name"]] = summary
        all_rows.extend(rows)
    decision["verdicts"] = {name: entry["verdict"] for name, entry in decision["datasets"].items()}
    with (output_dir / "step1_estimates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_eval_perturbation.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", required=True, help="the config.yaml saved with the training run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    checkpoint = Path(args.checkpoint)
    record = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "training_config": args.training_config, "training_config_sha256": sha256(Path(args.training_config)),
        "evaluation_config_sha256": sha256(Path(args.config)), "output_dir": args.output_dir,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    marker = claim(record["checkpoint_sha256"], config, MARKER_DIR, record)
    device = torch.device(args.device)
    localizer = TrainedLocalizer(load_localizer(str(checkpoint), device), device)
    decision = run(config, localizer, output_dir)
    decision["frozen"] = record
    decision["marker"] = str(marker)
    (output_dir / "step1_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision["verdicts"], indent=2))


if __name__ == "__main__":
    main()
