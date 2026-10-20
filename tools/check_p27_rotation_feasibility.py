"""P27 development check: untrained coarse matching on frozen DINOv3 tokens
under rotation (configs/p27_rotation_check.yaml).

For each P25 validation crop, the P25 level-3 translation and scale are kept
and the rotation magnitude is set to 0 / 10 / 20 / 30 degrees (the crop's own
sign). Two pairs per setting: T1 against its perturbed copy (self) and T1
against perturbed T2 (real). Mutual nearest neighbours of stride-8 and
stride-16 tokens feed a RANSAC similarity fit; the grid transfer error is
compared with 8 and 16 px. No training, no labels, validation images only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_crops, select_evenly
from rs_cd.affine_registration import (
    AffineRanges,
    affine_parameters,
    fit_similarity,
    grid_points,
    perturb,
    perturbation_matrix,
    transfer_error,
)
from rs_cd.coarse_matching import correspondences, mutual_nearest, token_centers
from rs_cd.datasets import PairedChangeDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("rotation check config is not marked preregistered")
    p25_path = Path(config["p25_config"])
    digest = hashlib.sha256(p25_path.read_bytes()).hexdigest()
    if digest != config["p25_config_sha256"]:
        raise SystemExit(f"{p25_path} has sha256 {digest}, the check expects {config['p25_config_sha256']}")
    p25 = yaml.safe_load(p25_path.read_text(encoding="utf-8"))
    if any(spec["split"] != "val" for spec in p25["datasets"]):
        raise SystemExit("the check reads validation splits only")
    return config, p25


def build_backbone(config: dict[str, Any], device: torch.device) -> tuple[Callable, list[int]]:
    import timm

    spec = config["backbone"]
    model = timm.create_model(
        f"local-dir:{spec['checkpoint_dir']}", pretrained=True, features_only=True,
        out_indices=tuple(int(index) for index in spec["feature_indices"]),
    ).to(device).eval()
    strides = [int(value) for value in model.feature_info.reduction()]

    @torch.no_grad()
    def features(images: np.ndarray) -> list[torch.Tensor]:
        batch = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().to(device) / 255.0
        batch = (batch - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        return list(model(batch))

    return features, strides


def rotated(params: tuple[float, float, float, float], magnitude: float) -> tuple[float, float, float, float]:
    angle, scale, tx, ty = params
    sign = -1.0 if angle < 0 else 1.0
    return sign * float(magnitude), scale, tx, ty


def summarize(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["pair"], row["stride"], row["rotation"]), []).append(row)
    summary = {}
    for (dataset, pair, stride, rotation), members in sorted(groups.items()):
        errors = [row["ate"] for row in members]
        entry = {
            "n": len(members),
            "median_ate": statistics.median(errors),
            "median_matches": statistics.median(row["matches"] for row in members),
            "converged_rate": sum(row["converged"] for row in members) / len(members),
        }
        for threshold in thresholds:
            entry[f"success_{threshold:g}"] = sum(error <= threshold for error in errors) / len(members)
        summary[f"{dataset}|{pair}|s{stride}|r{rotation:g}"] = entry
    return summary


def decide(summary: dict[str, Any], datasets: list[str], rotations: list[float], rule: dict[str, Any]) -> dict[str, Any]:
    key = f"success_{float(rule['success_px']):g}"

    def mean_success(rotation: float) -> float:
        values = [summary[f"{name}|{rule['pair']}|s{int(rule['stride'])}|r{rotation:g}"][key] for name in datasets]
        return sum(values) / len(values)

    base = mean_success(min(rotations))
    top = mean_success(max(rotations))
    retention = top / base if base > 0 else 0.0
    ok = base > 0 and retention >= float(rule["retention_min"])
    return {
        "success_at_min_rotation": base,
        "success_at_max_rotation": top,
        "retention": retention,
        "retention_min": float(rule["retention_min"]),
        "rotation_ok": ok,
        "version1": "similarity with rotation" if ok else "translation + scale (rotation dropped)",
    }


def run(config: dict[str, Any], p25: dict[str, Any], features: Callable, strides: list[int]) -> dict[str, Any]:
    perturbation = p25["perturbation"]
    size = int(perturbation["crop_size"])
    seed = int(p25["seed"])
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(float(perturbation["max_rotation_deg"]), float(low), float(high),
                          float(perturbation["max_translation_px"]), len(perturbation["levels"]))
    top_level = int(perturbation["levels"][-1])
    points = grid_points(size, size, int(config["grid_points"]))
    fit = config["fit"]
    rotations = [float(value) for value in config["rotations_deg"]]
    rows: list[dict[str, Any]] = []
    names = []
    for spec in p25["datasets"]:
        names.append(spec["name"])
        dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                      normalization="unit", mask_threshold=int(spec["mask_threshold"]))
        crops = select_crops(dataset.rows, int(spec["tile_size"]), size, int(p25["step1"]["max_crops_per_dataset"]))
        crops = select_evenly(crops, int(config["crops_per_dataset"]))
        cache = TileCache(dataset, int(spec["tile_size"]))
        for index, item in enumerate(crops, start=1):
            t1, t2, _ = cache.crop(item, size)
            base = affine_parameters(item["crop_id"], top_level, ranges, seed)
            for magnitude in rotations:
                params = rotated(base, magnitude)
                matrix = perturbation_matrix(*params, width=size, height=size)
                pairs = {"self": perturb(t1, matrix), "real": perturb(t2, matrix)}
                stacked = np.stack([t1, pairs["self"], pairs["real"]])
                levels = features(stacked)
                for stride, level in zip(strides, levels):
                    centers = token_centers(level.shape[-2], level.shape[-1], stride)
                    for position, pair in ((1, "self"), (2, "real")):
                        ref_idx, tgt_idx, _ = mutual_nearest(level[0], level[position])
                        source, target = correspondences(ref_idx, tgt_idx, centers)
                        estimate = fit_similarity(
                            source, target, reprojection_px=float(fit["reprojection_factor"]) * stride,
                            max_iters=int(fit["max_iters"]), confidence=float(fit["confidence"]),
                            refine_iters=int(fit["refine_iters"]), min_inliers=int(fit["min_inliers"]),
                            seed=seed,
                        )
                        error = (transfer_error(estimate.matrix, matrix, points, width=size, height=size)
                                 if estimate.converged else None)
                        rows.append({
                            "dataset": spec["name"], "crop_id": item["crop_id"], "pair": pair,
                            "stride": stride, "rotation": magnitude, "angle": params[0],
                            "scale": params[1], "tx": params[2], "ty": params[3],
                            "matches": len(ref_idx), "converged": bool(estimate.converged),
                            "inliers": estimate.inliers,
                            "ate": math.inf if error is None else error,
                        })
            if index % 25 == 0 or index == len(crops):
                print(f"[rotation check {spec['name']} {index}/{len(crops)}]")
    thresholds = [float(value) for value in config["success_px"]]
    summary = summarize(rows, thresholds)
    decision = decide(summary, names, rotations, config["decision"])
    return {"purpose": config["purpose"], "strides": strides, "summary": summary,
            "decision": decision, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_rotation_check.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, p25 = load_config(Path(args.config))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(config["seed"]))
    features, strides = build_backbone(config, torch.device(args.device))
    result = run(config, p25, features, strides)
    rows = result.pop("rows")
    with (output / "rotation_check_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "rotation_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nstride / pair / rotation: success@8, success@16, median ATE, median matches")
    for key, entry in result["summary"].items():
        print(f"  {key:32s} {entry['success_8']:.2f}  {entry['success_16']:.2f}  "
              f"{entry['median_ate']:.1f}  {entry['median_matches']:.0f}")
    print("\ndecision:", json.dumps(result["decision"]))


if __name__ == "__main__":
    main()
