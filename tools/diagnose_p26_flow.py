"""Read-only diagnosis of DiffRegCD's registration head after P26 step 1
failed its self-check (main record 191.1). Descriptive only.

On self-pairs built exactly like DiffRegCD's training data (the perturbed
image is warpAffine of the reference with a known matrix, ground truth flow
M p - p on the reference grid, resized with align_corners=True), it reports
for both decodings and both input orders:
  the mean predicted displacement, the mean ground-truth displacement,
  the end-point error against the ground truth and against its negative,
  and the mean cosine between predicted and true vectors;
plus the predicted displacement on identical pairs (truth: zero).
Runs in the DiffRegCD environment; validation images only; no labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rs_cd.affine_crops import TileCache, select_crops
from rs_cd.affine_registration import AffineRanges, affine_parameters, apply, perturb, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_flow_registration import coarse_points
from rs_cd.robustness import _stable_seed


def truth_field(matrix: np.ndarray, coarse_height: int, coarse_width: int, size: int) -> np.ndarray:
    """Training ground truth on the coarse grid, normalized: (M p - p) * 2 / (size - 1)."""

    points = coarse_points(coarse_height, coarse_width, size, size)
    return ((apply(matrix, points) - points) * 2.0 / (size - 1)).reshape(coarse_height, coarse_width, 2)


def compare(predicted: np.ndarray, truth: np.ndarray, size: int) -> dict[str, float]:
    scale = (size - 1) / 2.0
    pred = predicted.reshape(-1, 2) * scale
    true = truth.reshape(-1, 2) * scale
    inside = (np.abs(truth.reshape(-1, 2)) <= 1.0).all(axis=1)  # the training loss masks overflow
    pred, true = pred[inside], true[inside]
    pred_norm = np.hypot(pred[:, 0], pred[:, 1])
    true_norm = np.hypot(true[:, 0], true[:, 1])
    dot = pred[:, 0] * true[:, 0] + pred[:, 1] * true[:, 1]
    cosine = dot / np.maximum(pred_norm * true_norm, 1e-9)
    return {
        "points": int(inside.sum()),
        "pred_mean_px": float(pred_norm.mean()),
        "true_mean_px": float(true_norm.mean()),
        "epe_px": float(np.hypot(pred[:, 0] - true[:, 0], pred[:, 1] - true[:, 1]).mean()),
        "epe_negated_px": float(np.hypot(pred[:, 0] + true[:, 0], pred[:, 1] + true[:, 1]).mean()),
        "mean_cosine": float(cosine.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p26_diffregcd_registration.yaml")
    parser.add_argument("--crops", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from tools.run_p26_diffregcd_registration import DiffRegCDRegistration, load_configs

    p26, p25 = load_configs(Path(args.config))
    external = p26["external"]
    root = Path(external["root"])
    perturbation = p25["perturbation"]
    size = int(perturbation["crop_size"])
    seed = int(p25["seed"])
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(float(perturbation["max_rotation_deg"]), float(low), float(high),
                          float(perturbation["max_translation_px"]), len(perturbation["levels"]))
    specs = {spec["name"]: spec for spec in p25["datasets"]}
    report: dict[str, Any] = {"note": "descriptive; self-pairs only; P26 verdict unchanged", "datasets": {}}
    for name in p26["datasets"]:
        spec = specs[name]
        head_spec = external["registration_heads"][name]
        adapter = DiffRegCDRegistration(
            root, config_json=head_spec["config_json"], backbone=root / external["backbone"]["path"],
            head=root / head_spec["path"], timestep=int(external["feature_timestep"]),
            level=int(external["feature_level"]),
        )
        dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                      normalization="unit", mask_threshold=int(spec["mask_threshold"]))
        crops = select_crops(dataset.rows, int(spec["tile_size"]), size,
                             int(p25["step1"]["max_crops_per_dataset"]))[: args.crops]
        cache = TileCache(dataset, int(spec["tile_size"]))
        entries = []
        for item in crops:
            reference, _, _ = cache.crop(item, size)
            matrix = perturbation_matrix(*affine_parameters(item["crop_id"], 3, ranges, seed), width=size, height=size)
            perturbed = perturb(reference, matrix)
            crop_seed = _stable_seed(seed, item["crop_id"], "p26", 0, "diffusion")
            normal = adapter.flows(perturbed, reference, crop_seed)       # A = perturbed, B = reference
            swapped = adapter.flows(reference, perturbed, crop_seed)      # A = reference, B = perturbed
            identical = adapter.flows(reference, reference, crop_seed)
            coarse_h, coarse_w = normal["loss"].shape[:2]
            truth = truth_field(matrix, coarse_h, coarse_w, size)
            entry = {"crop_id": item["crop_id"], "coarse_grid": [coarse_h, coarse_w]}
            for order, flows in (("A=perturbed", normal), ("A=reference", swapped)):
                for decoding in ("loss", "model"):
                    entry[f"{order}|{decoding}"] = compare(flows[decoding], truth, size)
            for decoding in ("loss", "model"):
                zero = identical[decoding].reshape(-1, 2) * (size - 1) / 2.0
                entry[f"identical|{decoding}|pred_mean_px"] = float(np.hypot(zero[:, 0], zero[:, 1]).mean())
            entries.append(entry)
            print(json.dumps(entry))
        report["datasets"][name] = entries
        del adapter
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, entries in report["datasets"].items():
        print(f"\n[{name}] mean over {len(entries)} self-pairs")
        keys = [key for key in entries[0] if "|" in key and not key.startswith("identical")]
        for key in keys:
            values = {metric: float(np.mean([entry[key][metric] for entry in entries]))
                      for metric in ("pred_mean_px", "true_mean_px", "epe_px", "epe_negated_px", "mean_cosine")}
            print(f"  {key:22s} " + "  ".join(f"{metric} {value:.2f}" for metric, value in values.items()))
        for decoding in ("loss", "model"):
            value = float(np.mean([entry[f"identical|{decoding}|pred_mean_px"] for entry in entries]))
            print(f"  identical pair, {decoding}: predicted mean displacement {value:.2f} px (truth 0)")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
