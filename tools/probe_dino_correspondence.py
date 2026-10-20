from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from rs_cd.correspondence import global_cosine_offset
from rs_cd.datasets import PairedChangeDataset
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, select_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe whether DINO features recover known T2 shifts."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-level", type=int, default=0)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--spatial-step", type=int, default=2)
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[2, 3],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    data = config["data"]
    dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=args.split,
        image_size=data.get("eval_crop_size"),
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if not hasattr(model, "_encode"):
        raise TypeError("Correspondence probe requires an encoded feature model")

    rows: list[dict[str, Any]] = []
    for severity in args.severities:
        shifted = CorruptedPairDataset(
            dataset,
            CorruptionSpec(
                name="shift",
                severity=int(severity),
                mode="t2",
                seed=3407,
            ),
            normalization=str(data.get("normalization", "imagenet")),
        )
        print(f"severity={severity} samples={len(shifted)}")
        for index in range(len(shifted)):
            sample = shifted[index]
            t1 = sample["image_t1"].unsqueeze(0).to(device)
            t2 = sample["image_t2"].unsqueeze(0).to(device)
            with torch.no_grad():
                paired_features = model._encode(
                    torch.cat((t1, t2), dim=0)
                )[args.feature_level]
                feature_t1 = paired_features[0]
                feature_t2 = paired_features[1]
            input_height, input_width = t1.shape[-2:]
            feature_height, feature_width = feature_t1.shape[-2:]
            stride_x = input_width / feature_width
            stride_y = input_height / feature_height
            unchanged = 1.0 - F.adaptive_max_pool2d(
                sample["mask"].unsqueeze(0).to(device),
                output_size=(feature_height, feature_width),
            )[0]
            valid = F.interpolate(
                sample["valid_mask"].unsqueeze(0).to(device),
                size=(feature_height, feature_width),
                mode="nearest",
            )[0]
            probe = global_cosine_offset(
                feature_t1,
                feature_t2,
                radius=args.search_radius,
                valid_mask=unchanged * valid,
                spatial_step=args.spatial_step,
            )
            target_dx = float(sample["shift_dx"]) / stride_x
            target_dy = float(sample["shift_dy"]) / stride_y
            error_pixels = math.sqrt(
                ((probe.dx - target_dx) * stride_x) ** 2
                + ((probe.dy - target_dy) * stride_y) ** 2
            )
            row = {
                "id": sample["id"],
                "severity": severity,
                "shift_dx_pixels": int(sample["shift_dx"]),
                "shift_dy_pixels": int(sample["shift_dy"]),
                "target_dx_feature": target_dx,
                "target_dy_feature": target_dy,
                "predicted_dx_feature": probe.dx,
                "predicted_dy_feature": probe.dy,
                "error_pixels": error_pixels,
                "best_score": probe.best_score,
                "score_margin": probe.margin,
            }
            rows.append(row)
            print(
                f"[{index + 1:03d}/{len(shifted):03d}] "
                f"id={sample['id']} target=({target_dx:.2f},{target_dy:.2f}) "
                f"pred=({probe.dx},{probe.dy}) epe={error_pixels:.2f}px"
            )

    summary_by_severity: dict[str, dict[str, float | int]] = {}
    for severity in args.severities:
        subset = [row for row in rows if row["severity"] == severity]
        errors = [float(row["error_pixels"]) for row in subset]
        summary_by_severity[str(severity)] = {
            "samples": len(subset),
            "mean_epe_pixels": sum(errors) / max(1, len(errors)),
            "max_epe_pixels": max(errors, default=0.0),
            "exact_offset_rate": sum(
                int(row["predicted_dx_feature"])
                == round(float(row["target_dx_feature"]))
                and int(row["predicted_dy_feature"])
                == round(float(row["target_dy_feature"]))
                for row in subset
            )
            / max(1, len(subset)),
            "within_4px_rate": sum(error <= 4.0 for error in errors)
            / max(1, len(errors)),
            "mean_score_margin": sum(
                float(row["score_margin"]) for row in subset
            )
            / max(1, len(subset)),
        }
    output = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "feature_level": args.feature_level,
        "search_radius": args.search_radius,
        "spatial_step": args.spatial_step,
        "summary_by_severity": summary_by_severity,
        "results": rows,
    }
    save_json(args.output, output)
    print(json.dumps(summary_by_severity, indent=2))
    print(f"Saved correspondence probe to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
