from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rs_cd.datasets import PairedChangeDataset
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a P5c global registration checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
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
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    data = config["data"]
    base_dataset = PairedChangeDataset(
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

    rows: list[dict[str, Any]] = []
    for severity in args.severities:
        corruption_name = "clean" if int(severity) == 0 else "shift"
        dataset = CorruptedPairDataset(
            base_dataset,
            CorruptionSpec(
                name=corruption_name,
                severity=int(severity),
                mode="common" if int(severity) == 0 else "t2",
                seed=3407,
            ),
            normalization=str(data.get("normalization", "imagenet")),
        )
        loader = _loader(
            dataset=dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(data.get("num_workers", 4)),
            seed=seed + int(severity),
            device=device,
            prefetch_factor=int(data.get("prefetch_factor", 2)),
        )
        for batch in loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            estimate = getattr(model, "estimate_registration", None)
            outputs = (
                estimate(t1, t2)
                if callable(estimate)
                else model(t1, t2)
            )
            predicted = outputs["predicted_offset_pixels"]
            predicted_hard = outputs["predicted_hard_offset_pixels"]
            if not isinstance(predicted, torch.Tensor):
                raise TypeError("Model did not return predicted offsets")
            if not isinstance(predicted_hard, torch.Tensor):
                raise TypeError("Model did not return hard predicted offsets")
            target = torch.stack(
                (
                    torch.as_tensor(batch["shift_dx"]),
                    torch.as_tensor(batch["shift_dy"]),
                ),
                dim=1,
            ).to(predicted)
            errors = torch.linalg.vector_norm(predicted - target, dim=1)
            hard_errors = torch.linalg.vector_norm(
                predicted_hard - target,
                dim=1,
            )
            confidence = outputs["registration_confidence"]
            margin = outputs["registration_margin"]
            zero_probability = outputs.get(
                "registration_zero_probability"
            )
            for index, sample_id in enumerate(batch["id"]):
                row = {
                        "id": str(sample_id),
                        "severity": int(severity),
                        "target_dx_pixels": float(target[index, 0]),
                        "target_dy_pixels": float(target[index, 1]),
                        "predicted_dx_pixels": float(predicted[index, 0]),
                        "predicted_dy_pixels": float(predicted[index, 1]),
                        "epe_pixels": float(errors[index]),
                        "hard_predicted_dx_pixels": float(
                            predicted_hard[index, 0]
                        ),
                        "hard_predicted_dy_pixels": float(
                            predicted_hard[index, 1]
                        ),
                        "hard_epe_pixels": float(hard_errors[index]),
                        "confidence": float(confidence[index]),
                        "score_margin": float(margin[index]),
                    }
                if isinstance(zero_probability, torch.Tensor):
                    row["zero_probability"] = float(
                        zero_probability[index]
                    )
                rows.append(row)

    summary_by_severity: dict[str, dict[str, float | int]] = {}
    for severity in args.severities:
        subset = [row for row in rows if row["severity"] == severity]
        errors = [float(row["epe_pixels"]) for row in subset]
        hard_errors = [
            float(row["hard_epe_pixels"]) for row in subset
        ]
        summary_by_severity[str(severity)] = {
            "samples": len(subset),
            "mean_epe_pixels": sum(errors) / max(1, len(errors)),
            "max_epe_pixels": max(errors, default=0.0),
            "mean_hard_epe_pixels": sum(hard_errors)
            / max(1, len(hard_errors)),
            "max_hard_epe_pixels": max(hard_errors, default=0.0),
            "within_1px_rate": sum(error <= 1.0 for error in errors)
            / max(1, len(errors)),
            "within_2px_rate": sum(error <= 2.0 for error in errors)
            / max(1, len(errors)),
            "within_4px_rate": sum(error <= 4.0 for error in errors)
            / max(1, len(errors)),
            "hard_within_2px_rate": sum(
                error <= 2.0 for error in hard_errors
            )
            / max(1, len(hard_errors)),
            "hard_within_1px_rate": sum(
                error <= 1.0 for error in hard_errors
            )
            / max(1, len(hard_errors)),
            "hard_within_4px_rate": sum(
                error <= 4.0 for error in hard_errors
            )
            / max(1, len(hard_errors)),
            "mean_confidence": sum(
                float(row["confidence"]) for row in subset
            )
            / max(1, len(subset)),
            "mean_score_margin": sum(
                float(row["score_margin"]) for row in subset
            )
            / max(1, len(subset)),
            **(
                {
                    "mean_zero_probability": sum(
                        float(row["zero_probability"])
                        for row in subset
                    )
                    / max(1, len(subset))
                }
                if subset and "zero_probability" in subset[0]
                else {}
            ),
        }
    output = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "summary_by_severity": summary_by_severity,
        "results": rows,
    }
    save_json(args.output, output)
    print(json.dumps(summary_by_severity, indent=2, ensure_ascii=False))
    print(f"Saved registration evaluation to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
