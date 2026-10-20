from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rs_cd.datasets import PairedChangeDataset
from rs_cd.engine import evaluate
from rs_cd.losses import BinaryChangeLoss
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate P7 task-aware input-level alignment."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--offset-decoding",
        choices=("soft", "hard"),
        default="hard",
    )
    parser.add_argument(
        "--alignment-gate",
        choices=(
            "none",
            "soft_nonzero",
            "hard_nonzero",
            "threshold_nonzero",
        ),
        default="threshold_nonzero",
    )
    parser.add_argument(
        "--zero-probability-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.zero_probability_threshold <= 1.0:
        raise ValueError("zero-probability-threshold must be in [0, 1]")

    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config: dict[str, Any] = checkpoint["config"]
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    config["model"]["registration_only"] = False
    config["model"]["freeze_registration"] = True
    config["model"]["offset_decoding"] = args.offset_decoding
    config["model"]["alignment_gate"] = args.alignment_gate
    config["model"]["zero_probability_threshold"] = float(
        args.zero_probability_threshold
    )
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
    if not callable(getattr(model, "estimate_registration", None)):
        raise TypeError(
            "Checkpoint does not contain a P7 input registration model"
        )
    model.eval()
    loss_config = config["loss"]
    criterion = BinaryChangeLoss(
        bce_weight=float(loss_config.get("bce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)

    specs = [
        CorruptionSpec(
            name="clean",
            severity=0,
            mode="common",
            seed=3407,
        ),
        *[
            CorruptionSpec(
                name="shift",
                severity=severity,
                mode="t2",
                seed=3407,
            )
            for severity in (1, 2, 3)
        ],
    ]
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        dataset = CorruptedPairDataset(
            base_dataset,
            spec,
            normalization=str(data.get("normalization", "imagenet")),
        )
        loader = _loader(
            dataset=dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(data.get("num_workers", 4)),
            seed=seed + index,
            device=device,
            prefetch_factor=int(data.get("prefetch_factor", 2)),
        )
        metrics = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            threshold=float(
                config["evaluation"].get("threshold", 0.5)
            ),
        )
        row = {
            "scenario": spec.key,
            "corruption": spec.name,
            "severity": spec.severity,
            "mode": spec.mode,
            **metrics,
        }
        rows.append(row)
        print(
            f"{spec.key:20s} F1={float(metrics['f1']):.6f} "
            f"IoU={float(metrics['iou']):.6f}"
        )
    output = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "offset_decoding": args.offset_decoding,
        "alignment_gate": args.alignment_gate,
        "zero_probability_threshold": float(
            args.zero_probability_threshold
        ),
        "results": rows,
    }
    save_json(args.output, output)
    print(
        json.dumps(
            {row["scenario"]: row["f1"] for row in rows},
            indent=2,
        )
    )
    print(f"Saved input alignment probe to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
