from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rs_cd.datasets import PairedChangeDataset
from rs_cd.engine import evaluate
from rs_cd.losses import BinaryChangeLoss
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen CD checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override the dataset root stored in the checkpoint config.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
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
    loader = _loader(
        dataset=dataset,
        batch_size=int(
            config["evaluation"].get(
                "batch_size", config["training"]["batch_size"]
            )
        ),
        shuffle=False,
        num_workers=int(data.get("num_workers", 4)),
        seed=seed,
        device=device,
        prefetch_factor=int(data.get("prefetch_factor", 2)),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    loss_config = config["loss"]
    criterion = BinaryChangeLoss(
        bce_weight=float(loss_config.get("bce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.checkpoint).resolve().parents[1] / "test_outputs" / args.split
    )
    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        prediction_dir=output_dir / "predictions",
    )
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)
    print(f"Saved evaluation outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
