from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, Subset

from rs_cd.datasets import PairedChangeDataset
from rs_cd.engine import append_jsonl, evaluate, train_one_epoch
from rs_cd.losses import BinaryChangeLoss
from rs_cd.optimization import build_optimizer, build_scheduler
from rs_cd.runtime import (
    build_model,
    collect_system_info,
    create_grad_scaler,
    create_run_directory,
    load_config,
    save_json,
    seed_everything,
    seed_worker,
    select_device,
)


def _dataset(
    config: dict[str, Any], split: str, augment: bool
) -> PairedChangeDataset:
    data = config["data"]
    image_size = (
        data.get("train_crop_size", data.get("image_size"))
        if augment
        else data.get("eval_crop_size")
    )
    return PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=split,
        image_size=image_size,
        normalization=data.get("normalization", "imagenet"),
        augment=augment,
        random_flip=bool(data.get("train_flip", True)),
        random_rotate90=bool(data.get("train_rotate90", True)),
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
        positive_crop_probability=(
            float(data.get("positive_crop_probability", 0.0))
            if augment
            else 0.0
        ),
        positive_crop_min_pixels=int(
            data.get("positive_crop_min_pixels", 1)
        ),
        positive_crop_trials=int(data.get("positive_crop_trials", 8)),
        misregistration_probability=(
            float(data.get("misregistration_probability", 0.0))
            if augment
            else 0.0
        ),
        misregistration_pixels=tuple(
            int(value)
            for value in data.get(
                "misregistration_pixels",
                (2, 4, 8),
            )
        ),
        misregistration_target=str(
            data.get("misregistration_target", "t2")
        ),
        affine_misregistration_probability=(
            float(data.get("affine_misregistration_probability", 0.0))
            if augment
            else 0.0
        ),
        affine_max_translation_px=float(
            data.get("affine_max_translation_px", 25.0)
        ),
        affine_scale_range=tuple(
            float(value)
            for value in data.get("affine_scale_range", (0.80, 1.25))
        ),
        affine_levels=int(data.get("affine_levels", 3)),
        affine_context_margin=(
            data.get("affine_context_margin") if augment else None
        ),
    )


def _loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
    prefetch_factor: int = 2,
    samples_per_epoch: int | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if samples_per_epoch is not None:
        if not shuffle:
            raise ValueError("samples_per_epoch is only valid for shuffled loaders")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=int(samples_per_epoch),
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        **{
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "drop_last": False,
            "worker_init_fn": seed_worker,
            "generator": generator,
            "persistent_workers": num_workers > 0,
            **(
                {
                    "prefetch_factor": int(prefetch_factor),
                }
                if num_workers > 0
                else {}
            ),
        },
    )


def _append_registry(
    registry_path: Path, row: dict[str, Any]
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    exists = registry_path.is_file()
    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary CD baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment.seed without editing the YAML config.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override data.dataset_root while preserving all other config values.",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Use the first N training samples for both train and validation.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="Initialize model weights from a prior training checkpoint.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.initial_checkpoint is not None:
        config["training"]["initial_checkpoint"] = str(
            Path(args.initial_checkpoint).expanduser().resolve()
        )
    seed = int(config["experiment"]["seed"])
    training = config["training"]
    deterministic = bool(training.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(
            training.get("allow_tf32", True)
        )
        torch.backends.cudnn.allow_tf32 = bool(
            training.get("allow_tf32", True)
        )
    device = select_device(args.device)
    run_dir = create_run_directory(config)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "predictions").mkdir()

    with (run_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    system_info = collect_system_info(device)
    save_json(run_dir / "system_info.json", system_info)

    if args.overfit_samples > 0:
        data_config = config["data"]
        clean_train = PairedChangeDataset(
            dataset_root=data_config["dataset_root"],
            manifest_path=data_config["manifest"],
            split=str(data_config.get("train_split", "train")),
            image_size=data_config.get(
                "train_crop_size", data_config.get("image_size")
            ),
            normalization=data_config.get("normalization", "imagenet"),
            augment=False,
            mask_threshold=int(data_config.get("mask_threshold", 127)),
            cache_mode=str(data_config.get("cache_mode", "none")),
        )
        count = min(args.overfit_samples, len(clean_train))
        indices = list(range(count))
        train_dataset = Subset(clean_train, indices)
        val_dataset = Subset(clean_train, indices)
    else:
        train_dataset = _dataset(
            config,
            str(config["data"].get("train_split", "train")),
            augment=True,
        )
        val_dataset = _dataset(
            config,
            str(config["data"].get("val_split", "val")),
            augment=False,
        )

    batch_size = int(training["batch_size"])
    num_workers = int(config["data"].get("num_workers", 4))
    prefetch_factor = int(config["data"].get("prefetch_factor", 2))
    train_loader = _loader(
        train_dataset,
        batch_size,
        True,
        num_workers,
        seed,
        device,
        prefetch_factor,
        (
            None
            if args.overfit_samples > 0
            else training.get("samples_per_epoch")
        ),
    )
    val_loader = _loader(
        val_dataset,
        int(config["evaluation"].get("batch_size", batch_size)),
        False,
        num_workers,
        seed + 1,
        device,
        prefetch_factor,
    )

    model = build_model(config).to(device)
    initial_checkpoint_path = training.get("initial_checkpoint")
    if initial_checkpoint_path:
        initial_checkpoint = torch.load(
            initial_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        initial_state = initial_checkpoint.get(
            "model", initial_checkpoint
        )
        strict = bool(training.get("initial_checkpoint_strict", True))
        missing_keys, unexpected_keys = model.load_state_dict(
            initial_state,
            strict=strict,
        )
        print(f"Initialized model from {initial_checkpoint_path}")
        if not strict:
            print(
                f"initial_checkpoint_missing={len(missing_keys)} "
                f"unexpected={len(unexpected_keys)}"
            )
    loss_config = config["loss"]
    criterion = BinaryChangeLoss(
        bce_weight=float(loss_config.get("bce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    decoder_learning_rate = float(training["learning_rate"])
    backbone_learning_rate = training.get("backbone_learning_rate")
    decoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("backbone.")
    ]
    if not decoder_parameters and not backbone_parameters:
        raise ValueError("Model has no trainable parameters")
    parameter_groups: list[dict[str, Any]] = []
    if decoder_parameters:
        parameter_groups.append(
            {
                "params": decoder_parameters,
                "lr": decoder_learning_rate,
                "group_name": "task_head",
            }
        )
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(
                    backbone_learning_rate
                    if backbone_learning_rate is not None
                    else decoder_learning_rate
                ),
                "group_name": "backbone",
            }
        )
    # Defaults reproduce the frozen AdamW + cosine recipe; P21 configs name
    # a published optimizer and schedule instead (src/rs_cd/optimization.py).
    optimizer = build_optimizer(parameter_groups, training)
    print(
        f"optimizer={type(optimizer).__name__} "
        f"scheduler={training.get('scheduler', 'cosine')} "
        "optimizer_groups="
        + ", ".join(
            f"{group.get('group_name')}:{group['lr']:.3e}"
            for group in optimizer.param_groups
        )
    )
    epochs = int(training["epochs"])
    scheduler = build_scheduler(optimizer, training, epochs)
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = create_grad_scaler(device, enabled=amp_enabled)
    threshold = float(config["evaluation"].get("threshold", 0.5))
    validation_interval = int(
        config["evaluation"].get("validation_interval", 1)
    )
    checkpoint_interval = int(training.get("checkpoint_interval", 1))
    if validation_interval < 1 or checkpoint_interval < 1:
        raise ValueError(
            "validation_interval and checkpoint_interval must be positive"
        )
    print(
        f"train_samples={len(train_loader.sampler)} "
        f"steps_per_epoch={len(train_loader)} "
        f"validation_interval={validation_interval}"
    )

    best_f1 = -1.0
    best_epoch = -1
    history_path = run_dir / "metrics.jsonl"
    status = "failed"
    try:
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                grad_clip_norm=training.get("grad_clip_norm"),
                epoch=epoch,
                log_interval=int(training.get("log_interval", 20)),
            )
            should_validate = (
                epoch == 1
                or epoch % validation_interval == 0
                or epoch == epochs
            )
            val_metrics = (
                evaluate(model, val_loader, criterion, device, threshold)
                if should_validate
                else None
            )
            scheduler.step()
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rates": {
                    str(group.get("group_name", index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
                "train": train_metrics,
                "val": val_metrics,
            }
            append_jsonl(history_path, record)
            registration_progress = ""
            if "hard_clean_within_1px" in train_metrics:
                registration_progress = (
                    f" train_offset_epe="
                    f"{train_metrics['hard_offset_epe_pixels']:.3f}"
                    f" train_clean_pck1="
                    f"{train_metrics['hard_clean_within_1px']:.3f}"
                    f" train_shift_pck1="
                    f"{train_metrics['hard_shift_within_1px']:.3f}"
                )
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} "
                + registration_progress
                + " "
                + (
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f} "
                    f"val_iou={val_metrics['iou']:.4f}"
                    if val_metrics is not None
                    else "val=skipped"
                )
            )

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            }
            if (
                epoch % checkpoint_interval == 0
                or epoch == epochs
                or should_validate
            ):
                torch.save(
                    checkpoint, run_dir / "checkpoints" / "last.pth"
                )
                if bool(training.get("keep_epoch_checkpoints", False)):
                    torch.save(
                        checkpoint,
                        run_dir
                        / "checkpoints"
                        / f"epoch_{epoch:03d}.pth",
                    )
            if (
                val_metrics is not None
                and float(val_metrics["f1"]) > best_f1
            ):
                best_f1 = float(val_metrics["f1"])
                best_epoch = epoch
                torch.save(checkpoint, run_dir / "checkpoints" / "best.pth")

        best_checkpoint = torch.load(
            run_dir / "checkpoints" / "best.pth",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_checkpoint["model"])
        final_val = evaluate(
            model,
            val_loader,
            criterion,
            device,
            threshold,
            prediction_dir=run_dir / "predictions" / "val",
        )
        summary = {
            "status": "completed",
            "run_dir": str(run_dir.resolve()),
            "best_epoch": best_epoch,
            "best_f1": best_f1,
            "final_val": final_val,
            "overfit_samples": args.overfit_samples,
        }
        save_json(run_dir / "summary.json", summary)
        status = "completed"
    finally:
        registry = Path(config["experiment"].get("output_root", "outputs"))
        _append_registry(
            registry / "experiment_registry.csv",
            {
                "run_id": run_dir.name,
                "experiment": config["experiment"]["name"],
                "seed": seed,
                "model": config["model"]["name"],
                "dataset": config["data"]["dataset_root"],
                "status": status,
                "run_dir": str(run_dir.resolve()),
            },
        )


if __name__ == "__main__":
    main()
