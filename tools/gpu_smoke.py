from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rs_cd.losses import BinaryChangeLoss
from rs_cd.runtime import (
    build_model,
    create_grad_scaler,
    load_config,
    select_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure one model's GPU training-step memory and speed."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("gpu_smoke requires a CUDA device")
    config = load_config(args.config)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    image_size = args.image_size or int(config["data"]["train_crop_size"])
    amp_enabled = bool(config["training"].get("amp", True))

    model = build_model(config).to(device).train()
    loss_config = config["loss"]
    criterion = BinaryChangeLoss(
        bce_weight=float(loss_config.get("bce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    trainable_parameter_list = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameter_list:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameter_list,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = create_grad_scaler(device, enabled=amp_enabled)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["experiment"]["seed"]))
    t1 = torch.rand(
        batch_size,
        int(config["model"].get("in_channels", 3)),
        image_size,
        image_size,
        device=device,
        generator=generator,
    )
    t2 = torch.rand_like(t1)
    mask = (
        torch.rand(
            batch_size,
            1,
            image_size,
            image_size,
            device=device,
            generator=generator,
        )
        < 0.05
    ).float()
    smoke_batch = {
        "misregistration_target": ["t2"] * batch_size,
        "misregistration_dx": torch.full(
            (batch_size,), 4, dtype=torch.int64
        ),
        "misregistration_dy": torch.zeros(
            batch_size, dtype=torch.int64
        ),
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            outputs = model(t1, t2, targets=mask, batch=smoke_batch)
            loss, _ = criterion(outputs, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "batch_size": batch_size,
        "image_size": image_size,
        "steps": args.steps,
        "amp": amp_enabled,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        "images_per_second": args.steps * batch_size / elapsed,
        "final_loss": float(loss.detach()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
