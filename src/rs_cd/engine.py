from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import BinaryChangeLoss
from .metrics import BinaryChangeMetrics


def _move_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["image_t1"].to(device, non_blocking=True),
        batch["image_t2"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BinaryChangeLoss,
    device: torch.device,
    scaler: Any,
    amp_enabled: bool,
    grad_clip_norm: float | None,
    epoch: int,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    started = time.perf_counter()
    iterator = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(iterator, start=1):
        t1, t2, mask = _move_batch(batch, device)
        # Present only with the P29 affine misregistration augmentation.
        valid = batch.get("valid_mask")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(t1, t2, targets=mask, batch=batch)
            if isinstance(valid, torch.Tensor):
                loss, components = criterion(
                    outputs, mask, valid=valid.to(device, non_blocking=True)
                )
            else:
                loss, components = criterion(outputs, mask)
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(t1.shape[0])
        sample_count += batch_size
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        if step % max(1, log_interval) == 0:
            iterator.set_postfix(loss=f"{components['loss']:.4f}")

    elapsed = time.perf_counter() - started
    result = {key: value / max(1, sample_count) for key, value in totals.items()}
    result["seconds"] = elapsed
    result["samples_per_second"] = sample_count / max(elapsed, 1e-12)
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BinaryChangeLoss,
    device: torch.device,
    threshold: float,
    prediction_dir: str | Path | None = None,
) -> dict[str, float | int]:
    model.eval()
    metrics = BinaryChangeMetrics(threshold=threshold)
    total_loss = 0.0
    sample_count = 0
    output_dir = Path(prediction_dir) if prediction_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="evaluate", leave=False):
        t1, t2, mask = _move_batch(batch, device)
        outputs = model(t1, t2, targets=mask, batch=batch)
        logits = outputs["change_logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model did not return Tensor change_logits")
        loss, _ = criterion(outputs, mask)
        batch_size = int(t1.shape[0])
        sample_count += batch_size
        total_loss += float(loss) * batch_size
        valid_mask = batch.get("valid_mask")
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.to(device, non_blocking=True)
        else:
            valid_mask = None
        metrics.update(
            logits,
            mask,
            from_logits=True,
            valid_mask=valid_mask,
        )

        if output_dir is not None:
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            for index, sample_id in enumerate(batch["id"]):
                image = np.clip(probabilities[index, 0] * 255.0, 0, 255).astype(
                    np.uint8
                )
                Image.fromarray(image, mode="L").save(output_dir / f"{sample_id}.png")

    result = metrics.compute()
    result["loss"] = total_loss / max(1, sample_count)
    result["samples"] = sample_count
    return result


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
