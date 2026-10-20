"""Train the P27 localizer (docs/P27_OWN_LOCALIZER_DESIGN.md; P30 version 2 with
border-free targets and hold-out, docs/P30_BORDER_FREE_PROTOCOL.md).

Training splits only; a hashed 5% of training tiles is the development
hold-out, evaluated with the P25 step 1 rows (rotation fixed at 0). The best
checkpoint by hold-out success at the binding levels and the last checkpoint
are saved (trainable weights only; the frozen backbone is rebuilt on load).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from rs_cd.affine_crops import TileCache, cut_border_free, select_crops, select_evenly, select_margin_crops
from rs_cd.affine_registration import AffineRanges, dataset_summary, grid_points
from rs_cd.datasets.localizer_pairs import (
    LocalizerPairDataset,
    PerturbationMixture,
    Photometric,
    SyntheticChange,
    holdout_rows,
    sources_from_config,
)
from rs_cd.localizer_training import (
    TrainedLocalizer,
    batched_forward,
    localizer_loss,
    selection_score,
    step1_rows,
)
from rs_cd.models.change_aware_similarity_localizer import build_localizer
from rs_cd.similarity_geometry import grid


class RowSubset:
    """PairedChangeDataset restricted to some rows (for TileCache)."""

    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self.dataset = dataset
        self.rows = [dataset.rows[i] for i in indices]

    def _load_arrays(self, row):
        return self.dataset._load_arrays(row)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(config: dict[str, Any], total_samples: int) -> LocalizerPairDataset:
    data = config["data"]
    mixture = dict(data["perturbation"])
    mixture["large_scale"] = tuple(mixture["large_scale"])
    return LocalizerPairDataset(
        sources_from_config(config["sources"]),
        samples_per_epoch=total_samples,
        crop_size=int(data["crop_size"]),
        holdout_fraction=float(config["holdout"]["fraction"]),
        seed=int(config["seed"]),
        mixture=PerturbationMixture(**mixture),
        synthetic=SyntheticChange(**data["synthetic_change"]),
        jitter=Photometric(**{k: tuple(v) if isinstance(v, list) else v for k, v in data["photometric"].items()}),
        border_free_margin=data.get("border_free_margin"),
    )


def build_holdout(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Hold-out crops; with holdout.border_free_margin (P30) they keep that margin to the tile
    edge and carry the surrounding T2 region, so their perturbed T2 has no zero border."""
    size = int(config["data"]["crop_size"])
    margin = config["holdout"].get("border_free_margin")
    holdout = {}
    for source in sources_from_config(config["sources"]):
        dataset, indices = holdout_rows(source, float(config["holdout"]["fraction"]), int(config["seed"]))
        subset = RowSubset(dataset, indices)
        cap = int(config["holdout"]["crops_per_source"])
        if margin is None:
            crops = select_crops(subset.rows, source.tile_size, size, cap)
        else:
            crops = select_evenly(select_margin_crops(subset.rows, source.tile_size, size, int(margin)), cap)
        cache = TileCache(subset, source.tile_size)
        items = []
        for item in crops:
            t1, t2, mask = cache.crop(item, size)
            entry = {"crop_id": item["crop_id"], "t1": t1, "t2": t2,
                     "change_fraction": float((mask > source.mask_threshold).mean())}
            if margin is not None:
                top, left, m = item["top"], item["left"], int(margin)
                entry["region"] = cache.arrays[1][top - m:top + size + m, left - m:left + size + m].copy()
                entry["margin"] = m
            items.append(entry)
        holdout[source.name] = items
        print(f"[holdout] {source.name}: {len(indices)} tiles, {len(items)} crops")
    return holdout


def evaluate(model: torch.nn.Module, holdout: dict[str, list[dict[str, Any]]], config: dict[str, Any],
             device: torch.device) -> dict[str, Any]:
    spec = config["holdout"]["evaluation"]
    size = int(config["data"]["crop_size"])
    ranges = AffineRanges(0.0, float(spec["scale_range"][0]), float(spec["scale_range"][1]),
                          float(spec["max_translation_px"]), len(spec["levels"]))
    points = grid_points(size, size, int(spec["grid_points"]))
    localizer = TrainedLocalizer(model, device, amp=bool(config["optim"]["amp"]))
    summaries = {}
    for name, items in holdout.items():
        rows = []
        for item in items:
            make_target = None
            if "region" in item:
                make_target = (lambda m, region=item["region"], margin=item["margin"]:
                               cut_border_free(region, m, margin, size))
            rows.extend(step1_rows(
                localizer, dataset=name, crop_id=item["crop_id"], t1=item["t1"], t2=item["t2"],
                change_fraction=item["change_fraction"], ranges=ranges, levels=[int(v) for v in spec["levels"]],
                seed=int(config["seed"]), points=points, tolerance=float(spec["success_ate_px"]),
                trigger_probability=float(spec["trigger_probability"]), make_target=make_target,
            ))
        summaries[name] = dataset_summary(rows, spec)
    model.train()
    return {"summaries": summaries, "score": selection_score(summaries, spec["binding_levels"])}


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")}


def learning_rate(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(config: dict[str, Any], output: Path, device: torch.device, *, backbone: torch.nn.Module | None = None,
          max_steps: int | None = None) -> dict[str, Any]:
    optim = config["optim"]
    steps = int(max_steps or optim["steps"])
    batch_size = int(optim["batch_size"])
    seed_all(int(config["seed"]))
    dataset = build_dataset(config, steps * batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                        num_workers=int(config["data"]["num_workers"]), pin_memory=device.type == "cuda")
    model = build_localizer(config["model"], backbone=backbone).to(device)
    model.train()
    parameters = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(parameters, lr=float(optim["lr"]), weight_decay=float(optim["weight_decay"]))
    use_amp = bool(optim["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_spec = config["loss"]
    points = grid(int(config["data"]["crop_size"]), int(config["data"]["crop_size"]), int(loss_spec["grid"]), device=device)
    holdout = build_holdout(config)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    log = (output / "train_log.jsonl").open("a", encoding="utf-8")
    best = {"score": -math.inf, "step": None}
    start = time.perf_counter()
    step = 0
    for batch in loader:
        if step >= steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, steps, int(optim["warmup_steps"]), float(optim["lr"]))
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out1, out2, out_self = batched_forward(model, batch, device)
        with torch.autocast(device_type=device.type, enabled=False):
            loss, parts = localizer_loss(
                out1, out2, out_self,
                batch["matrix1"].to(device), batch["matrix2"].to(device), batch["change"].to(device),
                weights=loss_spec["weights"], stage_weights=loss_spec["stage_weights"], points=points,
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(optim["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % int(config["log_every"]) == 0 or step == 1:
            record = {"step": step, "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                      "seconds": time.perf_counter() - start, **parts}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record))
        if step % int(config["eval_every"]) == 0 or step == steps:
            result = evaluate(model, holdout, config, device)
            result["step"] = step
            (output / f"holdout_step{step:06d}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            payload = {"model_config": config["model"], "state": trainable_state(model), "step": step,
                       "holdout_score": result["score"]}
            torch.save(payload, output / "checkpoints" / "last.pth")
            if result["score"] > best["score"]:
                best = {"score": result["score"], "step": step}
                torch.save(payload, output / "checkpoints" / "best.pth")
            brief = {name: {level: s["levels"][level].get("success_rate", s["levels"][level]["trigger_rate"])
                            for level in s["levels"]} for name, s in result["summaries"].items()}
            print(json.dumps({"holdout_step": step, "score": result["score"], "levels": brief}))
    log.close()
    summary = {"steps": step, "best": best, "seconds": time.perf_counter() - start}
    (output / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_localizer_v1.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=None, help="smoke runs only")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps(train(config, output, torch.device(args.device), max_steps=args.max_steps)))


if __name__ == "__main__":
    main()
