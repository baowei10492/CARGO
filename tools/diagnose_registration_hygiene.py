from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from rs_cd.datasets import PairedChangeDataset
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader
from tools.train_evaluate_task_benefit_router import _write_csv


WHOLE_IMAGE_TILE_INDEX = -1


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tile_dispersion_summary(
    records: list[dict[str, Any]],
    *,
    agreement_thresholds: tuple[float, ...] = (1.0, 2.0),
) -> dict[str, Any]:
    """Aggregate per-image dispersion of independent tile offset estimates.

    Under a *known non-zero* injected offset the tiles all have real
    correlation signal to lock onto, so agreement measures whether the
    residual misalignment is spatially constant: a pure global
    translation makes tiles agree, a spatially varying field
    (parallax, off-nadir, local registration error) makes them diverge.

    On clean pairs this quantity is NOT interpretable for the frozen
    conditional localizer, because that head cannot emit a zero offset:
    it is forced to pick some non-zero grid candidate, so tile
    disagreement measures the estimator's forced-choice noise floor and
    is *highest* where the true misalignment is smallest. See
    docs/P7_REGISTRATION_HYGIENE_DIAGNOSTIC_V2.md.

    ``records`` need ``id``, ``tile_index`` (``WHOLE_IMAGE_TILE_INDEX``
    for the full-image estimate), ``est_dx`` and ``est_dy``; optional
    ``gt_dx``/``gt_dy`` add error metrics against the injected offset.
    """

    by_id: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    truth: dict[str, tuple[float, float]] = {}
    for record in records:
        by_id[str(record["id"])][int(record["tile_index"])] = (
            float(record["est_dx"]),
            float(record["est_dy"]),
        )
        if "gt_dx" in record and "gt_dy" in record:
            truth[str(record["id"])] = (
                float(record["gt_dx"]),
                float(record["gt_dy"]),
            )
    if not by_id:
        raise RuntimeError("No records to summarize")

    whole_q: list[bool] = []
    tile_q: list[bool] = []
    for record in records:
        if "q_gate" not in record:
            continue
        value = record["q_gate"]
        flag = value if isinstance(value, bool) else (
            str(value).strip().lower() == "true"
        )
        if int(record["tile_index"]) == WHOLE_IMAGE_TILE_INDEX:
            whole_q.append(flag)
        else:
            tile_q.append(flag)

    max_pairwise: list[float] = []
    dispersion_radius: list[float] = []
    tile_magnitude: list[float] = []
    global_magnitude: list[float] = []
    agreement: dict[float, list[float]] = {
        threshold: [] for threshold in agreement_thresholds
    }
    tile_counts: set[int] = set()
    tile_error: list[float] = []
    global_error: list[float] = []
    injected_magnitude: list[float] = []

    for sample_id, entries in by_id.items():
        tiles = [
            offset
            for index, offset in entries.items()
            if index != WHOLE_IMAGE_TILE_INDEX
        ]
        if len(tiles) < 2:
            raise RuntimeError(
                f"Sample {sample_id} has fewer than two tile estimates"
            )
        tile_counts.add(len(tiles))
        widest = 0.0
        for first in range(len(tiles)):
            for second in range(first + 1, len(tiles)):
                widest = max(
                    widest,
                    _distance(tiles[first], tiles[second]),
                )
        max_pairwise.append(widest)
        centre = (
            sum(tile[0] for tile in tiles) / len(tiles),
            sum(tile[1] for tile in tiles) / len(tiles),
        )
        dispersion_radius.append(
            sum(_distance(tile, centre) for tile in tiles) / len(tiles)
        )
        tile_magnitude.extend(
            _distance(tile, (0.0, 0.0)) for tile in tiles
        )
        whole = entries.get(WHOLE_IMAGE_TILE_INDEX)
        if whole is not None:
            global_magnitude.append(_distance(whole, (0.0, 0.0)))
            for threshold in agreement_thresholds:
                agreement[threshold].append(
                    sum(
                        1
                        for tile in tiles
                        if _distance(tile, whole) <= threshold
                    )
                    / len(tiles)
                )
        injected = truth.get(sample_id)
        if injected is not None:
            injected_magnitude.append(_distance(injected, (0.0, 0.0)))
            tile_error.extend(
                _distance(tile, injected) for tile in tiles
            )
            if whole is not None:
                global_error.append(_distance(whole, injected))

    summary: dict[str, Any] = {
        "images": len(by_id),
        "tiles_per_image": sorted(tile_counts),
        "median_max_pairwise_tile_distance": _median(max_pairwise),
        "mean_max_pairwise_tile_distance": (
            sum(max_pairwise) / len(max_pairwise)
        ),
        "median_tile_dispersion_radius": _median(dispersion_radius),
        "median_tile_offset_magnitude": _median(tile_magnitude),
        "fraction_images_max_pairwise_leq_2px": (
            sum(value <= 2.0 for value in max_pairwise)
            / len(max_pairwise)
        ),
    }
    if global_magnitude:
        summary["median_global_offset_magnitude"] = _median(
            global_magnitude
        )
        for threshold in agreement_thresholds:
            summary[
                f"mean_tile_agreement_within_{threshold:g}px"
            ] = sum(agreement[threshold]) / len(agreement[threshold])
    if whole_q:
        summary["whole_image_q_trigger_rate"] = sum(whole_q) / len(
            whole_q
        )
    if tile_q:
        summary["tile_q_trigger_rate"] = sum(tile_q) / len(tile_q)
    if tile_error:
        summary["median_injected_offset_magnitude"] = _median(
            injected_magnitude
        )
        summary["forced_choice_regime"] = bool(
            _median(injected_magnitude) < 1e-6
        )
        summary["median_tile_error_magnitude"] = _median(tile_error)
        for threshold in (2.0, 4.0):
            summary[f"tile_error_pck{threshold:g}"] = sum(
                value <= threshold for value in tile_error
            ) / len(tile_error)
        if global_error:
            summary["median_global_error_magnitude"] = _median(
                global_error
            )
    return summary


def center_crop(image: torch.Tensor, size: int | None) -> torch.Tensor:
    """Crop the centre of a batch of images, for matched comparisons.

    Tile dispersion depends on image size and on how many tiles an image
    is cut into, so comparing a 512 px dataset against a 1024 px one at
    the same tile size is not apples-to-apples. Cropping the larger
    dataset down equalises both.
    """

    if size is None:
        return image
    height, width = int(image.shape[-2]), int(image.shape[-1])
    if size > height or size > width:
        raise ValueError(
            f"center_crop {size} exceeds image size {height}x{width}"
        )
    top = (height - size) // 2
    left = (width - size) // 2
    return image[:, :, top : top + size, left : left + size]


def _split_tiles(
    image: torch.Tensor,
    tile_size: int,
) -> list[torch.Tensor]:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    rows = height // int(tile_size)
    columns = width // int(tile_size)
    if rows < 1 or columns < 1:
        raise ValueError(
            "tile_size must not exceed the image dimensions"
        )
    if rows * columns < 2:
        raise ValueError("tile_size must yield at least two tiles")
    tiles: list[torch.Tensor] = []
    for row in range(rows):
        for column in range(columns):
            tiles.append(
                image[
                    :,
                    :,
                    row * tile_size : (row + 1) * tile_size,
                    column * tile_size : (column + 1) * tile_size,
                ]
            )
    return tiles


@torch.no_grad()
def collect_tile_records(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    *,
    severities: list[int],
    tile_size: int,
    crop_size: int | None,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    seed: int,
    amp: bool,
    device: torch.device,
) -> list[dict[str, Any]]:
    estimate = getattr(model, "estimate_registration", None)
    if not callable(estimate):
        raise TypeError("Model does not expose estimate_registration")

    def run(t1: torch.Tensor, t2: torch.Tensor) -> dict[str, Any]:
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            outputs = estimate(t1, t2)
        return {
            "offsets": outputs["conditional_hard_offset_pixels"]
            .float()
            .cpu(),
            "gate": (outputs["alignment_sample_gate"] >= 0.5).cpu(),
        }

    records: list[dict[str, Any]] = []
    for order, severity in enumerate(severities):
        spec = CorruptionSpec(
            name="clean" if int(severity) == 0 else "shift",
            severity=int(severity),
            mode="common" if int(severity) == 0 else "t2",
            seed=seed,
        )
        print(f"[hygiene {order + 1}/{len(severities)}] {spec.key}")
        dataset = CorruptedPairDataset(base_dataset, spec)
        loader = _loader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            seed=seed + order,
            device=device,
            prefetch_factor=int(prefetch_factor),
        )
        for batch in loader:
            t1 = center_crop(
                batch["image_t1"].to(device, non_blocking=True),
                crop_size,
            ).contiguous()
            t2 = center_crop(
                batch["image_t2"].to(device, non_blocking=True),
                crop_size,
            ).contiguous()
            whole = run(t1, t2)
            tiles_t1 = _split_tiles(t1, tile_size)
            tiles_t2 = _split_tiles(t2, tile_size)
            tile_outputs = [
                run(first.contiguous(), second.contiguous())
                for first, second in zip(tiles_t1, tiles_t2)
            ]
            for index, sample_id in enumerate(batch["id"]):
                records.append(
                    {
                        "id": str(sample_id),
                        "scenario": spec.key,
                        "tile_index": WHOLE_IMAGE_TILE_INDEX,
                        "est_dx": float(whole["offsets"][index, 0]),
                        "est_dy": float(whole["offsets"][index, 1]),
                        "gt_dx": float(batch["shift_dx"][index]),
                        "gt_dy": float(batch["shift_dy"][index]),
                        "q_gate": bool(whole["gate"][index]),
                    }
                )
                for tile_index, output in enumerate(tile_outputs):
                    records.append(
                        {
                            "id": str(sample_id),
                            "scenario": spec.key,
                            "tile_index": tile_index,
                            "est_dx": float(output["offsets"][index, 0]),
                            "est_dy": float(output["offsets"][index, 1]),
                            "gt_dx": float(batch["shift_dx"][index]),
                            "gt_dy": float(batch["shift_dy"][index]),
                            "q_gate": bool(output["gate"][index]),
                        }
                    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label-free registration hygiene diagnostic. Estimates the "
            "conditional offset independently on image tiles and on the "
            "whole image, then reports how much the tiles disagree. Low "
            "dispersion means the residual misalignment is close to a "
            "global translation (selective alignment is well posed); "
            "high dispersion means a spatially varying misalignment "
            "field that no per-image global estimate can represent. "
            "Reads no change labels."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--label", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--center-crop",
        type=int,
        default=None,
        help=(
            "Centre-crop every image to this size before estimating. "
            "Use it to match image size and tile count across datasets "
            "of different native resolutions."
        ),
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help=(
            "Injected shift severities. Severity 0 (clean) is accepted "
            "but is NOT interpretable for the conditional localizer "
            "(forced-choice noise floor); see the v2 protocol."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = select_device(args.device)
    seed_everything(int(args.seed))

    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["registration_only"] = False
    config["model"]["freeze_registration"] = True
    config["model"]["offset_decoding"] = "hard"
    config["model"]["alignment_gate"] = "threshold_nonzero"
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    data = config["data"]
    dataset = PairedChangeDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split=str(args.split),
        image_size=None,
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode="none",
    )
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            range(min(int(args.max_samples), len(dataset))),
        )

    records = collect_tile_records(
        model,
        dataset,
        severities=[int(value) for value in args.severities],
        tile_size=int(args.tile_size),
        crop_size=(
            int(args.center_crop)
            if args.center_crop is not None
            else None
        ),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        seed=int(args.seed),
        amp=True,
        device=device,
    )
    _write_csv(output_dir / "tile_offset_records.csv", records)

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scenario[str(record["scenario"])].append(record)
    summary = {
        "label": str(args.label),
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": str(args.split),
        "tile_size": int(args.tile_size),
        "center_crop": (
            int(args.center_crop)
            if args.center_crop is not None
            else None
        ),
        "scenarios": {
            scenario: tile_dispersion_summary(
                rows,
                agreement_thresholds=(2.0, 4.0),
            )
            for scenario, rows in by_scenario.items()
        },
    }
    save_json(output_dir / "hygiene_summary.json", summary)
    print("===== REGISTRATION HYGIENE SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved hygiene diagnostic to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
