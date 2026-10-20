from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from rs_cd.datasets import PairedChangeDataset
from rs_cd.dual_path import select_aligned_by_task_loss
from rs_cd.robustness import CorruptedPairDataset
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.evaluate_safe_alignment_transfer import (
    load_safe_classifier_bundle,
    merge_detector_state,
)
from tools.train import _loader
from tools.train_evaluate_safe_alignment_classifier import (
    _classifier_probability,
)
from tools.train_evaluate_task_benefit_router import (
    _forward_experts,
    _scenario_specs,
    _valid_mask,
)


"""Render qualitative raw/aligned/routed panels on a non-sealed split.

The frozen one-time test reports metrics only; it stores no prediction
rasters, and the freeze protocol forbids rerunning sealed inference to
obtain nicer pictures. Validation was already consumed during method
development, so re-running inference there adds no new information and
cannot influence any decision. This tool therefore refuses to touch the
test split, computes no threshold, and writes no gate verdict: it only
visualises decisions that the frozen router already makes.
"""


FORBIDDEN_SPLITS = {"test"}

# PairedChangeDataset emits the identifier as "id", not "sample_id".
REQUIRED_BATCH_KEYS = {"id", "image_t1", "image_t2", "mask"}


def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo imagenet normalisation for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = tensor.detach().cpu().float() * std + mean
    array = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def _mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    array = mask.detach().cpu().float().squeeze().numpy()
    return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def overlay_error_map(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Colour true positives, false positives and false negatives."""
    predicted = prediction > 127
    truth = target > 127
    panel = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    panel[predicted & truth] = (255, 255, 255)
    panel[predicted & ~truth] = (214, 96, 77)
    panel[~predicted & truth] = (69, 117, 180)
    return panel


def _sample_f1(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold: float,
) -> float:
    probability = torch.sigmoid(logits.float())
    predicted = (probability >= threshold) & (valid > 0)
    truth = (mask > 0.5) & (valid > 0)
    true_positive = float((predicted & truth).sum())
    false_positive = float((predicted & ~truth).sum())
    false_negative = float((~predicted & truth).sum())
    denominator = 2.0 * true_positive + false_positive + false_negative
    if denominator <= 0.0:
        return float("nan")
    return 2.0 * true_positive / denominator


def change_bounding_box(
    masks: list[np.ndarray],
    *,
    margin_fraction: float = 0.6,
    minimum_size: int = 192,
) -> tuple[int, int, int, int]:
    """Locate the region worth showing, as (top, left, height, width).

    LEVIR-CD changes have a median equivalent diameter near 37 pixels in
    a 1024-pixel tile, so a full-frame panel prints as an empty image.
    The window covers the union of ground truth and both predictions, so
    a false positive outside the labelled region is not cropped away.
    """
    height, width = masks[0].shape[:2]
    union = np.zeros((height, width), dtype=bool)
    for mask in masks:
        union |= mask > 127
    if not union.any():
        side = min(height, width, max(minimum_size, 1))
        return (
            (height - side) // 2,
            (width - side) // 2,
            side,
            side,
        )

    rows = np.where(union.any(axis=1))[0]
    columns = np.where(union.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1

    side = max(bottom - top, right - left)
    side = int(round(side * (1.0 + margin_fraction)))
    side = max(side, minimum_size)
    side = min(side, height, width)

    centre_row = (top + bottom) // 2
    centre_column = (left + right) // 2
    top = int(np.clip(centre_row - side // 2, 0, height - side))
    left = int(np.clip(centre_column - side // 2, 0, width - side))
    return top, left, side, side


def _crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, height, width = box
    return array[top: top + height, left: left + width]


def _stack_panels(panels: list[np.ndarray], *, gap: int = 6) -> np.ndarray:
    height = max(panel.shape[0] for panel in panels)
    total = sum(panel.shape[1] for panel in panels) + gap * (
        len(panels) - 1
    )
    canvas = np.full((height, total, 3), 255, dtype=np.uint8)
    offset = 0
    for panel in panels:
        rgb = (
            panel
            if panel.ndim == 3
            else np.repeat(panel[:, :, None], 3, axis=2)
        )
        canvas[: rgb.shape[0], offset: offset + rgb.shape[1]] = rgb
        offset += rgb.shape[1] + gap
    return canvas


def select_cases(
    records: list[dict[str, Any]],
    *,
    per_category: int,
) -> dict[str, list[dict[str, Any]]]:
    """Pick illustrative decisions without using any test data.

    Four categories make the routing behaviour legible: a correct
    abstention on a clean pair that alignment would have damaged, a
    correct adoption under large displacement, a missed opportunity, and
    a harmful adoption if one exists at the frozen threshold.
    """
    clean = [row for row in records if row["scenario_severity"] == 0]
    shifted = [row for row in records if row["scenario_severity"] > 0]

    correct_abstention = sorted(
        (
            row
            for row in clean
            if not row["selected_aligned"]
            and row["aligned_f1"] < row["raw_f1"]
        ),
        key=lambda row: row["raw_f1"] - row["aligned_f1"],
        reverse=True,
    )
    correct_adoption = sorted(
        (
            row
            for row in shifted
            if row["selected_aligned"]
            and row["aligned_f1"] > row["raw_f1"]
        ),
        key=lambda row: row["aligned_f1"] - row["raw_f1"],
        reverse=True,
    )
    missed_opportunity = sorted(
        (
            row
            for row in shifted
            if not row["selected_aligned"]
            and row["aligned_f1"] > row["raw_f1"]
        ),
        key=lambda row: row["aligned_f1"] - row["raw_f1"],
        reverse=True,
    )
    harmful_adoption = sorted(
        (
            row
            for row in records
            if row["selected_aligned"]
            and row["aligned_f1"] < row["raw_f1"]
        ),
        key=lambda row: row["raw_f1"] - row["aligned_f1"],
        reverse=True,
    )
    return {
        "correct_abstention_clean": correct_abstention[:per_category],
        "correct_adoption_shift": correct_adoption[:per_category],
        "missed_opportunity": missed_opportunity[:per_category],
        "harmful_adoption": harmful_adoption[:per_category],
    }


@torch.no_grad()
def collect_records(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    bundle: dict[str, Any],
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    keep_rasters: bool,
) -> list[dict[str, Any]]:
    threshold = float(protocol["threshold"])
    data = config["data"]
    loss_config = config["loss"]
    pos_weight = loss_config.get("pos_weight")
    records: list[dict[str, Any]] = []
    raster_dir = output_dir / "rasters"
    if keep_rasters:
        raster_dir.mkdir(parents=True, exist_ok=True)

    for scenario_index, spec in enumerate(_scenario_specs(protocol)):
        dataset = CorruptedPairDataset(
            base_dataset,
            spec,
            normalization=str(data.get("normalization", "imagenet")),
        )
        loader = _loader(
            dataset=dataset,
            batch_size=int(protocol["batch_size"]),
            shuffle=False,
            num_workers=int(protocol["num_workers"]),
            seed=int(protocol["seed"]) + scenario_index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        print(f"[panels] {spec.key}")
        for batch in loader:
            missing = REQUIRED_BATCH_KEYS - set(batch)
            if missing:
                raise KeyError(
                    f"Batch is missing {sorted(missing)}; available keys "
                    f"are {sorted(batch)}"
                )
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = _valid_mask(
                batch,
                mask,
                border=int(protocol["evaluation_border_pixels"]),
                device=device,
            )
            registration, raw_logits, aligned_logits, features = (
                _forward_experts(
                    model,
                    t1,
                    t2,
                    amp=bool(protocol.get("amp", True)),
                    maximum_offset_pixels=float(
                        protocol["maximum_offset_pixels"]
                    ),
                    threshold=threshold,
                    valid=valid,
                )
            )
            _, oracle_gate, _, _ = select_aligned_by_task_loss(
                raw_logits,
                aligned_logits,
                mask,
                valid,
                bce_weight=float(loss_config.get("bce_weight", 1.0)),
                dice_weight=float(loss_config.get("dice_weight", 1.0)),
                pos_weight=(
                    float(pos_weight) if pos_weight is not None else None
                ),
            )
            q_gate = registration["alignment_sample_gate"] >= 0.5
            probability = _classifier_probability(bundle, features)
            selected = q_gate & (
                probability >= float(bundle["probability_threshold"])
            )
            offsets = registration["conditional_hard_offset_pixels"]

            for index in range(t1.shape[0]):
                sample_id = str(batch["id"][index])
                raw_f1 = _sample_f1(
                    raw_logits[index],
                    mask[index],
                    valid[index],
                    threshold=threshold,
                )
                aligned_f1 = _sample_f1(
                    aligned_logits[index],
                    mask[index],
                    valid[index],
                    threshold=threshold,
                )
                record = {
                    "sample_id": sample_id,
                    "scenario": spec.key,
                    "scenario_severity": int(spec.value),
                    "raw_f1": raw_f1,
                    "aligned_f1": aligned_f1,
                    "router_probability": float(probability[index]),
                    "q_gate": bool(q_gate[index]),
                    "selected_aligned": bool(selected[index]),
                    "oracle_prefers_aligned": bool(oracle_gate[index]),
                    "estimated_offset_px": [
                        float(offsets[index, 0]),
                        float(offsets[index, 1]),
                    ],
                }
                if keep_rasters:
                    stem = f"{spec.key.replace(':', '_')}__{sample_id}"
                    payload = {
                        "t1": _to_uint8_image(t1[index]),
                        "t2": _to_uint8_image(t2[index]),
                        "gt": _mask_to_uint8(mask[index]),
                        "raw": _mask_to_uint8(
                            (torch.sigmoid(raw_logits[index].float())
                             >= threshold).float()
                        ),
                        "aligned": _mask_to_uint8(
                            (torch.sigmoid(aligned_logits[index].float())
                             >= threshold).float()
                        ),
                    }
                    np.savez_compressed(
                        raster_dir / f"{stem}.npz", **payload
                    )
                    record["raster"] = f"rasters/{stem}.npz"
                records.append(record)
    return records


def render_panel(
    raster: dict[str, np.ndarray],
    record: dict[str, Any],
    output_path: Path,
    *,
    crop_to_change: bool = True,
) -> tuple[int, int, int, int] | None:
    raw_overlay = overlay_error_map(raster["raw"], raster["gt"])
    aligned_overlay = overlay_error_map(raster["aligned"], raster["gt"])
    selected = (
        aligned_overlay if record["selected_aligned"] else raw_overlay
    )
    layers = [
        raster["t1"],
        raster["t2"],
        np.repeat(raster["gt"][:, :, None], 3, axis=2),
        raw_overlay,
        aligned_overlay,
        selected,
    ]
    box = None
    if crop_to_change:
        box = change_bounding_box(
            [raster["gt"], raster["raw"], raster["aligned"]]
        )
        layers = [_crop(layer, box) for layer in layers]
    Image.fromarray(_stack_panels(layers)).save(output_path)
    return box


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render raw/aligned/routed qualitative panels on a "
            "non-sealed split using frozen weights. Computes no "
            "threshold and makes no selection."
        )
    )
    parser.add_argument("--p7-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--classifier-task", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help=(
            "Render full tiles. The default crops to the changed region "
            "because LEVIR-CD changes are small relative to the tile."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if str(args.split).strip().lower() in FORBIDDEN_SPLITS:
        raise SystemExit(
            "Refusing to render panels from a sealed test split: the "
            "frozen protocol allows exactly one test evaluation and it "
            "has been consumed. Use the validation split."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    device = select_device(args.device)
    seed_everything(int(protocol["seed"]))

    checkpoint = torch.load(
        Path(args.p7_checkpoint).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["registration_only"] = False
    config["model"]["freeze_registration"] = True
    config["model"]["offset_decoding"] = "hard"
    config["model"]["alignment_gate"] = "threshold_nonzero"
    model = build_model(config).to(device)
    detector = torch.load(
        Path(args.detector_checkpoint).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    state, replaced = merge_detector_state(
        checkpoint["model"], detector["model"]
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Merged {replaced} detector tensors")

    # The bundle stays on CPU; _classifier_probability moves it to the
    # feature device on first use, matching the other evaluators.
    bundle = load_safe_classifier_bundle(Path(args.classifier_task))

    dataset = PairedChangeDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split=str(args.split),
        image_size=None,
        normalization=config["data"].get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(config["data"].get("mask_threshold", 127)),
        cache_mode="none",
    )

    records = collect_records(
        model,
        dataset,
        bundle,
        protocol=protocol,
        config=config,
        device=device,
        output_dir=output_dir,
        keep_rasters=True,
    )
    cases = select_cases(records, per_category=int(args.per_category))

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, list[str]] = {}
    for category, chosen in cases.items():
        rendered[category] = []
        for rank, record in enumerate(chosen):
            raster_path = output_dir / str(record["raster"])
            with np.load(raster_path) as payload:
                raster = {key: payload[key] for key in payload.files}
            name = f"{category}_{rank + 1}.png"
            box = render_panel(
                raster,
                record,
                figures / name,
                crop_to_change=not bool(args.no_crop),
            )
            record["crop_box_top_left_height_width"] = (
                list(box) if box is not None else None
            )
            rendered[category].append(name)

    save_json(
        output_dir / "qualitative_index.json",
        {
            "status": "completed",
            "split": str(args.split),
            "test_split_touched": False,
            "reporting_only_no_selection": True,
            "panel_order": [
                "T1",
                "T2",
                "ground truth",
                "raw expert (TP white, FP red, FN blue)",
                "aligned expert",
                "routed output",
            ],
            "router_threshold": float(bundle["probability_threshold"]),
            "samples": len(records),
            "cases": cases,
            "figures": rendered,
        },
    )
    print(f"Wrote qualitative panels to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
