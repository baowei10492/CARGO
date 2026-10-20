from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


DIRECTIONS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("min", "q25", "median", "q75", "q90", "max")}
    points = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        key: float(value)
        for key, value in zip(
            ("min", "q25", "median", "q75", "q90", "max"),
            points,
        )
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float | None:
    if values.size == 0:
        return None
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    output = np.zeros_like(mask)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    output[
        source_y0 + dy : source_y1 + dy,
        source_x0 + dx : source_x1 + dx,
    ] = mask[source_y0:source_y1, source_x0:source_x1]
    return output


def _paired_support(
    height: int,
    width: int,
    dx: int,
    dy: int,
) -> np.ndarray:
    support = np.ones((height, width), dtype=bool)
    border_x = abs(dx)
    border_y = abs(dy)
    if border_x:
        support[:, :border_x] = False
        support[:, width - border_x :] = False
    if border_y:
        support[:border_y, :] = False
        support[height - border_y :, :] = False
    return support


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    """Measure target scale and sensitivity to the project shift protocol."""

    binary = np.asarray(mask, dtype=bool)
    uint8 = binary.astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        uint8,
        connectivity=8,
    )
    component_areas = stats[1:count, cv2.CC_STAT_AREA].astype(np.float64)
    component_diameters = 2.0 * np.sqrt(component_areas / math.pi)
    changed = int(binary.sum())

    boundary_bands: dict[str, int] = {}
    for radius in (2, 4, 8):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        eroded = cv2.erode(
            uint8,
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        boundary_bands[str(radius)] = int((binary & ~eroded).sum())

    self_intersection: dict[str, int] = {}
    self_union: dict[str, int] = {}
    for pixels in (2, 4, 8):
        intersection = 0
        union = 0
        for direction_x, direction_y in DIRECTIONS:
            dx = direction_x * pixels
            dy = direction_y * pixels
            shifted = _shift_mask(binary, dx, dy)
            support = _paired_support(*binary.shape, dx, dy)
            intersection += int((binary & shifted & support).sum())
            union += int(((binary | shifted) & support).sum())
        self_intersection[str(pixels)] = intersection
        self_union[str(pixels)] = union

    return {
        "changed_pixels": changed,
        "total_pixels": int(binary.size),
        "component_areas": component_areas.tolist(),
        "component_diameters": component_diameters.tolist(),
        "boundary_band_pixels": boundary_bands,
        "self_shift_intersection": self_intersection,
        "self_shift_union": self_union,
    }


def audit_dataset(
    *,
    dataset_root: Path,
    manifest_path: Path,
    split: str,
    mask_threshold: int,
    dataset_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows for split {split}: {manifest_path}")

    total_pixels = 0
    changed_pixels = 0
    positive_samples = 0
    image_fractions: list[float] = []
    component_areas: list[float] = []
    component_diameters: list[float] = []
    boundary_pixels = {str(value): 0 for value in (2, 4, 8)}
    intersections = {str(value): 0 for value in (2, 4, 8)}
    unions = {str(value): 0 for value in (2, 4, 8)}
    pixel_values: set[int] = set()

    for index, row in enumerate(rows, start=1):
        mask_path = dataset_root / row["mask_path"]
        values = np.asarray(Image.open(mask_path))
        if values.ndim == 3:
            values = values[..., 0]
        pixel_values.update(int(value) for value in np.unique(values))
        geometry = mask_geometry(values > mask_threshold)
        changed = int(geometry["changed_pixels"])
        total = int(geometry["total_pixels"])
        changed_pixels += changed
        total_pixels += total
        positive_samples += int(changed > 0)
        image_fractions.append(changed / total)
        component_areas.extend(geometry["component_areas"])
        component_diameters.extend(geometry["component_diameters"])
        for pixels in ("2", "4", "8"):
            boundary_pixels[pixels] += int(
                geometry["boundary_band_pixels"][pixels]
            )
            intersections[pixels] += int(
                geometry["self_shift_intersection"][pixels]
            )
            unions[pixels] += int(
                geometry["self_shift_union"][pixels]
            )
        if index % 100 == 0 or index == len(rows):
            print(f"{dataset_name}: {index}/{len(rows)} masks")

    areas = np.asarray(component_areas, dtype=np.float64)
    diameters = np.asarray(component_diameters, dtype=np.float64)
    return {
        "status": "completed",
        "dataset": dataset_name,
        "dataset_root": str(dataset_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "split": split,
        "mask_threshold": int(mask_threshold),
        "test_split_touched": split == "test",
        "samples": len(rows),
        "positive_samples": positive_samples,
        "pixel_values": sorted(pixel_values),
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "change_fraction": changed_pixels / total_pixels,
        "image_change_fraction": _quantiles(image_fractions),
        "components": {
            "count": int(areas.size),
            "area_pixels": _quantiles(areas),
            "equivalent_diameter_pixels": _quantiles(diameters),
            "area_weighted_equivalent_diameter": {
                "q25": _weighted_quantile(diameters, areas, 0.25),
                "median": _weighted_quantile(diameters, areas, 0.5),
                "q75": _weighted_quantile(diameters, areas, 0.75),
            },
        },
        "boundary_band_fraction_of_change": {
            pixels: boundary_pixels[pixels] / max(1, changed_pixels)
            for pixels in ("2", "4", "8")
        },
        "mask_self_iou_on_paired_support": {
            pixels: intersections[pixels] / max(1, unions[pixels])
            for pixels in ("2", "4", "8")
        },
        "notes": [
            "All statistics are descriptive and validation-only.",
            "Self-IoU averages the same eight integer directions used by "
            "the project shift protocol and uses paired symmetric support.",
            "No detector predictions or test labels are read.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit validation-mask target scale and shift sensitivity."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mask-threshold", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("This audit is validation-only; test is forbidden")
    result = audit_dataset(
        dataset_root=Path(args.dataset_root),
        manifest_path=Path(args.manifest),
        split=str(args.split),
        mask_threshold=int(args.mask_threshold),
        dataset_name=str(args.dataset_name),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
