"""Measure label self-overlap as a function of displacement magnitude.

Every applicability result so far compares datasets at a fixed 8-pixel
shift. That number was calibrated on LEVIR-CD, where changed objects have
a median area-weighted diameter of 37.5 pixels, so 8 pixels is 21% of an
object. On DSIFN-CD, whose objects are 286 pixels across, the same 8
pixels is under 3%. Holding the pixel count fixed therefore asks a
different question of each dataset, and the resulting picture of a narrow
applicability window may be an artefact of the scale at which it was
measured.

This tool sweeps the displacement instead of fixing it, so the condition
can be stated in a scale-free way: how much displacement, relative to
object size, does a dataset need before a shift costs real overlap. It
reads labels only, loads no model, and touches no test split. The shift
geometry and the object-scale measurement are both imported from the
frozen morphology auditor so this report and the frozen audits state the
same quantities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from tools.audit_change_target_scale import (
    DIRECTIONS,
    _paired_support,
    _shift_mask,
    _weighted_quantile,
)


def self_iou_at(binary: np.ndarray, pixels: int) -> tuple[int, int]:
    """Intersection and union over the eight project shift directions."""
    intersection = 0
    union = 0
    for direction_x, direction_y in DIRECTIONS:
        dx = direction_x * pixels
        dy = direction_y * pixels
        shifted = _shift_mask(binary, dx, dy)
        support = _paired_support(*binary.shape, dx, dy)
        intersection += int((binary & shifted & support).sum())
        union += int(((binary | shifted) & support).sum())
    return intersection, union


def scan(
    dataset_root: Path,
    manifest: Path,
    *,
    split: str,
    mask_threshold: int,
    shifts: tuple[int, ...],
    max_samples: int | None,
) -> dict[str, Any]:
    intersections = {shift: 0 for shift in shifts}
    unions = {shift: 0 for shift in shifts}
    areas: list[float] = []
    diameters: list[float] = []
    samples = 0
    changed_pixels = 0
    total_pixels = 0

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest has no rows for split {split!r}")
    if max_samples is not None:
        rows = rows[:max_samples]

    for index, row in enumerate(rows, start=1):
        with Image.open(dataset_root / row["mask_path"]) as handle:
            mask = np.asarray(handle.convert("L"))
        binary = mask > mask_threshold
        changed_pixels += int(binary.sum())
        total_pixels += int(binary.size)
        samples += 1

        # Object scale, measured exactly as the frozen morphology auditor
        # measures it, so the ratio below divides two same-口径 numbers
        # instead of one measured value by one quoted constant.
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8),
            connectivity=8,
        )
        component_areas = stats[1:count, cv2.CC_STAT_AREA].astype(np.float64)
        areas.extend(component_areas.tolist())
        diameters.extend((2.0 * np.sqrt(component_areas / np.pi)).tolist())

        for shift in shifts:
            intersection, union = self_iou_at(binary, shift)
            intersections[shift] += intersection
            unions[shift] += union
        if index % 200 == 0:
            print(f"  {index}/{len(rows)}")

    if changed_pixels == 0:
        raise SystemExit(
            f"No change pixels at mask_threshold {mask_threshold}; the "
            "threshold does not match the label encoding"
        )

    # A shift with no paired support at all produces union = 0. Report that
    # as null rather than as a self-IoU of 0.0, which reads like a measured
    # collapse when it is really an undefined point.
    curve = {
        str(shift): (
            intersections[shift] / unions[shift] if unions[shift] else None
        )
        for shift in shifts
    }

    # A large shift leaves only part of the tile paired with itself, so the
    # far end of the curve rests on less data than the near end. Ask the
    # frozen primitive rather than reimplementing its geometry: it trims
    # |d| from BOTH sides, so the usable area shrinks twice as fast as a
    # one-sided reading suggests and hits zero at 2*shift = tile size.
    height, width = binary.shape
    support_fraction = {
        str(shift): min(
            float(_paired_support(height, width, dx * shift, dy * shift).mean())
            for dx, dy in DIRECTIONS
        )
        for shift in shifts
    }
    return {
        "status": "completed",
        "split": split,
        "samples": samples,
        "mask_threshold": mask_threshold,
        "change_fraction": changed_pixels / total_pixels,
        "shifts": list(shifts),
        "self_iou_curve": curve,
        "tile_shape": [int(height), int(width)],
        "min_paired_support_fraction": support_fraction,
        "area_weighted_median_diameter_px": _weighted_quantile(
            np.asarray(diameters, dtype=np.float64),
            np.asarray(areas, dtype=np.float64),
            0.5,
        ),
        "components": len(diameters),
        "labels_only": True,
        "test_split_touched": split == "test",
    }


def equivalent_shift(
    curve: dict[str, float],
    *,
    reference_self_iou: float,
) -> float | None:
    """Displacement at which self-IoU falls to the reference value.

    Linear interpolation between the two bracketing measured shifts.
    Returns None when the sweep never reaches the reference, which means
    the dataset needs a larger displacement than was scanned.
    """
    points = sorted(
        (int(shift), value)
        for shift, value in curve.items()
        if value is not None
    )
    for (low_shift, low_value), (high_shift, high_value) in zip(
        points, points[1:]
    ):
        if low_value >= reference_self_iou >= high_value:
            if low_value == high_value:
                return float(low_shift)
            span = (low_value - reference_self_iou) / (low_value - high_value)
            return low_shift + span * (high_shift - low_shift)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep displacement magnitude and report label self-IoU, so "
            "the applicability condition can be stated relative to object "
            "scale rather than in fixed pixels."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mask-threshold", type=int, required=True)
    parser.add_argument(
        "--shifts",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--reference-self-iou",
        type=float,
        default=0.5199746945008742,
        help="LEVIR-CD self-IoU at 8 px, the one measured positive case",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.split.strip().lower().startswith("test"):
        raise SystemExit(
            "refusing to read a test split: every candidate dataset still "
            "has its test partition sealed"
        )

    report = scan(
        Path(args.dataset_root).expanduser().resolve(),
        Path(args.manifest).expanduser().resolve(),
        split=args.split,
        mask_threshold=int(args.mask_threshold),
        shifts=tuple(sorted(set(int(value) for value in args.shifts))),
        max_samples=args.max_samples,
    )
    report["dataset"] = args.dataset_name
    report["reference_self_iou"] = float(args.reference_self_iou)
    report["equivalent_shift_pixels"] = equivalent_shift(
        report["self_iou_curve"],
        reference_self_iou=float(args.reference_self_iou),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n{args.dataset_name}: change fraction "
          f"{report['change_fraction']:.4f}, area-weighted median diameter "
          f"{report['area_weighted_median_diameter_px']:.1f} px")
    for shift in report["shifts"]:
        value = report["self_iou_curve"][str(shift)]
        support = report["min_paired_support_fraction"][str(shift)]
        rendered = "undefined" if value is None else f"{value:.4f}"
        print(f"  {shift:>4d} px -> self-IoU {rendered:>9s}"
              f"   (paired support >= {support:.2f})")
    equivalent = report["equivalent_shift_pixels"]
    print(
        "  equivalent displacement (self-IoU = "
        f"{report['reference_self_iou']:.3f}): "
        + ("beyond the scanned range"
           if equivalent is None else f"{equivalent:.1f} px")
    )


if __name__ == "__main__":
    main()
