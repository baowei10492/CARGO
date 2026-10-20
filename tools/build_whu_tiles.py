from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rs_cd.runtime import save_json


Image.MAX_IMAGE_PIXELS = None

TWO_PERIOD_DIR = "1. The two-period image data"
EARLY_YEAR = "2012"
LATE_YEAR = "2016"


def full_tile_grid(
    width: int,
    height: int,
    tile_size: int,
) -> tuple[int, int]:
    """Return (columns, rows) of *complete* tiles at a fixed origin.

    Incomplete edge tiles are dropped rather than padded, so that every
    emitted tile is real imagery. This is the only intentional deviation
    from the shipped 512 grid, which pads the final column and row.
    """

    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    columns = width // tile_size
    rows = height // tile_size
    if columns < 1 or rows < 1:
        raise ValueError("tile_size exceeds the raster dimensions")
    return columns, rows


def column_partition(
    total_columns: int,
    validation_columns: int,
    buffer_columns: int,
) -> dict[str, list[int]]:
    """Split raster columns into train / dropped buffer / validation.

    The buffer columns are discarded entirely so that no training tile is
    spatially adjacent to a validation tile. Splitting by whole columns
    (full-height strips) keeps the separation simple to state and audit.
    """

    if validation_columns < 1 or buffer_columns < 0:
        raise ValueError(
            "validation_columns must be >= 1 and buffer_columns >= 0"
        )
    train_count = total_columns - validation_columns - buffer_columns
    if train_count < 1:
        raise ValueError(
            "Not enough columns for train / buffer / validation"
        )
    train = list(range(train_count))
    buffer = list(range(train_count, train_count + buffer_columns))
    validation = list(
        range(train_count + buffer_columns, total_columns)
    )
    if len(validation) != validation_columns:
        raise RuntimeError("Validation column count is inconsistent")
    overlap = (set(train) & set(validation)) | (
        set(train) & set(buffer)
    )
    if overlap:
        raise RuntimeError("Column partitions overlap")
    return {
        "train": train,
        "buffer_dropped": buffer,
        "val": validation,
    }


def _load_raster(path: Path, *, expect_rgb: bool) -> np.ndarray:
    image = Image.open(path)
    if expect_rgb:
        array = np.array(image.convert("RGB"))
    else:
        array = np.array(image)
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        elif array.dtype != np.uint8:
            raise TypeError(f"Unexpected label dtype {array.dtype}")
        if array.ndim != 2:
            raise ValueError(f"Label raster is not 2-D: {array.shape}")
        unique = np.unique(array)
        if not set(unique.tolist()) <= {0, 255}:
            raise ValueError(
                f"Label raster is not binary, values={unique[:8]}"
            )
    return array


def _write_tile(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=False)


def build_split(
    raw_root: Path,
    output_root: Path,
    raster_split: str,
    *,
    tile_size: int,
    validation_columns: int,
    buffer_columns: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    two_period = raw_root / TWO_PERIOD_DIR
    early = next(
        (
            two_period
            / EARLY_YEAR
            / "whole_image"
            / raster_split
            / "image"
        ).glob("*.tif")
    )
    late = next(
        (
            two_period
            / LATE_YEAR
            / "whole_image"
            / raster_split
            / "image"
        ).glob("*.tif")
    )
    label = next(
        (two_period / "change_label" / raster_split).glob(
            "change_label.tif"
        )
    )
    print(f"[{raster_split}] A={early.name} B={late.name} L={label.name}")

    a_array = _load_raster(early, expect_rgb=True)
    b_array = _load_raster(late, expect_rgb=True)
    label_array = _load_raster(label, expect_rgb=False)
    shapes = {
        "A": a_array.shape[:2],
        "B": b_array.shape[:2],
        "label": label_array.shape[:2],
    }
    if len({value for value in shapes.values()}) != 1:
        raise ValueError(f"Raster shapes disagree: {shapes}")
    height, width = a_array.shape[:2]
    columns, rows = full_tile_grid(width, height, tile_size)
    print(
        f"[{raster_split}] raster {width}x{height} -> "
        f"{columns} cols x {rows} rows complete tiles "
        f"(dropped {width - columns * tile_size} px right, "
        f"{height - rows * tile_size} px bottom)"
    )

    if raster_split == "train":
        partition = column_partition(
            columns,
            validation_columns,
            buffer_columns,
        )
    else:
        partition = {
            "test": list(range(columns)),
            "buffer_dropped": [],
        }
    for name, values in partition.items():
        print(f"[{raster_split}] {name}: {len(values)} columns {values}")

    column_to_split: dict[int, str] = {}
    for name, values in partition.items():
        if name == "buffer_dropped":
            continue
        for column in values:
            column_to_split[column] = name

    records: list[dict[str, Any]] = []
    for row in range(rows):
        for column in range(columns):
            target_split = column_to_split.get(column)
            if target_split is None:
                continue
            top = row * tile_size
            left = column * tile_size
            mask_tile = label_array[
                top : top + tile_size, left : left + tile_size
            ]
            sample_id = f"whu_{target_split}_r{row:02d}c{column:02d}"
            records.append(
                {
                    "sample_id": sample_id,
                    "split": target_split,
                    "raster_split": raster_split,
                    "row": row,
                    "column": column,
                    "top": top,
                    "left": left,
                    "change_fraction": float(
                        (mask_tile >= 128).mean()
                    ),
                }
            )
            if dry_run:
                continue
            _write_tile(
                a_array[
                    top : top + tile_size, left : left + tile_size
                ],
                output_root / target_split / "A" / f"{sample_id}.png",
            )
            _write_tile(
                b_array[
                    top : top + tile_size, left : left + tile_size
                ],
                output_root / target_split / "B" / f"{sample_id}.png",
            )
            _write_tile(
                mask_tile,
                output_root
                / target_split
                / "label"
                / f"{sample_id}.png",
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the WHU building change detection dataset in the "
            "pipeline layout. All three channels (2012 image, 2016 "
            "image, change label) are cropped from the whole-image "
            "rasters with one shared convention, so no reverse "
            "engineering of the shipped edge padding is needed. The "
            "official train/test raster separation is preserved; the "
            "validation split is carved from the train raster as "
            "disjoint column strips with a discarded buffer, and tile "
            "ids encode (row, column) so later protocols can partition "
            "spatially instead of by hashed id."
        )
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--validation-columns", type=int, default=6)
    parser.add_argument("--buffer-columns", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for raster_split in ("train", "test"):
        records.extend(
            build_split(
                raw_root,
                output_root,
                raster_split,
                tile_size=int(args.tile_size),
                validation_columns=int(args.validation_columns),
                buffer_columns=int(args.buffer_columns),
                dry_run=bool(args.dry_run),
            )
        )

    counts: dict[str, int] = {}
    change: dict[str, list[float]] = {}
    for record in records:
        split = str(record["split"])
        counts[split] = counts.get(split, 0) + 1
        change.setdefault(split, []).append(
            float(record["change_fraction"])
        )
    summary = {
        "tile_size": int(args.tile_size),
        "validation_columns": int(args.validation_columns),
        "buffer_columns": int(args.buffer_columns),
        "counts": counts,
        "change_fraction_mean": {
            key: sum(values) / len(values)
            for key, values in change.items()
        },
        "empty_tile_fraction": {
            key: sum(value == 0.0 for value in values) / len(values)
            for key, values in change.items()
        },
        "dry_run": bool(args.dry_run),
    }
    print("===== WHU TILE BUILD SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.dry_run:
        return
    index_path = output_root / "tile_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)
    save_json(output_root / "tile_build_summary.json", summary)
    print(f"Wrote {len(records)} tiles to {output_root.resolve()}")
    print(f"Spatial index: {index_path.resolve()}")


if __name__ == "__main__":
    main()
