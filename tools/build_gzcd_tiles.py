from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


"""Build GZ-CD tiles with a raster-level split.

GZ-CD ships 20 large bi-temporal rasters and no official partition, so
both the tiling and the split are defined here. Every tile cut from one
raster is assigned to the same split, which makes spatial disjointness a
property of the construction rather than something an audit has to
discover afterwards. That is the WHU-CD approach, and it is mandatory
after the DSIFN-CD leakage found in main record 110.2.

Filenames encode different acquisition dates for T1, T2 and the label,
so pairing is by the `testN` token rather than by filename stem.
"""


Image.MAX_IMAGE_PIXELS = None

RASTER_TOKEN = re.compile(r"P_GZ_(test\d+)_")

# Frozen in docs/P17_GZCD_SECOND_POSITIVE_PROTOCOL.md before any tile was
# written. Assignment is by raster, generated deterministically from tile
# counts to approach 70/15/15.
SPLIT_ASSIGNMENT: dict[str, str] = {
    "test1": "train",
    "test12": "train",
    "test18": "train",
    "test2": "train",
    "test27": "train",
    "test28": "train",
    "test4": "train",
    "test5": "train",
    "test6": "train",
    "test13": "val",
    "test15": "val",
    "test17": "val",
    "test21": "val",
    "test22": "val",
    "test10": "test",
    "test19": "test",
    "test23": "test",
    "test24": "test",
    "test26": "test",
    "test9": "test",
}

# Geometric maxima at tile size 512, before the no-data filter.
EXPECTED_GEOMETRIC_TILES = {"train": 511, "val": 109, "test": 109}


def raster_token(path: Path) -> str:
    match = RASTER_TOKEN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse raster token from {path.name}")
    return match.group(1)


def index_by_raster(folder: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    found: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in suffixes or path.name.startswith("."):
            continue
        token = raster_token(path)
        if token in found:
            raise ValueError(
                f"Duplicate raster token {token}: {found[token].name} "
                f"and {path.name}"
            )
        found[token] = path
    return found


def is_no_data(tile: np.ndarray, *, max_black_fraction: float) -> bool:
    """Reject tiles dominated by pure black, which marks mosaic gaps.

    Preregistered rule: a tile is dropped when either temporal image has
    at least ``max_black_fraction`` of exactly-zero RGB pixels.
    """
    black = np.all(tile == 0, axis=2)
    return float(black.mean()) >= max_black_fraction


def tile_origins(width: int, height: int, tile: int) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(0, height - tile + 1, tile)
        for column in range(0, width - tile + 1, tile)
    ]


def build(
    source_root: Path,
    output_root: Path,
    *,
    tile: int,
    max_black_fraction: float,
    manifest_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    t1_files = index_by_raster(source_root / "T1", (".tif", ".tiff"))
    t2_files = index_by_raster(source_root / "T2", (".tif", ".tiff"))
    label_files = index_by_raster(source_root / "labels_change", (".png",))

    tokens = sorted(set(t1_files) & set(t2_files) & set(label_files))
    missing = sorted(
        (set(t1_files) | set(t2_files) | set(label_files)) - set(tokens)
    )
    if missing:
        raise ValueError(f"Rasters missing a T1/T2/label member: {missing}")
    unassigned = sorted(set(tokens) - set(SPLIT_ASSIGNMENT))
    if unassigned:
        raise ValueError(
            f"Rasters absent from the frozen split assignment: {unassigned}"
        )

    rows: list[dict[str, str]] = []
    per_split: dict[str, int] = defaultdict(int)
    geometric: dict[str, int] = defaultdict(int)
    dropped: dict[str, int] = defaultdict(int)
    raster_report: list[dict[str, Any]] = []

    for token in tokens:
        split = SPLIT_ASSIGNMENT[token]
        with Image.open(t1_files[token]) as handle:
            t1 = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(t2_files[token]) as handle:
            t2 = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        with Image.open(label_files[token]) as handle:
            label = np.asarray(handle.convert("L"), dtype=np.uint8)
        if not (t1.shape[:2] == t2.shape[:2] == label.shape[:2]):
            raise ValueError(
                f"{token}: shape mismatch "
                f"T1{t1.shape[:2]} T2{t2.shape[:2]} label{label.shape[:2]}"
            )
        height, width = t1.shape[:2]
        origins = tile_origins(width, height, tile)
        geometric[split] += len(origins)
        kept = 0
        for row, column in origins:
            t1_tile = t1[row: row + tile, column: column + tile]
            t2_tile = t2[row: row + tile, column: column + tile]
            if is_no_data(
                t1_tile, max_black_fraction=max_black_fraction
            ) or is_no_data(
                t2_tile, max_black_fraction=max_black_fraction
            ):
                dropped[split] += 1
                continue
            sample_id = f"{token}_{row:05d}_{column:05d}"
            relative = {
                "t1": f"{split}/A/{sample_id}.png",
                "t2": f"{split}/B/{sample_id}.png",
                "mask": f"{split}/label/{sample_id}.png",
            }
            if not dry_run:
                for key, array in (
                    ("t1", t1_tile),
                    ("t2", t2_tile),
                    ("mask", label[row: row + tile, column: column + tile]),
                ):
                    target = output_root / relative[key]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(target)
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "t1_path": relative["t1"],
                    "t2_path": relative["t2"],
                    "mask_path": relative["mask"],
                    "parent_id": token,
                }
            )
            kept += 1
            per_split[split] += 1
        raster_report.append(
            {
                "raster": token,
                "split": split,
                "size": [int(width), int(height)],
                "geometric_tiles": len(origins),
                "kept_tiles": kept,
            }
        )
        print(
            f"{token:8s} {split:5s} {width}x{height} "
            f"tiles {kept}/{len(origins)}"
        )

    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "split",
                    "t1_path",
                    "t2_path",
                    "mask_path",
                    "parent_id",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    spanning = {
        parent: sorted(splits)
        for parent, splits in _parents_to_splits(rows).items()
        if len(splits) > 1
    }
    if spanning:
        raise ValueError(f"Raster spans multiple splits: {spanning}")

    geometric_matches = {
        split: geometric[split] == EXPECTED_GEOMETRIC_TILES[split]
        for split in EXPECTED_GEOMETRIC_TILES
    }
    return {
        "status": "completed",
        "tile_size": tile,
        "max_black_fraction": max_black_fraction,
        "rasters": len(tokens),
        "geometric_tiles": dict(geometric),
        "expected_geometric_tiles": EXPECTED_GEOMETRIC_TILES,
        "geometric_matches_protocol": geometric_matches,
        "dropped_no_data_tiles": dict(dropped),
        "tiles_per_split": dict(per_split),
        "total_tiles": len(rows),
        "raster_report": raster_report,
        "split_assignment_source": (
            "docs/P17_GZCD_SECOND_POSITIVE_PROTOCOL.md, frozen before "
            "any tile was written"
        ),
        "spatial_disjointness": (
            "guaranteed by construction: every tile of a raster is in one "
            "split"
        ),
    }


def _parents_to_splits(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        mapping[row["parent_id"]].add(row["split"])
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tile GZ-CD rasters into a leakage-free dataset using the "
            "frozen raster-level split."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--max-black-fraction",
        type=float,
        default=0.5,
        help=(
            "Drop a tile when either temporal image is at least this "
            "fraction pure black, which marks mosaic gaps."
        ),
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build(
        Path(args.source_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
        tile=int(args.tile_size),
        max_black_fraction=float(args.max_black_fraction),
        manifest_path=Path(args.manifest),
        dry_run=bool(args.dry_run),
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "tiles_per_split",
            "geometric_tiles",
            "geometric_matches_protocol",
            "dropped_no_data_tiles",
            "total_tiles",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
