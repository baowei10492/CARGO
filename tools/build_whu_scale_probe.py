from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rs_cd.runtime import save_json
from tools.build_whu_tiles import (
    EARLY_YEAR,
    LATE_YEAR,
    TWO_PERIOD_DIR,
    _load_raster,
)

Image.MAX_IMAGE_PIXELS = None

WHU_GSD_M = 0.075


def enumerate_blocks(
    x_max: int,
    y_max: int,
    scale: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Top-left corners of scale x scale blocks fully inside the region."""

    if scale <= 0 or stride <= 0:
        raise ValueError("scale and stride must be positive")
    if scale > x_max or scale > y_max:
        return []
    origins = []
    for top in range(0, y_max - scale + 1, stride):
        for left in range(0, x_max - scale + 1, stride):
            origins.append((left, top))
    return origins


def select_evenly(items: list[Any], cap: int) -> list[Any]:
    """Deterministically pick <= cap items spread across the list."""

    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(items) <= cap:
        return list(items)
    step = len(items) / cap
    return [items[int(index * step)] for index in range(cap)]


def blocks_for_scale(
    x_max: int,
    y_max: int,
    scale: int,
    *,
    max_blocks: int,
) -> tuple[list[tuple[int, int]], int]:
    """Non-overlapping grid, densified to half stride only when sparse.

    Preregistered rule: use stride = scale; if that yields fewer than
    max_blocks candidates, use stride = scale // 2 instead (accepting
    half-overlap correlation, which is reported via the stride field).
    Then cap with an even spread. Deterministic throughout.
    """

    stride = scale
    origins = enumerate_blocks(x_max, y_max, scale, stride)
    if len(origins) < max_blocks:
        stride = max(1, scale // 2)
        origins = enumerate_blocks(x_max, y_max, scale, stride)
    return select_evenly(origins, max_blocks), stride


def downsample_block(block: np.ndarray, output_size: int) -> np.ndarray:
    """Box-filter downsample: the physical pixel-binning model.

    Every probe scale is an integer multiple of the output size, so BOX
    resampling is exact block averaging - the standard model for a
    coarser sensor integrating light over a larger footprint. No
    sharpening, no aliasing tricks.
    """

    image = Image.fromarray(block)
    resized = image.resize(
        (output_size, output_size),
        Image.Resampling.BOX,
    )
    return np.asarray(resized)


def build_probe(
    raw_root: Path,
    output_root: Path,
    *,
    scales: list[int],
    output_size: int,
    max_blocks: int,
    x_max: int,
    y_max: int,
    dry_run: bool,
) -> dict[str, Any]:
    two_period = raw_root / TWO_PERIOD_DIR
    early = next(
        (
            two_period / EARLY_YEAR / "whole_image" / "train" / "image"
        ).glob("*.tif")
    )
    late = next(
        (
            two_period / LATE_YEAR / "whole_image" / "train" / "image"
        ).glob("*.tif")
    )
    print(f"A={early.name} B={late.name} (train raster only)")
    a_array = _load_raster(early, expect_rgb=True)
    b_array = _load_raster(late, expect_rgb=True)
    if a_array.shape != b_array.shape:
        raise ValueError(
            f"Raster shapes disagree: {a_array.shape} vs {b_array.shape}"
        )
    height, width = a_array.shape[:2]
    if x_max > width or y_max > height:
        raise ValueError(
            f"Requested region {x_max}x{y_max} exceeds raster "
            f"{width}x{height}"
        )

    for scale in scales:
        if scale % output_size != 0:
            raise ValueError(
                f"Scale {scale} is not an integer multiple of the "
                f"output size {output_size}; BOX downsampling would "
                "not be exact block averaging"
            )

    zero_label = np.zeros((output_size, output_size), dtype=np.uint8)
    index_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "raster": {"width": width, "height": height},
        "region": {"x_max": x_max, "y_max": y_max},
        "output_size": output_size,
        "gsd_m": WHU_GSD_M,
        "scales": {},
    }
    for scale in scales:
        origins, stride = blocks_for_scale(
            x_max,
            y_max,
            scale,
            max_blocks=max_blocks,
        )
        if not origins:
            raise ValueError(f"Region too small for scale {scale}")
        split = f"scale{scale:04d}"
        coverage_m = scale * WHU_GSD_M
        effective_gsd = coverage_m / output_size
        summary["scales"][split] = {
            "scale_px": scale,
            "stride_px": stride,
            "blocks": len(origins),
            "coverage_m": coverage_m,
            "effective_gsd_m": effective_gsd,
            "shift8_physical_m": 8.0 * effective_gsd,
        }
        print(
            f"[{split}] blocks={len(origins)} stride={stride} "
            f"coverage={coverage_m:.1f} m "
            f"effective_gsd={effective_gsd:.3f} m "
            f"shift8={8.0 * effective_gsd:.2f} m"
        )
        for left, top in origins:
            sample_id = f"whu_probe_s{scale:04d}_y{top:05d}_x{left:05d}"
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "scale": scale,
                    "left": left,
                    "top": top,
                    "coverage_m": f"{coverage_m:.3f}",
                }
            )
            if dry_run:
                continue
            for folder, source in (
                ("A", a_array),
                ("B", b_array),
            ):
                block = source[top : top + scale, left : left + scale]
                tile = downsample_block(block, output_size)
                path = output_root / split / folder / f"{sample_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile).save(path)
            # Placeholder zero labels: the probe computes no labeled
            # metric, and an all-zero mask makes any accidental F1 on
            # probe data obviously degenerate instead of plausibly wrong.
            label_path = (
                output_root / split / "label" / f"{sample_id}.png"
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(zero_label).save(label_path)

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "probe_index.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(index_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(index_rows)
        save_json(output_root / "probe_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the WHU scale/coverage probe: fixed 512 px outputs "
            "whose ground coverage varies by cutting larger blocks and "
            "box-downsampling them, simulating the same scene imaged "
            "by a coarser sensor. Train-raster region only (never the "
            "val strip), measured with the LEVIR P7C instrument which "
            "has never seen WHU."
        )
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 3072],
    )
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--max-blocks", type=int, default=60)
    parser.add_argument(
        "--train-x-max",
        type=int,
        default=17408,
        help=(
            "Exclusive x limit: train columns 0..33 end at 34*512. "
            "Blocks never touch the buffer or validation strips."
        ),
    )
    parser.add_argument(
        "--y-max",
        type=int,
        default=14848,
        help="Exclusive y limit: the audited 29-row grid ends at 29*512.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    build_probe(
        Path(args.raw_root).expanduser(),
        Path(args.output_root).expanduser(),
        scales=[int(scale) for scale in args.scales],
        output_size=int(args.output_size),
        max_blocks=int(args.max_blocks),
        x_max=int(args.train_x_max),
        y_max=int(args.y_max),
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
