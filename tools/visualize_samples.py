from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rs_cd.datasets.manifest import read_manifest


def _fit(image: Image.Image, size: int, resample: Image.Resampling) -> Image.Image:
    return image.resize((size, size), resample=resample)


def _difference(t1: Image.Image, t2: Image.Image) -> Image.Image:
    left = np.asarray(t1, dtype=np.int16)
    right = np.asarray(t2, dtype=np.int16)
    difference = np.abs(left - right).mean(axis=2)
    scale = max(1.0, float(np.percentile(difference, 99)))
    normalized = np.clip(difference / scale * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L").convert("RGB")


def _overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    color = np.zeros_like(base)
    color[..., 0] = 255
    alpha = binary[..., None].astype(np.float32) * 0.45
    blended = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paired-data contact sheet.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    rows = read_manifest(args.manifest, split=args.split)
    rng = random.Random(args.seed)
    selected = rng.sample(rows, min(args.count, len(rows)))
    labels = ("T1", "T2", "Mask", "|T1-T2|", "Mask overlay")
    header_h = 24
    id_h = 24
    width = len(labels) * args.tile_size
    height = header_h + len(selected) * (args.tile_size + id_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for column, label in enumerate(labels):
        draw.text(
            (column * args.tile_size + 6, 6),
            label,
            fill="black",
            font=font,
        )

    for row_index, row in enumerate(selected):
        with Image.open(root / row.t1_path) as image:
            t1 = image.convert("RGB")
        with Image.open(root / row.t2_path) as image:
            t2 = image.convert("RGB")
        with Image.open(root / row.mask_path) as image:
            mask = image.convert("L")
        panels = (
            t1,
            t2,
            mask.convert("RGB"),
            _difference(t1, t2),
            _overlay(t2, mask),
        )
        y0 = header_h + row_index * (args.tile_size + id_h)
        for column, panel in enumerate(panels):
            fitted = _fit(
                panel,
                args.tile_size,
                Image.Resampling.NEAREST
                if column == 2
                else Image.Resampling.BILINEAR,
            )
            canvas.paste(fitted, (column * args.tile_size, y0))
        draw.text(
            (6, y0 + args.tile_size + 5),
            f"{row.split}: {row.sample_id}",
            fill="black",
            font=font,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()

