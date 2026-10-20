from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from rs_cd.datasets.manifest import build_manifest


def _make_sample(
    rng: np.random.Generator, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = rng.normal(80, 10, size=(size, size, 3)).clip(0, 255)
    gradient_x = np.linspace(0, 30, size, dtype=np.float32)[None, :, None]
    gradient_y = np.linspace(0, 20, size, dtype=np.float32)[:, None, None]
    t1 = (base + gradient_x + gradient_y).clip(0, 255)
    t2 = t1.copy()

    # Common unchanged object in both views.
    common_size = max(6, size // 6)
    cx = int(rng.integers(2, max(3, size - common_size - 2)))
    cy = int(rng.integers(2, max(3, size - common_size - 2)))
    common_color = rng.integers(110, 180, size=3)
    t1[cy : cy + common_size, cx : cx + common_size] = common_color
    t2[cy : cy + common_size, cx : cx + common_size] = common_color

    # Added building-like rectangle defines the change mask.
    change_w = int(rng.integers(max(5, size // 8), max(6, size // 3)))
    change_h = int(rng.integers(max(5, size // 8), max(6, size // 3)))
    x0 = int(rng.integers(1, max(2, size - change_w - 1)))
    y0 = int(rng.integers(1, max(2, size - change_h - 1)))
    t2[y0 : y0 + change_h, x0 : x0 + change_w] = rng.integers(
        190, 245, size=3
    )
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[y0 : y0 + change_h, x0 : x0 + change_w] = 255

    # Mild global appearance difference should not be labeled as change.
    t2 = (t2 * float(rng.uniform(0.96, 1.04))).clip(0, 255)
    return t1.astype(np.uint8), t2.astype(np.uint8), mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic CD dataset.")
    parser.add_argument("--output-root", default="data/synthetic")
    parser.add_argument("--manifest", default="manifests/datasets/synthetic.csv")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--train", type=int, default=8)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate only the explicitly selected synthetic root.",
    )
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Synthetic root already exists: {root}. Use --overwrite."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    counts = {"train": args.train, "val": args.val, "test": args.test}
    for split, count in counts.items():
        for name in ("A", "B", "label"):
            (root / split / name).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            sample_id = f"{split}_{index:04d}"
            t1, t2, mask = _make_sample(rng, args.size)
            Image.fromarray(t1, mode="RGB").save(root / split / "A" / f"{sample_id}.png")
            Image.fromarray(t2, mode="RGB").save(root / split / "B" / f"{sample_id}.png")
            Image.fromarray(mask, mode="L").save(
                root / split / "label" / f"{sample_id}.png"
            )

    rows = build_manifest(root, args.manifest)
    print(f"Created {len(rows)} synthetic paired samples at {root}")
    print(f"Manifest: {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()

