"""Dose-response: how much pasted change it takes to pull the registration off (main record 192.71).

Self-pairs remove the real pair's own residual offset: the reference is the T1 crop, the target is the same T1 tile
with rectangles of another tile's T1 (same place) pasted into the crop window until they cover a set fraction of
it, then perturbed as in the border-free protocol (whole tile warped, crop cut; level-3 draw of the crop) and also
left aligned. The true transform is therefore exact, and every registration error comes from the pasted content.
Rectangles, draws and crops are the same for every localizer. Test tiles, evenly spaced margin crops.
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.localizer_training import step1_rows
from rs_cd.p30_geometry import BorderFreeGeometry
from tools import run_p30
from tools.p31_localizer_ablation import build
from tools.run_p27_selector import harness, open_dataset, with_split
from tools.run_p31_plugins import load

VARIANTS = ("v2_full", "v2_no_reliability", "roma_similarity")
FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
CROPS_PER_DATASET = 256
SIDE = (24, 96)                  # rectangle sides, px
SUCCESS_PX = 2.0
FIELDS = ("variant", "dataset", "crop_id", "fraction", "coverage", "level", "ate_raw", "estimate_displacement_px",
          "misaligned_probability", "trigger", "converged")


def rectangles(crop_id: str, fraction: float, size: int, seed: int) -> np.ndarray:
    """Boolean (size, size) mask of random rectangles covering at least ``fraction`` of the window."""
    covered = np.zeros((size, size), dtype=bool)
    if fraction <= 0:
        return covered
    rng = np.random.default_rng(seed + zlib.crc32(f"{crop_id}|{fraction:.2f}".encode()))
    while covered.mean() < fraction:
        height, width = (int(v) for v in rng.integers(SIDE[0], SIDE[1] + 1, size=2))
        top, left = int(rng.integers(0, size - height + 1)), int(rng.integers(0, size - width + 1))
        covered[top:top + height, left:left + width] = True
    return covered


def pasted_tile(tile: np.ndarray, source: np.ndarray, top: int, left: int, covered: np.ndarray) -> np.ndarray:
    """``tile`` with ``source`` pasted where ``covered`` (crop-window coordinates)."""
    size = covered.shape[0]
    out = tile.copy()
    window = out[top:top + size, left:left + size]
    window[covered] = source[top:top + size, left:left + size][covered]
    return out


def evenly(items: Sequence[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return list(items)
    return [items[int(k * len(items) / count)] for k in range(count)]


def other_index(crops: Sequence[dict[str, Any]], index: int) -> int:
    """A crop of another tile, half the list away (as tools.p27_diffregcd_rerun.Views.other)."""
    count = len(crops)
    for step in range(count // 2, count // 2 + count):
        candidate = (index + step) % count
        if crops[candidate]["row"] != crops[index]["row"]:
            return candidate
    raise SystemExit("dose-response needs crops from at least two tiles")


@torch.no_grad()
def dataset_rows(variant: str, spec: dict[str, Any], evaluation: dict[str, Any], localizer, *, margin: int,
                 crops_per_dataset: int = CROPS_PER_DATASET, fractions: Sequence[float] = FRACTIONS) -> list[dict[str, Any]]:
    h = harness(evaluation)
    size, top_level = h["size"], h["levels"][-1]
    dataset = open_dataset(spec, spec["split"])
    crops = evenly(run_p30.margin_crops(dataset, spec, size, margin), crops_per_dataset)
    cache, other_cache = TileCache(dataset, int(spec["tile_size"])), TileCache(dataset, int(spec["tile_size"]))
    out = []
    for index, item in enumerate(crops):
        t1 = cache.crop(item, size)[0]
        tile = cache.arrays[0]
        other_cache.crop(crops[other_index(crops, index)], size)
        source = other_cache.arrays[0]
        if source.shape != tile.shape:
            raise SystemExit(f"{spec['name']}: tiles of different sizes")
        for fraction in fractions:
            covered = rectangles(item["crop_id"], fraction, size, int(evaluation["seed"]))
            target_tile = pasted_tile(tile, source, item["top"], item["left"], covered)
            geometry = BorderFreeGeometry(target_tile, item["top"], item["left"], size)
            target = target_tile[item["top"]:item["top"] + size, item["left"]:item["left"] + size]
            rows = step1_rows(localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=target,
                              change_fraction=float(covered.mean()), ranges=h["ranges"], levels=[top_level],
                              seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
                              trigger_probability=h["trigger_probability"], make_target=geometry.perturb)
            for row in rows:
                out.append({"variant": variant, "dataset": spec["name"], "crop_id": item["crop_id"],
                            "fraction": fraction, "coverage": float(covered.mean()),
                            **{k: row[k] for k in FIELDS[5:]}})
        if (index + 1) % 64 == 0 or index + 1 == len(crops):
            print(f"[dose {variant} {spec['name']} {index + 1}/{len(crops)}]", flush=True)
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per variant, dataset (and 'all'), level and fraction: success rate, median and mean transfer error."""
    out = []
    keys = sorted({(r["variant"], r["dataset"], int(r["level"]), float(r["fraction"])) for r in rows})
    groups: dict[tuple, list[float]] = {}
    for variant, name, level, fraction in keys:
        errors = [float(r["ate_raw"]) for r in rows if (r["variant"], r["dataset"], int(r["level"]), float(r["fraction"]))
                  == (variant, name, level, fraction)]
        groups[(variant, name, level, fraction)] = errors
        groups.setdefault((variant, "all", level, fraction), []).extend(errors)
    for (variant, name, level, fraction), errors in sorted(groups.items()):
        values = np.array(errors)
        out.append({"variant": variant, "dataset": name, "level": level, "fraction": fraction, "pairs": len(values),
                    "success": float((values <= SUCCESS_PX).mean()), "median_error": float(np.median(values)),
                    "mean_error": float(values.mean())})
    return out


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p31_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--crops-per-dataset", type=int, default=CROPS_PER_DATASET)
    parser.add_argument("--output-dir", required=True, help="resumable: finished variants are kept")
    args = parser.parse_args()
    config, p30, evaluation, _ = load(Path(args.config))
    evaluation = with_split(evaluation, config["ablation"]["split"])
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        path = output_dir / f"dose_{variant}.csv"
        if not path.is_file():
            torch.manual_seed(int(evaluation["seed"]))
            localizer = build(variant, p30, evaluation, device)
            chunk = [r for spec in evaluation["datasets"]
                     for r in dataset_rows(variant, spec, evaluation, localizer, margin=int(p30["crop_margin"]),
                                           crops_per_dataset=args.crops_per_dataset)]
            write_rows(path, chunk)
            del localizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    table = summarize(rows)
    write_rows(output_dir / "dose_summary.csv", table)
    (output_dir / "dose_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for entry in table:
        if entry["dataset"] == "all":
            print(f"{entry['variant']:<18} L{entry['level']} fraction {entry['fraction']:.1f}: success "
                  f"{100 * entry['success']:5.1f}%  median {entry['median_error']:6.2f} px")


if __name__ == "__main__":
    main()
