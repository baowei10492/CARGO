"""Probe: how far apart are the two dates in S2Looking, as seen by the frozen P30 corrector?

Read-only diagnosis for the paper's real-data question; not a gate and not a result table.
No labels are needed for the offsets; if a label folder is present it only feeds the
change-fraction bins. For every 256 px crop (4 x 4 grid per 1024 px tile, 64 px margin, as in
the test protocol) it records the corrector's estimate T1 -> T2, the reverse estimate, and the
estimate on an identical pair (T1, T1) as the noise floor. The same statistics are recomputed
from the frozen P30 test records (level 0) of LEVIR-CD, DSIFN-CD and WHU-CD for comparison.

Usage (repository root, on the machine that holds outputs/p30_localizer):
  python -m tools.probe_s2looking --s2looking-root /root/autodl-tmp/rs_cd_work/data/S2Looking \
      --split test --output-dir reports/probe_s2looking/$(date -u +%Y%m%dT%H%M%SZ)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from rs_cd.affine_registration import compose, grid_points, mean_displacement
from rs_cd.localizer_training import TrainedLocalizer, load_localizer

CROP = 256
MARGIN = 64
GRID = 4
THRESHOLDS = (1.0, 2.0, 5.0, 8.0, 25.0)
CHANGE_BINS = ((0.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01))
T1_NAMES = ("Image1", "image1", "A", "T1", "t1", "im1")
T2_NAMES = ("Image2", "image2", "B", "T2", "t2", "im2")
LABEL_NAMES = ("label", "Label", "OUT", "gt", "mask")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def find_folder(split_dir: Path, names: Sequence[str], required: bool) -> Path | None:
    for name in names:
        if (split_dir / name).is_dir():
            return split_dir / name
    if required:
        raise SystemExit(f"none of {names} under {split_dir}")
    return None


def list_images(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES}


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def crop_origins(height: int, width: int) -> list[tuple[int, int]]:
    ys = np.linspace(MARGIN, height - MARGIN - CROP, GRID).round().astype(int)
    xs = np.linspace(MARGIN, width - MARGIN - CROP, GRID).round().astype(int)
    return [(int(y), int(x)) for y in ys for x in xs]


def log_scale(matrix: np.ndarray) -> float:
    scale = math.sqrt(abs(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]))
    return abs(math.log(scale)) if scale > 0 else math.inf


def describe(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    out = {"n": int(v.size), "q25": float(np.quantile(v, 0.25)), "q50": float(np.quantile(v, 0.5)),
           "q75": float(np.quantile(v, 0.75)), "q90": float(np.quantile(v, 0.9))}
    out.update({f"share_above_{t:g}px": float((v > t).mean()) for t in THRESHOLDS})
    return out


def probe(localizer: TrainedLocalizer, root: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    split_dir = root / split
    t1_dir, t2_dir = find_folder(split_dir, T1_NAMES, True), find_folder(split_dir, T2_NAMES, True)
    label_dir = find_folder(split_dir, LABEL_NAMES, False)
    t1s, t2s = list_images(t1_dir), list_images(t2_dir)
    labels = list_images(label_dir) if label_dir else {}
    names = sorted(set(t1s) & set(t2s))
    if not names:
        raise SystemExit(f"no matching file names in {t1_dir} and {t2_dir}")
    if limit:
        names = names[:limit]
    points = grid_points(CROP, CROP, 17)
    rows = []
    for index, name in enumerate(names, 1):
        a, b = read_rgb(t1s[name]), read_rgb(t2s[name])
        if a.shape != b.shape:
            raise SystemExit(f"{name}: T1 {a.shape} and T2 {b.shape} differ")
        mask = read_mask(labels[name]) if name in labels else None
        origins = crop_origins(a.shape[0], a.shape[1])
        c1 = [a[y:y + CROP, x:x + CROP] for y, x in origins]
        c2 = [b[y:y + CROP, x:x + CROP] for y, x in origins]
        forward = localizer.estimate_many(c1, c2)
        backward = localizer.estimate_many(c2, c1)
        identical = localizer.estimate_many(c1, c1)
        for (y, x), f, r, s in zip(origins, forward, backward, identical):
            rows.append({
                "tile": name, "y": y, "x": x,
                "change_fraction": None if mask is None else float(mask[y:y + CROP, x:x + CROP].mean()),
                "converged": f.converged,
                "estimate_displacement_px": mean_displacement(f.matrix, points),
                "log_scale_magnitude": log_scale(f.matrix),
                "forward_backward_px": mean_displacement(compose(r.matrix, f.matrix), points),
                "misaligned_probability": f.misaligned_probability,
                "identical_pair_px": mean_displacement(s.matrix, points),
            })
        if index % 100 == 0:
            print(f"{index}/{len(names)} tiles", flush=True)
    return rows


def reference_rows(records_dir: Path) -> dict[str, list[dict[str, str]]]:
    out = {}
    for dataset in ("LEVIR-CD", "DSIFN-CD", "WHU-CD"):
        path = records_dir / f"records_proposed_{dataset}.csv"
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                out[dataset] = [row for row in csv.DictReader(handle) if row["level"] == "0"]
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = {key: describe([float(r[key]) for r in rows])
               for key in ("estimate_displacement_px", "forward_backward_px", "log_scale_magnitude")}
    summary["misaligned_probability_above_0.5"] = float(np.mean([float(r["misaligned_probability"]) > 0.5
                                                                 for r in rows]))
    return summary


def by_change(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or rows[0].get("change_fraction") in (None, ""):
        return []
    out = []
    for low, high in CHANGE_BINS:
        members = [r for r in rows if (float(r["change_fraction"]) == 0.0 if high == 0.0
                                       else low < float(r["change_fraction"]) <= high)]
        out.append({"bin": f"{low:g}-{high:g}" if high else "none",
                    "displacement_px": describe([float(r["estimate_displacement_px"]) for r in members])})
    return out


def render(result: dict[str, Any]) -> str:
    def line(label: str, stats: dict[str, Any]) -> str:
        if not stats.get("n"):
            return f"  {label:28s} n=0"
        shares = " ".join(f">{t:g}px {stats[f'share_above_{t:g}px']:.3f}" for t in THRESHOLDS)
        return (f"  {label:28s} n={stats['n']:6d}  q25/q50/q75/q90 {stats['q25']:.2f}/{stats['q50']:.2f}/"
                f"{stats['q75']:.2f}/{stats['q90']:.2f}  {shares}")

    lines = ["estimated displacement on unperturbed pairs (px, mean over a 17x17 grid of a 256 px crop)"]
    lines.append(line(f"S2Looking {result['split']}", result["s2looking"]["estimate_displacement_px"]))
    lines.append(line("S2Looking identical (T1,T1)", result["s2looking_identical"]))
    for name, summary in result["reference"].items():
        lines.append(line(f"{name} test (P30 records)", summary["estimate_displacement_px"]))
    lines.append("forward-backward inconsistency (px)")
    lines.append(line(f"S2Looking {result['split']}", result["s2looking"]["forward_backward_px"]))
    for name, summary in result["reference"].items():
        lines.append(line(f"{name} test", summary["forward_backward_px"]))
    lines.append("misaligned probability > 0.5: S2Looking "
                 f"{result['s2looking']['misaligned_probability_above_0.5']:.3f}; "
                 + "; ".join(f"{n} {s['misaligned_probability_above_0.5']:.3f}" for n, s in result["reference"].items()))
    if result["s2looking_by_change"]:
        lines.append("S2Looking displacement by change fraction of the crop")
        for entry in result["s2looking_by_change"]:
            lines.append(line(f"change {entry['bin']}", entry["displacement_px"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s2looking-root", required=True, help="folder holding train/ val/ test/")
    parser.add_argument("--split", default="test")
    parser.add_argument("--localizer-pointer", default="outputs/p30_localizer/LATEST")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--records-pointer", default="reports/p30/test/LATEST",
                        help="P30 test run whose level-0 records give the reference rows")
    parser.add_argument("--limit", type=int, default=None, help="first N tiles only (smoke run)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_dir = Path(Path(args.localizer_pointer).read_text(encoding="utf-8").strip())
    checkpoint = run_dir / "checkpoints" / args.checkpoint
    device = torch.device(args.device)
    localizer = TrainedLocalizer(load_localizer(str(checkpoint), device), device)

    start = time.perf_counter()
    rows = probe(localizer, Path(args.s2looking_root), args.split, args.limit)
    pointer = Path(args.records_pointer)
    records_dir = Path(pointer.read_text(encoding="utf-8").strip()) if pointer.is_file() else pointer
    reference = {name: summarize(r) for name, r in reference_rows(records_dir).items()}
    result = {
        "purpose": "read-only probe of real offsets in S2Looking; not a gate",
        "checkpoint": str(checkpoint), "split": args.split, "tiles": len({r["tile"] for r in rows}),
        "crops": len(rows), "seconds": time.perf_counter() - start,
        "s2looking": summarize(rows),
        "s2looking_identical": describe([r["identical_pair_px"] for r in rows]),
        "s2looking_by_change": by_change(rows),
        "reference": reference,
    }
    with (output_dir / "crops.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = render(result)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
