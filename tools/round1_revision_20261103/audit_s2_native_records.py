"""CPU-only audit of saved native S2Looking predictions; no model loading.

Read original per-crop confusion counts and scores, verify the saved full-test
F1 anchors, then bootstrap paired source tiles with a fixed selector threshold.
Run manually on the server; this script performs no SSH, downloads or deletion.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def f1(counts):
    counts = np.asarray(counts)
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator,
                     out=np.zeros_like(denominator, dtype=np.float64),
                     where=denominator != 0)


def write_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path("reports/p33/selector/v2/dinov3_s2looking")
    ap.add_argument("--records", type=Path, default=base / "records_S2Looking.csv")
    ap.add_argument("--anchor", type=Path, default=base / "test_S2Looking.json")
    ap.add_argument("--output", type=Path,
                    default=Path("reports/round1_20261103/S2_native_audit"))
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    started = time.monotonic()
    if args.draws < 2:
        raise ValueError("at least two bootstrap draws required")
    if not args.records.is_file() or not args.anchor.is_file():
        raise FileNotFoundError("Original records and result JSON are required; no inference will be started.")
    anchor = json.loads(args.anchor.read_text(encoding="utf-8-sig"))
    if anchor.get("dataset") != "S2Looking":
        raise ValueError("wrong dataset in saved result anchor")
    tau = math.inf if anchor["tau"] is None else float(anchor["tau"])
    tiles = {}
    seen = set()
    crop_rows = []
    levels = {}
    with args.records.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"tile_id", "crop_id", "level", "selector_probability"}
        required.update(f"{path}_{key}" for path in ("raw", "estimated")
                        for key in ("tp", "fp", "fn", "tn"))
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV fields: {sorted(missing)}")
        for row in reader:
            level = int(row["level"])
            levels[level] = levels.get(level, 0) + 1
            if level != 0:
                continue
            key = (row["tile_id"], row["crop_id"])
            if key in seen:
                raise ValueError(f"duplicate native crop {key}")
            seen.add(key)
            score = float(row["selector_probability"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"invalid selector score at {key}")
            raw, estimated = [np.array([int(row[f"{path}_{key}"])
                                       for key in ("tp", "fp", "fn", "tn")], dtype=np.int64)
                              for path in ("raw", "estimated")]
            if (raw < 0).any() or (estimated < 0).any() or raw.sum() != estimated.sum():
                raise ValueError(f"inconsistent scoring counts at {key}")
            adopted = score >= tau
            cargo = estimated if adopted else raw
            entry = tiles.setdefault(row["tile_id"], {"counts": np.zeros((3, 4), dtype=np.int64),
                                                      "crops": 0, "adopted": 0})
            entry["counts"] += np.stack((raw, estimated, cargo))
            entry["crops"] += 1
            entry["adopted"] += int(adopted)
            crop_rows.append({"tile_id": row["tile_id"], "crop_id": row["crop_id"],
                              "selector_score": score, "adopted": int(adopted),
                              **{f"{name}_{key}": int(value)
                                 for name, counts in (("raw", raw), ("estimated", estimated), ("cargo", cargo))
                                 for key, value in zip(("tp", "fp", "fn", "tn"), counts)}})
    expected_crops = int(anchor["crops"])
    if len(seen) != expected_crops or len(tiles) != 1000 or any(v["crops"] != 16 for v in tiles.values()):
        raise ValueError(f"incomplete native test: {len(seen)} crops / {len(tiles)} tiles; expected {expected_crops}/1000 and 16 crops per tile")
    tile_ids = sorted(tiles)
    counts = np.stack([tiles[tile]["counts"] for tile in tile_ids])
    pooled = f1(counts.sum(axis=0))
    paths = ("raw", "estimated", "selector")
    anchor_differences = {}
    for index, name in enumerate(paths):
        difference = float(pooled[index]) - float(anchor["f1"]["0"][name])
        anchor_differences[name] = difference
        if abs(difference) > 1e-8:
            raise ValueError(f"saved F1 anchor mismatch for {name}: {pooled[index]} vs {anchor['f1']['0'][name]}")
    adopted = sum(v["adopted"] for v in tiles.values())
    rate = adopted / len(seen)
    if abs(rate - float(anchor["selection_rate"]["0"]["selector"])) > 1e-12:
        raise ValueError("saved adoption-rate anchor mismatch")
    print(f"ANCHORS_OK native crops={len(seen)} tiles={len(tiles)} tau={tau} adopted={adopted}", flush=True)
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty((args.draws, 3), dtype=np.float64)
    for start in range(0, args.draws, 128):
        stop = min(start + 128, args.draws)
        draws = rng.integers(0, len(tiles), size=(stop - start, len(tiles)))
        bootstrap[start:stop] = f1(counts[draws].sum(axis=1))
    differences = {}
    for name, first, second in (("cargo_minus_raw", 2, 0), ("cargo_minus_always", 2, 1)):
        distribution = 100 * (bootstrap[:, first] - bootstrap[:, second])
        differences[name] = {"estimate_pp": float(100 * (pooled[first] - pooled[second])),
                             "ci95_pp": np.quantile(distribution, [.025, .975]).tolist()}
    result = {"dataset": "S2Looking", "level": 0, "added_synthetic_perturbation": False,
              "crops": len(seen), "tiles": len(tiles), "tau": None if math.isinf(tau) else tau,
              "f1_percent": dict(zip(("raw", "always", "cargo"), (100 * pooled).tolist())),
              "adopted_crops": adopted, "adopted_percent": 100 * rate,
              "differences": differences,
              "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "source tile",
                            "paired_draws_shared_across_paths": True, "interval": "percentile 95%",
                            "scope": "test-tile sampling with fixed detector, registration module and selector checkpoints"},
              "counts_by_level_in_original_csv": levels,
              "anchor_f1_differences_fraction": anchor_differences,
              "sources": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in (args.records, args.anchor)},
              "does_not_measure": "independent geometric residuals or detector-training variance",
              "elapsed_seconds": time.monotonic() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    write_text(args.output / "summary.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = [f"Native S2Looking: {len(seen)} crops / {len(tiles)} source tiles; no synthetic perturbation.",
             "F1 (%): " + ", ".join(f"{name}={value:.5f}" for name, value in result["f1_percent"].items()),
             f"Adopted: {100*rate:.5f}% ({adopted}/{len(seen)}); fixed tau={tau}."]
    for name, item in differences.items():
        low, high = item["ci95_pp"]
        lines.append(f"{name}: {item['estimate_pp']:+.5f} pp [{low:+.5f}, {high:+.5f}]")
    lines += [f"95% percentile intervals from {args.draws} paired tile draws; fixed checkpoints.",
              "Independent geometric accuracy and detector-training variance are not measured."]
    write_text(args.output / "summary.txt", "\n".join(lines) + "\n")
    with (args.output / "native_crop_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(crop_rows[0]))
        writer.writeheader()
        writer.writerows(crop_rows)
    print("\n".join(lines), flush=True)
    print("COMPLETE", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
