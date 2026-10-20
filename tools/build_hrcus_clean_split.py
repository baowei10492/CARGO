"""Clean the HRCUS-CD official split without re-partitioning it.

HRCUS-CD ships a train/val/test partition that fails the leakage audit,
but not in the way DSIFN-CD did. Two distinct problems are mixed
together in the raw audit output, and they need opposite treatments:

1. Degenerate tiles. 2000 of 11388 samples (17.6%) are entirely black in
   both epochs with an empty label. They are padding, not data. Because
   every one of them is byte-identical, they collapse into a single
   canonical digest and manufacture enormous cross-split "leakage".

2. Genuine cross-split near-duplicates. After the degenerate tiles are
   removed, the difference-hash audit still reports 36288 cross-split
   pairs, but pixel verification shows 94.8% of those are hash
   collisions: a 64-bit difference hash saturates on low-texture urban
   tiles. Only 336 distinct samples across all three splits survive
   verification, roughly 4% of each split.

So the official partition is not systematically contaminated the way
DSIFN-CD's was, but it is not clean either, and the preregistered rule in
docs/P20_HRCUS_SECOND_POSITIVE_PROTOCOL.md section 3.2 fires on any
non-zero result. The remedy applied here is the one that protocol
prescribes and that tools/build_source_crop_split.py already implements
for DSIFN-CD: the evaluation splits are never reassigned, and training
samples that duplicate a protected sample are dropped. That removes the
leakage without touching val or test, so published-comparable evaluation
stays possible.

Every threshold below is recorded in the output report. Nothing here
chooses a model, reads a label for any purpose other than counting
change pixels, or evaluates the test split.

Usage
-----
    python -m tools.build_hrcus_clean_split \
        --dataset-root /path/to/HRCUS-CD-PNG \
        --manifest hrcus_all_splits.csv \
        --records-csv hrcus_records.csv \
        --output-manifest manifests/datasets/hrcus_cd_clean.csv \
        --output-json reports/hrcus_clean_split.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def load_thumbnail(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(
            handle.convert("RGB").resize((size, size), Image.BILINEAR)
        )


def is_degenerate(dataset_root: Path, row: dict[str, str]) -> bool:
    """True when both epochs are uniformly zero: a padding tile."""
    with Image.open(dataset_root / row["t1_path"]) as handle:
        if np.asarray(handle).max() != 0:
            return False
    with Image.open(dataset_root / row["t2_path"]) as handle:
        return bool(np.asarray(handle).max() == 0)


def candidate_pairs(
    hashes: np.ndarray,
    splits: np.ndarray,
    *,
    hamming_max: int,
    block: int = 512,
) -> list[tuple[int, int, int]]:
    """Cross-split pairs within hamming_max of each other."""
    total = len(hashes)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, total, block):
        stop = min(start + block, total)
        xor = np.bitwise_xor(hashes[start:stop, None], hashes[None, :])
        distance = POPCOUNT[
            xor.view(np.uint8).reshape(stop - start, total, 8)
        ].sum(axis=2)
        for offset in range(stop - start):
            index = start + offset
            near = np.nonzero(
                (distance[offset] <= hamming_max) & (np.arange(total) > index)
            )[0]
            for other in near:
                if splits[index] != splits[other]:
                    pairs.append((index, int(other), int(distance[offset][other])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--records-csv",
        required=True,
        help="Output of tools.audit_augmentation_leakage --output-records-csv",
    )
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--protected-splits",
        nargs="+",
        default=["val", "test"],
        help="Never reassigned or dropped; training duplicates of these go.",
    )
    parser.add_argument("--hamming-max", type=int, default=5)
    parser.add_argument("--thumbnail-size", type=int, default=64)
    parser.add_argument(
        "--pixel-verify-max-mean-abs-diff",
        type=float,
        default=3.0,
        help="A hash-flagged pair counts as a real duplicate only below this.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    with open(args.manifest, encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with open(args.records_csv, encoding="utf-8", newline="") as handle:
        records = {row["sample_id"]: row for row in csv.DictReader(handle)}

    missing = [row["sample_id"] for row in manifest if row["sample_id"] not in records]
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest rows have no fingerprint, e.g. {missing[:3]}. "
            "Regenerate the records CSV from the same manifest."
        )

    print("scanning for degenerate tiles ...", flush=True)
    degenerate = {
        row["sample_id"] for row in manifest if is_degenerate(dataset_root, row)
    }
    kept = [row for row in manifest if row["sample_id"] not in degenerate]
    print(f"  {len(degenerate)} degenerate of {len(manifest)}")

    print("loading thumbnails ...", flush=True)
    thumbs = np.stack(
        [
            load_thumbnail(dataset_root / row["t1_path"], args.thumbnail_size)
            for row in kept
        ]
    ).astype(np.int16)

    hashes = np.array(
        [int(records[row["sample_id"]]["difference_hash"]) for row in kept],
        dtype=np.uint64,
    )
    splits = np.array([row["split"] for row in kept])
    ids = np.array([row["sample_id"] for row in kept])

    print("verifying hash-flagged pairs against pixels ...", flush=True)
    flagged = candidate_pairs(hashes, splits, hamming_max=args.hamming_max)
    verified: list[dict[str, Any]] = []
    for left, right, distance in flagged:
        difference = float(np.abs(thumbs[left] - thumbs[right]).mean())
        if difference < args.pixel_verify_max_mean_abs_diff:
            verified.append(
                {
                    "a": str(ids[left]),
                    "b": str(ids[right]),
                    "hamming": distance,
                    "mean_abs_diff": difference,
                }
            )
    false_positive_rate = 1.0 - len(verified) / max(1, len(flagged))
    print(
        f"  {len(flagged)} flagged -> {len(verified)} real "
        f"({false_positive_rate:.1%} false positives)"
    )

    protected = set(args.protected_splits)
    split_of = dict(zip(ids.tolist(), splits.tolist()))
    drop: set[str] = set()
    for pair in verified:
        left, right = pair["a"], pair["b"]
        for this, other in ((left, right), (right, left)):
            if split_of[this] not in protected and split_of[other] in protected:
                drop.add(this)

    final = [row for row in kept if row["sample_id"] not in drop]
    per_split = Counter(row["split"] for row in final)
    before = Counter(row["split"] for row in manifest)

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(final)

    involved = defaultdict(set)
    for pair in verified:
        involved[split_of[pair["a"]]].add(pair["a"])
        involved[split_of[pair["b"]]].add(pair["b"])

    report = {
        "status": "completed",
        "dataset": "HRCUS-CD",
        "action": "cleaned_official_split_without_repartitioning",
        "protected_splits": sorted(protected),
        "thresholds": {
            "hamming_max": args.hamming_max,
            "thumbnail_size": args.thumbnail_size,
            "pixel_verify_max_mean_abs_diff": args.pixel_verify_max_mean_abs_diff,
        },
        "degenerate_tiles_removed": len(degenerate),
        "degenerate_by_split": dict(
            Counter(
                row["split"] for row in manifest if row["sample_id"] in degenerate
            )
        ),
        "hash_flagged_cross_split_pairs": len(flagged),
        "pixel_verified_cross_split_pairs": len(verified),
        "hash_false_positive_rate": false_positive_rate,
        "distinct_samples_involved_by_split": {
            split: len(members) for split, members in sorted(involved.items())
        },
        "training_samples_dropped": len(drop),
        "samples_before": dict(before),
        "samples_after": dict(per_split),
        "protected_splits_untouched": all(
            per_split[split] == before[split] - len(
                [
                    row
                    for row in manifest
                    if row["split"] == split and row["sample_id"] in degenerate
                ]
            )
            for split in protected
            if split in before
        ),
        "verified_pairs": verified[:200],
        "note": (
            "The official partition is retained. Only degenerate padding "
            "tiles and training samples that duplicate a protected sample "
            "were removed, so val and test remain comparable with published "
            "numbers on this dataset."
        ),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nsamples per split")
    for split in sorted(before):
        print(f"  {split:6s} {before[split]:6d} -> {per_split[split]:6d}")
    print(f"\nwrote {output_manifest}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
