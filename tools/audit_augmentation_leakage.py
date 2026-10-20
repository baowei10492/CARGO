from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


"""Detect augmentation-induced split leakage.

DSIFN-CD releases 3600 train / 340 val patches that were produced by
augmenting only 394 source crops, so the same source tile can appear in
both splits under a rotation or flip. That matters here because the
deployment threshold tau is chosen on validation as a max statistic
(main record 57.2 / 83): a validation set that is a rotated copy of the
fitting set understates tau.

The audit is deliberately label-free and geometry-only: it canonicalises
each pair over the dihedral group (4 rotations x optional mirror) and
compares canonical digests across splits. An exact digest match survives
any lossless flip/rotation augmentation; the perceptual difference hash
additionally catches re-encoded or mildly photometric variants.
"""


DIHEDRAL_TRANSFORMS = 8


def dihedral_variants(array: np.ndarray) -> Iterable[np.ndarray]:
    """Yield the 8 dihedral views of an image array."""
    for mirrored in (False, True):
        base = np.fliplr(array) if mirrored else array
        for rotation in range(4):
            yield np.ascontiguousarray(np.rot90(base, rotation))


def canonical_pair_digest(t1: np.ndarray, t2: np.ndarray) -> str:
    """SHA256 of the lexicographically smallest joint dihedral view.

    T1 and T2 are transformed jointly, so a pair that was augmented as a
    unit collapses onto the same digest regardless of which of the eight
    views was released into which split.
    """
    if t1.shape != t2.shape:
        raise ValueError(
            f"T1/T2 shape mismatch: {t1.shape} vs {t2.shape}"
        )
    best: bytes | None = None
    for first, second in zip(
        dihedral_variants(t1), dihedral_variants(t2), strict=True
    ):
        payload = first.tobytes() + second.tobytes()
        if best is None or payload < best:
            best = payload
    assert best is not None
    return hashlib.sha256(best).hexdigest()


def difference_hash(array: np.ndarray, *, size: int = 8) -> int:
    """64-bit perceptual difference hash of a single view."""
    image = Image.fromarray(array).convert("L").resize(
        (size + 1, size), Image.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def canonical_difference_hash(array: np.ndarray, *, size: int = 8) -> int:
    """Smallest difference hash over the 8 dihedral views."""
    return min(
        difference_hash(view, size=size)
        for view in dihedral_variants(array)
    )


def hamming_distance(first: int, second: int) -> int:
    return int(bin(first ^ second).count("1"))


def leakage_report(
    records: list[dict[str, Any]],
    *,
    near_duplicate_hamming_max: int = 5,
    example_limit: int = 20,
) -> dict[str, Any]:
    """Summarise exact and near-duplicate overlap between splits."""
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_digest[str(record["canonical_digest"])].append(record)

    cross_groups: list[dict[str, Any]] = []
    within_groups: list[dict[str, Any]] = []
    for digest, group in by_digest.items():
        if len(group) < 2:
            continue
        splits = sorted({str(item["split"]) for item in group})
        entry = {
            "canonical_digest": digest,
            "splits": splits,
            "sample_ids": sorted(str(item["sample_id"]) for item in group),
        }
        if len(splits) > 1:
            cross_groups.append(entry)
        else:
            within_groups.append(entry)

    exact_cross_split_samples = sum(
        len(group["sample_ids"]) for group in cross_groups
    )
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["split"])] += 1

    near_pairs: list[dict[str, Any]] = []
    exact_pairs = {
        (first, second)
        for group in cross_groups
        for first in group["sample_ids"]
        for second in group["sample_ids"]
        if first < second
    }
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record["split"])].append(record)
    split_names = sorted(by_split)
    for index, left_split in enumerate(split_names):
        for right_split in split_names[index + 1:]:
            for left in by_split[left_split]:
                for right in by_split[right_split]:
                    distance = hamming_distance(
                        int(left["difference_hash"]),
                        int(right["difference_hash"]),
                    )
                    if distance > near_duplicate_hamming_max:
                        continue
                    pair = tuple(
                        sorted(
                            (
                                str(left["sample_id"]),
                                str(right["sample_id"]),
                            )
                        )
                    )
                    if pair in exact_pairs:
                        continue
                    near_pairs.append(
                        {
                            "sample_ids": list(pair),
                            "splits": [left_split, right_split],
                            "hamming_distance": distance,
                        }
                    )

    official_split_usable = (
        not cross_groups and not near_pairs
    )
    # Aggregate over every group and pair, not just the stored examples:
    # the examples are truncated, so any claim about which splits are
    # affected must come from the full counters.
    exact_by_pair: dict[str, int] = defaultdict(int)
    for group in cross_groups:
        exact_by_pair["+".join(group["splits"])] += 1
    near_by_pair: dict[str, int] = defaultdict(int)
    distance_histogram: dict[str, int] = defaultdict(int)
    for pair in near_pairs:
        near_by_pair["+".join(sorted(pair["splits"]))] += 1
        distance_histogram[str(pair["hamming_distance"])] += 1
    splits_involved = sorted(
        {split for group in cross_groups for split in group["splits"]}
        | {split for pair in near_pairs for split in pair["splits"]}
    )
    return {
        "status": "completed",
        "samples": len(records),
        "samples_per_split": dict(sorted(counts.items())),
        "near_duplicate_hamming_max": near_duplicate_hamming_max,
        "cross_split_exact_duplicate_groups": len(cross_groups),
        "cross_split_exact_duplicate_samples": exact_cross_split_samples,
        "cross_split_near_duplicate_pairs": len(near_pairs),
        "within_split_exact_duplicate_groups": len(within_groups),
        "cross_split_exact_groups_by_split_pair": dict(
            sorted(exact_by_pair.items())
        ),
        "cross_split_near_pairs_by_split_pair": dict(
            sorted(near_by_pair.items())
        ),
        "near_pair_hamming_histogram": dict(
            sorted(distance_histogram.items(), key=lambda kv: int(kv[0]))
        ),
        "splits_involved_in_leakage": splits_involved,
        "example_limit": example_limit,
        "cross_split_exact_duplicate_examples": cross_groups[
            :example_limit
        ],
        "cross_split_near_duplicate_examples": near_pairs[:example_limit],
        "decision": {
            "official_split_usable": official_split_usable,
            "reason": (
                "no cross-split duplicate or near-duplicate detected"
                if official_split_usable
                else (
                    "augmented copies of the same source crop appear in "
                    "more than one split; tau must not be calibrated on "
                    "the official validation split"
                )
            ),
            "required_action": (
                "use official splits"
                if official_split_usable
                else (
                    "regroup by canonical_digest and re-split by source "
                    "crop, then declare it a dataset-specific split"
                )
            ),
        },
    }


def _load_array(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def collect_records(
    dataset_root: Path,
    manifest: Path,
    *,
    splits: tuple[str, ...],
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in splits:
                continue
            t1 = _load_array(dataset_root / row["t1_path"])
            t2 = _load_array(dataset_root / row["t2_path"])
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "canonical_digest": canonical_pair_digest(t1, t2),
                    "difference_hash": canonical_difference_hash(t1),
                }
            )
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise RuntimeError("Manifest produced no records for these splits")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether augmented copies of the same source crop are "
            "split across train/val/test. Reads images only, never "
            "labels, and computes no performance metric."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"]
    )
    parser.add_argument(
        "--near-duplicate-hamming-max", type=int, default=5
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--output-records-csv",
        default=None,
        help=(
            "Cache per-sample digests so the re-split tool does not have "
            "to decode every image again."
        ),
    )
    parser.add_argument(
        "--fail-on-leakage",
        action="store_true",
        help="Exit non-zero when the official splits are unusable.",
    )
    args = parser.parse_args()

    records = collect_records(
        Path(args.dataset_root).expanduser().resolve(),
        Path(args.manifest).expanduser().resolve(),
        splits=tuple(args.splits),
        max_samples=args.max_samples,
    )
    report = leakage_report(
        records,
        near_duplicate_hamming_max=int(args.near_duplicate_hamming_max),
        example_limit=int(args.example_limit),
    )
    if args.output_records_csv:
        records_csv = Path(args.output_records_csv)
        records_csv.parent.mkdir(parents=True, exist_ok=True)
        with records_csv.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "split",
                    "canonical_digest",
                    "difference_hash",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
        report["records_csv"] = str(records_csv)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_leakage and not report["decision"][
        "official_split_usable"
    ]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
