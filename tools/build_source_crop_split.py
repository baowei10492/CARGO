from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.audit_augmentation_leakage import hamming_distance


"""Rebuild a train/val split that respects source crops.

DSIFN-CD augments 394 source crops into 3940 patches and then splits the
augmented pool, so rotated copies of one crop land in both train and
val. Calibrating tau on such a validation split understates it, because
tau is a max statistic over clean calibration samples (main record 57.2
and 83).

This tool groups patches into source-crop clusters (identical dihedral
digest, or perceptually near-identical), assigns every cluster as a unit,
and writes a manifest whose parent_id is the cluster id so the existing
ID-hash partitioning in the router pipeline keeps clusters together.

The protected split (test by default) is never reassigned: comparability
with published numbers on the official test set matters more than
cluster balance. Instead, training patches that share a cluster with a
protected sample are dropped, which removes the leakage without touching
the evaluation set.
"""


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self._parent[right] = left


def build_clusters(
    records: list[dict[str, Any]],
    *,
    cluster_hamming_max: int,
) -> dict[str, str]:
    """Map sample_id -> cluster_id."""
    ids = [str(record["sample_id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Duplicate sample_id in records: the manifest reuses one id "
            "for more than one image pair. Rebuild it with "
            "--sample-id-mode split_stem and re-run the audit."
        )
    union = UnionFind(ids)

    by_digest: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_digest[str(record["canonical_digest"])].append(
            str(record["sample_id"])
        )
    for group in by_digest.values():
        for other in group[1:]:
            union.union(group[0], other)

    if cluster_hamming_max >= 0:
        hashes = [
            (str(record["sample_id"]), int(record["difference_hash"]))
            for record in records
        ]
        for index, (left_id, left_hash) in enumerate(hashes):
            for right_id, right_hash in hashes[index + 1:]:
                if (
                    hamming_distance(left_hash, right_hash)
                    <= cluster_hamming_max
                ):
                    union.union(left_id, right_id)

    roots = {}
    for sample_id in ids:
        roots.setdefault(union.find(sample_id), len(roots))
    return {
        sample_id: f"cluster_{roots[union.find(sample_id)]:05d}"
        for sample_id in ids
    }


def _fraction(cluster_id: str, *, salt: str) -> float:
    digest = hashlib.sha256(
        f"{salt}|{cluster_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_splits(
    records: list[dict[str, Any]],
    clusters: dict[str, str],
    *,
    protected_split: str,
    holdout_fraction: float,
    salt: str,
) -> dict[str, Any]:
    """Assign whole clusters to train/val; never move protected rows."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cluster[clusters[str(record["sample_id"])]].append(record)

    protected_clusters = {
        cluster
        for cluster, group in by_cluster.items()
        if any(
            str(item["split"]) == protected_split for item in group
        )
    }
    contaminated = {
        cluster
        for cluster in protected_clusters
        if any(
            str(item["split"]) != protected_split
            for item in by_cluster[cluster]
        )
    }

    assignments: dict[str, str] = {}
    dropped: list[str] = []
    for cluster, group in sorted(by_cluster.items()):
        for record in group:
            sample_id = str(record["sample_id"])
            split = str(record["split"])
            if split == protected_split:
                assignments[sample_id] = protected_split
                continue
            if cluster in protected_clusters:
                dropped.append(sample_id)
                continue
            assignments[sample_id] = (
                "val"
                if _fraction(cluster, salt=salt) < holdout_fraction
                else "train"
            )

    counts = Counter(assignments.values())
    cluster_sizes = sorted(
        (len(group) for group in by_cluster.values()), reverse=True
    )
    largest_fraction = (
        cluster_sizes[0] / len(records) if cluster_sizes else 0.0
    )
    return {
        "assignments": assignments,
        "dropped_sample_ids": sorted(dropped),
        "summary": {
            "samples": len(records),
            "clusters": len(by_cluster),
            "largest_cluster_size": (
                cluster_sizes[0] if cluster_sizes else 0
            ),
            "largest_cluster_fraction": largest_fraction,
            "median_cluster_size": (
                cluster_sizes[len(cluster_sizes) // 2]
                if cluster_sizes
                else 0
            ),
            "clusters_touching_protected_split": len(
                protected_clusters
            ),
            "clusters_contaminating_protected_split": len(contaminated),
            "dropped_for_protected_contamination": len(dropped),
            "split_counts": dict(sorted(counts.items())),
            "protected_split": protected_split,
            "holdout_fraction": holdout_fraction,
            "chaining_warning": largest_fraction > 0.05,
        },
    }


def rewrite_manifest(
    manifest_rows: list[dict[str, str]],
    *,
    assignments: dict[str, str],
    clusters: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in assignments:
            continue
        updated = dict(row)
        updated["split"] = assignments[sample_id]
        updated["parent_id"] = clusters[sample_id]
        rows.append(updated)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regroup augmented patches into source-crop clusters and "
            "write a leakage-free dataset-specific split."
        )
    )
    parser.add_argument("--records-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--protected-split", default="test")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--cluster-hamming-max", type=int, default=5)
    parser.add_argument("--salt", default="p15-dsifn-source-split")
    args = parser.parse_args()

    records = _read_csv(Path(args.records_csv))
    manifest_rows = _read_csv(Path(args.manifest))
    clusters = build_clusters(
        records, cluster_hamming_max=int(args.cluster_hamming_max)
    )
    result = assign_splits(
        records,
        clusters,
        protected_split=str(args.protected_split),
        holdout_fraction=float(args.holdout_fraction),
        salt=str(args.salt),
    )
    rows = rewrite_manifest(
        manifest_rows,
        assignments=result["assignments"],
        clusters=clusters,
    )
    _write_csv(Path(args.output_manifest), rows)

    report = {
        "status": "completed",
        "cluster_hamming_max": int(args.cluster_hamming_max),
        "salt": str(args.salt),
        "input_manifest": str(args.manifest),
        "output_manifest": str(args.output_manifest),
        "split_kind": "dataset_specific_source_crop_split",
        **result["summary"],
        "dropped_sample_ids": result["dropped_sample_ids"][:200],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["chaining_warning"]:
        print(
            "WARNING: the largest cluster covers more than 5% of the "
            "dataset; near-duplicate chaining may have merged unrelated "
            "crops. Re-run with --cluster-hamming-max 2 and compare.",
        )


if __name__ == "__main__":
    main()
