from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rs_cd.datasets.manifest import ManifestRow, read_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def audit(
    dataset_root: str | Path,
    manifest_path: str | Path,
    compute_hashes: bool = True,
    mask_threshold: int = 127,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    rows = read_manifest(manifest_path)
    split_counts = Counter(row.split for row in rows)
    size_counts: Counter[str] = Counter()
    mask_value_counts: Counter[int] = Counter()
    change_pixels = 0
    total_pixels = 0
    invalid: list[dict[str, Any]] = []
    non_binary_samples: list[dict[str, Any]] = []
    non_binary_pixels = 0
    hash_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    rows_with_parent_id = 0

    for row in rows:
        paths = {
            "t1": _resolve(root, row.t1_path),
            "t2": _resolve(root, row.t2_path),
            "mask": _resolve(root, row.mask_path),
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            invalid.append({"id": row.sample_id, "missing": missing})
            continue
        with Image.open(paths["t1"]) as image:
            t1_size = image.size
        with Image.open(paths["t2"]) as image:
            t2_size = image.size
        with Image.open(paths["mask"]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
            mask_size = image.size
        if t1_size != t2_size or t1_size != mask_size:
            invalid.append(
                {
                    "id": row.sample_id,
                    "sizes": {
                        "t1": t1_size,
                        "t2": t2_size,
                        "mask": mask_size,
                    },
                }
            )
        size_counts[f"{t1_size[0]}x{t1_size[1]}"] += 1
        unique, counts = np.unique(mask, return_counts=True)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=True):
            mask_value_counts[int(value)] += int(count)
        unexpected = [
            {"value": int(value), "pixels": int(count)}
            for value, count in zip(unique.tolist(), counts.tolist(), strict=True)
            if int(value) not in {0, 255}
        ]
        if unexpected:
            non_binary_samples.append(
                {
                    "id": row.sample_id,
                    "split": row.split,
                    "values": unexpected,
                }
            )
            non_binary_pixels += sum(item["pixels"] for item in unexpected)
        binary = mask > mask_threshold
        change_pixels += int(binary.sum())
        total_pixels += int(binary.size)
        if row.parent_id:
            rows_with_parent_id += 1
            parent_splits[row.parent_id].add(row.split)
        if compute_hashes:
            triplet_hash = hashlib.sha256(
                (
                    _sha256(paths["t1"])
                    + _sha256(paths["t2"])
                    + _sha256(paths["mask"])
                ).encode("ascii")
            ).hexdigest()
            hash_splits[triplet_hash].add(row.split)

    cross_split_hashes = [
        {"sha256": digest, "splits": sorted(splits)}
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    ]
    cross_split_parents = [
        {"parent_id": parent, "splits": sorted(splits)}
        for parent, splits in parent_splits.items()
        if len(splits) > 1
    ]
    return {
        "dataset_root": str(root),
        "manifest": str(Path(manifest_path).resolve()),
        "samples": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "image_sizes": dict(size_counts.most_common()),
        "mask_pixel_values": {
            str(key): value for key, value in sorted(mask_value_counts.items())
        },
        "change_pixels": change_pixels,
        "total_pixels": total_pixels,
        "change_fraction": change_pixels / max(1, total_pixels),
        "mask_threshold": mask_threshold,
        "invalid_samples": invalid,
        "non_binary_mask_samples": non_binary_samples,
        "non_binary_mask_pixels": non_binary_pixels,
        "cross_split_duplicate_triplets": cross_split_hashes,
        "cross_split_parent_ids": cross_split_parents,
        "parent_id_rows": rows_with_parent_id,
        "parent_id_coverage": rows_with_parent_id / max(1, len(rows)),
        "hash_audit_enabled": compute_hashes,
        "passed": not invalid
        and not cross_split_hashes
        and not cross_split_parents,
        "limitations": (
            (
                [
                    "parent leakage is incomplete because some rows lack a trusted parent_id"
                ]
                if rows_with_parent_id < len(rows)
                else []
            )
            + [
                f"mask pixels greater than {mask_threshold} are interpreted as changed"
            ]
        ),
        "warnings": (
            [
                (
                    f"{non_binary_pixels} mask pixels in "
                    f"{len(non_binary_samples)} samples are neither 0 nor 255; "
                    f"their class is determined by the >{mask_threshold} threshold"
                )
            ]
            if non_binary_samples
            else []
        ),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 数据审计报告",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Samples: `{report['samples']}`",
        f"- Splits: `{report['split_counts']}`",
        f"- Image sizes: `{report['image_sizes']}`",
        f"- Change fraction: `{report['change_fraction']:.6f}`",
        f"- Mask threshold: `>{report['mask_threshold']}`",
        f"- Mask values: `{report['mask_pixel_values']}`",
        f"- Invalid samples: `{len(report['invalid_samples'])}`",
        f"- Parent ID coverage: `{report['parent_id_coverage']:.3f}`",
        (
            "- Non-binary mask pixels: "
            f"`{report['non_binary_mask_pixels']}` in "
            f"`{len(report['non_binary_mask_samples'])}` samples"
        ),
        (
            "- Cross-split duplicate triplets: "
            f"`{len(report['cross_split_duplicate_triplets'])}`"
        ),
        f"- Cross-split parent IDs: `{len(report['cross_split_parent_ids'])}`",
        "",
        "## 局限",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    if report["warnings"]:
        lines.extend(["", "## 警告", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a paired CD dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--mask-threshold", type=int, default=127)
    args = parser.parse_args()

    report = audit(
        args.dataset_root,
        args.manifest,
        compute_hashes=not args.skip_hashes,
        mask_threshold=args.mask_threshold,
    )
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, md_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
