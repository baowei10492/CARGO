from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    split: str
    t1_path: str
    t2_path: str
    mask_path: str
    parent_id: str


def _index_by_stem(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {folder}")
    result: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(
                f"Duplicate filename stem '{path.stem}' in {folder}: "
                f"{result[path.stem].name}, {path.name}"
            )
        result[path.stem] = path
    return result


def _read_split_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        sample_id = Path(text.split()[0]).stem
        if sample_id in seen:
            raise ValueError(f"Duplicate ID '{sample_id}' in split file {path}")
        seen.add(sample_id)
        ids.append(sample_id)
    if not ids:
        raise ValueError(f"Split file is empty: {path}")
    return ids


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _rows_for_folders(
    root: Path,
    split: str,
    t1_dir: Path,
    t2_dir: Path,
    mask_dir: Path,
    allowed_ids: Sequence[str] | None = None,
    parent_id_mode: str = "none",
    sample_id_mode: str = "stem",
) -> list[ManifestRow]:
    t1_files = _index_by_stem(t1_dir)
    t2_files = _index_by_stem(t2_dir)
    mask_files = _index_by_stem(mask_dir)

    id_sets = {
        "T1": set(t1_files),
        "T2": set(t2_files),
        "mask": set(mask_files),
    }
    common = set.intersection(*id_sets.values())
    union = set.union(*id_sets.values())
    if not union:
        raise ValueError(
            f"No supported paired images found in {t1_dir}, {t2_dir}, {mask_dir}"
        )
    if common != union:
        details = []
        for name, ids in id_sets.items():
            missing = sorted(union - ids)
            if missing:
                details.append(f"{name} missing {len(missing)} IDs, e.g. {missing[:5]}")
        raise ValueError("Unpaired files detected: " + "; ".join(details))

    ids_to_use = list(allowed_ids) if allowed_ids is not None else sorted(common)
    unknown = sorted(set(ids_to_use) - common)
    if unknown:
        raise ValueError(
            f"Split '{split}' references {len(unknown)} missing IDs, e.g. {unknown[:5]}"
        )

    def _identifier(stem: str) -> str:
        return f"{split}_{stem}" if sample_id_mode == "split_stem" else stem

    return [
        ManifestRow(
            sample_id=_identifier(sample_id),
            split=split,
            t1_path=_relative(t1_files[sample_id], root),
            t2_path=_relative(t2_files[sample_id], root),
            mask_path=_relative(mask_files[sample_id], root),
            parent_id=(
                _identifier(sample_id)
                if parent_id_mode == "sample_id"
                else ""
            ),
        )
        for sample_id in ids_to_use
    ]


def build_manifest(
    dataset_root: str | Path,
    output_csv: str | Path,
    splits: Iterable[str] = ("train", "val", "test"),
    t1_name: str = "A",
    t2_name: str = "B",
    mask_name: str = "label",
    split_files: Mapping[str, str | Path] | None = None,
    parent_id_mode: str = "none",
    sample_id_mode: str = "stem",
) -> list[ManifestRow]:
    """Build a strict paired-image manifest.

    Two layouts are supported:
    1. ``root/{split}/{A,B,label}``
    2. ``root/{A,B,label}`` plus one ID list for each split.

    ``sample_id_mode`` controls how the identifier is formed. ``stem``
    keeps the bare filename stem, which only works when filenames are
    globally unique. Datasets that restart numbering inside every split
    (DSIFN-CD names every split ``1.jpg``, ``2.jpg``, ...) need
    ``split_stem``, otherwise one identifier denotes three different
    image pairs.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    split_list = list(splits)
    if not split_list:
        raise ValueError("At least one split is required")
    if parent_id_mode not in {"none", "sample_id"}:
        raise ValueError("parent_id_mode must be 'none' or 'sample_id'")
    if sample_id_mode not in {"stem", "split_stem"}:
        raise ValueError(
            "sample_id_mode must be 'stem' or 'split_stem'"
        )

    rows: list[ManifestRow] = []
    if split_files:
        flat_dirs = (root / t1_name, root / t2_name, root / mask_name)
        for split in split_list:
            if split not in split_files:
                raise ValueError(f"No split file was provided for '{split}'")
            split_path = Path(split_files[split])
            if not split_path.is_absolute():
                split_path = root / split_path
            allowed_ids = _read_split_ids(split_path)
            rows.extend(
                _rows_for_folders(
                    root,
                    split,
                    *flat_dirs,
                    allowed_ids=allowed_ids,
                    parent_id_mode=parent_id_mode,
                    sample_id_mode=sample_id_mode,
                )
            )
    else:
        for split in split_list:
            split_root = root / split
            rows.extend(
                _rows_for_folders(
                    root,
                    split,
                    split_root / t1_name,
                    split_root / t2_name,
                    split_root / mask_name,
                    parent_id_mode=parent_id_mode,
                    sample_id_mode=sample_id_mode,
                )
            )

    # A sample_id must denote exactly one image pair: every per-sample
    # CSV, router feature table and ID-hash partition in this project is
    # keyed by it. Collisions are silent corruption downstream, so they
    # fail the build instead.
    counts = Counter(row.sample_id for row in rows)
    duplicates = sorted(
        sample_id for sample_id, total in counts.items() if total > 1
    )
    if duplicates:
        raise ValueError(
            f"Manifest has {len(duplicates)} duplicate sample_id values "
            f"across splits, e.g. {duplicates[:5]}. Rebuild with "
            "sample_id_mode='split_stem' (CLI: --sample-id-mode "
            "split_stem) when filenames restart inside each split."
        )

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return rows


def read_manifest(
    path: str | Path, split: str | None = None
) -> list[ManifestRow]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = set(ManifestRow.__dataclass_fields__)
        actual = set(reader.fieldnames or [])
        missing = expected - actual
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = [
            ManifestRow(**{key: record[key] for key in expected})
            for record in reader
            if split is None or record["split"] == split
        ]
    if not rows:
        suffix = f" for split '{split}'" if split is not None else ""
        raise ValueError(f"Manifest has no rows{suffix}: {manifest_path}")
    return rows
