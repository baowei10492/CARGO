from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _format_value(key: str, value: str) -> str:
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return f"{key}={int(text):03d}"
    return f"{key}={text}"


def column_parent_ids(
    index_rows: list[dict[str, str]],
    *,
    keys: tuple[str, ...] = ("raster_split", "column"),
) -> dict[str, str]:
    """Map sample_id -> a parent id derived from its spatial strip.

    For a dataset tiled out of large rasters, the meaningful leakage unit
    is the spatial strip, not the tile. Making the parent id the strip
    turns the audit's cross-split parent check into a verification of the
    exact invariant the split was designed around. Without any parent_id
    the check silently does nothing and reports a vacuous zero.

    The key MUST identify the strip globally. Using the column index
    alone is wrong whenever splits come from different rasters: column 0
    of the train raster and column 0 of the test raster are unrelated
    locations, and keying on the bare column number reports them as a
    shared parent, i.e. a false leakage alarm. Hence the default key set
    includes ``raster_split``.
    """

    mapping: dict[str, str] = {}
    for row in index_rows:
        sample_id = str(row["sample_id"])
        if sample_id in mapping:
            raise ValueError(f"Duplicate sample_id in index: {sample_id}")
        missing = [key for key in keys if key not in row]
        if missing:
            raise KeyError(f"Spatial index lacks columns: {missing}")
        mapping[sample_id] = "|".join(
            _format_value(key, row[key]) for key in keys
        )
    if not mapping:
        raise ValueError("Spatial index produced no rows")
    return mapping


def apply_parent_ids(
    manifest_rows: list[dict[str, str]],
    mapping: dict[str, str],
) -> tuple[list[dict[str, str]], int]:
    updated = 0
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        new_row = dict(row)
        parent = mapping.get(str(row["sample_id"]))
        if parent is None:
            raise KeyError(
                f"sample_id missing from spatial index: {row['sample_id']}"
            )
        new_row["parent_id"] = parent
        updated += 1
        rows.append(new_row)
    return rows, updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fill the manifest parent_id column from a spatial tile "
            "index so that the dataset audit's cross-split parent check "
            "actually runs. A zero cross-split-parent count is "
            "meaningless while parent_id_coverage is 0."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tile-index", required=True)
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["raster_split", "column"],
        help=(
            "Spatial-index columns that together identify the strip. "
            "Must be globally unique: keying on a bare column index is "
            "wrong when splits come from different rasters."
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        manifest_rows = list(reader)
    if "parent_id" not in fieldnames:
        raise ValueError("Manifest has no parent_id column")
    with Path(args.tile_index).expanduser().open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        index_rows = list(csv.DictReader(handle))

    mapping = column_parent_ids(
        index_rows,
        keys=tuple(str(key) for key in args.keys),
    )
    rows, updated = apply_parent_ids(manifest_rows, mapping)

    parents_by_split: dict[str, set[str]] = {}
    for row in rows:
        parents_by_split.setdefault(row["split"], set()).add(
            row["parent_id"]
        )
    shared = []
    splits = sorted(parents_by_split)
    for first in range(len(splits)):
        for second in range(first + 1, len(splits)):
            overlap = (
                parents_by_split[splits[first]]
                & parents_by_split[splits[second]]
            )
            if overlap:
                shared.append(
                    (splits[first], splits[second], sorted(overlap))
                )

    output = Path(args.output or args.manifest).expanduser()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows_updated={updated}")
    for split in splits:
        values = sorted(parents_by_split[split])
        print(
            f"{split}: {len(values)} parents "
            f"{values[0]}..{values[-1]}"
        )
    if shared:
        print("SHARED PARENTS ACROSS SPLITS (leakage!):")
        for first, second, overlap in shared:
            print(f"  {first} & {second}: {overlap}")
    else:
        print("no parent id appears in more than one split")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
