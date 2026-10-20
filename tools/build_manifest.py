from __future__ import annotations

import argparse
from pathlib import Path

from rs_cd.datasets.manifest import build_manifest


def _parse_split_file(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"Expected SPLIT=PATH, received '{value}'"
            )
        split, path = value.split("=", 1)
        if not split or not path:
            raise argparse.ArgumentTypeError(
                f"Expected SPLIT=PATH, received '{value}'"
            )
        result[split] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a strict T1/T2/mask CSV manifest."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--t1-dir", default="A")
    parser.add_argument("--t2-dir", default="B")
    parser.add_argument("--mask-dir", default="label")
    parser.add_argument(
        "--parent-id-mode",
        choices=["none", "sample_id"],
        default="none",
        help=(
            "Use sample_id as parent_id only when every file is an original "
            "uncut parent image. Leave as none for patches without a trusted map."
        ),
    )
    parser.add_argument(
        "--sample-id-mode",
        choices=["stem", "split_stem"],
        default="stem",
        help=(
            "How to form sample_id. Use split_stem when filenames "
            "restart inside each split (DSIFN-CD names every split "
            "1.jpg, 2.jpg, ...), otherwise one id denotes three pairs."
        ),
    )
    parser.add_argument(
        "--split-file",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help="Use flat A/B/label folders plus ID lists. Repeat per split.",
    )
    args = parser.parse_args()

    split_files = _parse_split_file(args.split_file) if args.split_file else None
    rows = build_manifest(
        dataset_root=args.dataset_root,
        output_csv=args.output,
        splits=args.splits,
        t1_name=args.t1_dir,
        t2_name=args.t2_dir,
        mask_name=args.mask_dir,
        split_files=split_files,
        parent_id_mode=args.parent_id_mode,
        sample_id_mode=args.sample_id_mode,
    )
    counts = {split: sum(row.split == split for row in rows) for split in args.splits}
    print(f"Wrote {len(rows)} paired samples to {Path(args.output).resolve()}")
    print("Split counts:", counts)


if __name__ == "__main__":
    main()
