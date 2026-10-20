"""P31 selector ablation under the border-free protocol (CPU only).

The P27 selector ablation (tools/p27_selector_ablation.py), unchanged, on the border-free rows: DINOv3 (P30) and
BIT, CD-Lamba, CDxLSTM (P31), each on LEVIR-CD, DSIFN-CD and WHU-CD. For every row the selector is retrained with
feature subsets on that row's border-free training records and scored on its test records, next to always
correcting and the localizer gate; the frozen selectors are not touched. The retrained full selector must
reproduce the frozen one, so run this on the machine that trained the selectors.

Output: <output-root>/<row>/<tag>/selector_ablation.json (the layout tools/p27_ablation_tables.selector_cells reads,
with BORDER_FREE_ROWS).
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from tools import p27_selector_ablation, run_p30
from tools.run_p27_step1 import sha256

DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
BORDER_FREE_ROWS = (("dinov3", "DINOv3", DATASETS), ("bit", "BIT", DATASETS),
                    ("cdlamba", "CD-Lamba", DATASETS), ("cdxlstm", "CDxLSTM", DATASETS))
SELECTOR_CONFIG = Path("configs/p27_selector.yaml")


def row_root(row: str) -> Path:
    """Where a row's run lives: the P31 rows under reports/p31, the P32 ones under reports/p32."""
    for root in (Path("reports/p31"), Path("reports/p32")):
        if (root / row / "test" / "p31_test.json").is_file():
            return root / row
    raise SystemExit(f"{row}: no finished test run under reports/p31 or reports/p32")


def sources(row: str) -> tuple[Path, list[Path]]:
    """(selector training root with features/ and bundles/, test record files) of a border-free row."""
    if row == "dinov3":
        test = run_p30.pointed("reports/p30/test/LATEST")
        return run_p30.pointed("reports/p30/selector/LATEST"), [test / f"records_proposed_{name}.csv" for name in DATASETS]
    root = row_root(row)
    return root, [root / "test" / f"records_{name}.csv" for name in DATASETS]


def check_bundles(train_root: Path, names: Sequence[str], config_sha: str) -> str:
    """The row's selectors are border-free (P30 protocol), from the P27 recipe, for one localizer."""
    localizers = set()
    for name in names:
        bundle = torch.load(train_root / "bundles" / f"selector_{name}.pth", map_location="cpu", weights_only=False)
        if bundle.get("protocol") != "p30" or bundle.get("config_sha256") != config_sha:
            raise SystemExit(f"{train_root} {name}: not a border-free selector trained with {SELECTOR_CONFIG}")
        localizers.add(bundle["localizer_sha256"])
    if len(localizers) != 1:
        raise SystemExit(f"{train_root}: selectors of one row belong to different localizers")
    return localizers.pop()


def concatenate(paths: Sequence[Path], out: Path) -> Path:
    fields: list[str] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields += [field for field in reader.fieldnames or [] if field not in fields]
            rows.extend(reader)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return out


def run_row(row: str, config: dict[str, Any], config_sha: str, output_dir: Path, *, train_root: Path | None = None,
            records: Sequence[Path] | None = None, names: Sequence[str] = DATASETS,
            sweep_step: float = 0.02) -> dict[str, Any]:
    if train_root is None:
        train_root, records = sources(row)
    localizer = check_bundles(train_root, names, config_sha)
    output_dir.mkdir(parents=True, exist_ok=False)
    combined = concatenate(records, output_dir / "test_records.csv")
    result = p27_selector_ablation.run(config, train_root, combined, sweep_step=sweep_step)
    result.update(row=row, protocol="p30 border-free", localizer_sha256=localizer, selector_config_sha256=config_sha,
                  train_root=str(train_root), record_files=[str(path) for path in records])
    (output_dir / "selector_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    p27_selector_ablation.write_csv(output_dir / "tau_sweep.csv",
                                    [{"dataset": name, **entry} for name, value in result["datasets"].items()
                                     for entry in value["sweep"]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", default=[row for row, _, _ in BORDER_FREE_ROWS])
    parser.add_argument("--output-root", default="reports/p31/selector_ablation")
    args = parser.parse_args()
    config = yaml.safe_load(SELECTOR_CONFIG.read_text(encoding="utf-8"))
    config_sha = sha256(SELECTOR_CONFIG)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reproduced = {}
    for row in args.rows:
        result = run_row(row, config, config_sha, Path(args.output_root) / row / tag)
        print(f"\n===== {row} =====\n" + p27_selector_ablation.render(result), flush=True)
        reproduced[row] = {name: entry["full_reproduces_main"] for name, entry in result["datasets"].items()}
    print(json.dumps({"full_selector_reproduces_frozen": reproduced}, indent=1))
    if not all(check["tau_equal"] for row in reproduced.values() for check in row.values()):
        raise SystemExit("a retrained full selector does not reproduce the frozen one; run on the training machine")


if __name__ == "__main__":
    main()
