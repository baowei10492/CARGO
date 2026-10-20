"""P27 paper tables are built from the summary CSV and the DiffRegCD results only."""
from __future__ import annotations

import csv
import json

import pytest

from tools.p27_paper_tables import build

LEVELS = ("0", "1", "2", "3")


def _summary(path, rows) -> None:
    fields = ["row", "stage", "dataset"] + [f"f1_{level}_{p}" for level in LEVELS for p in ("raw", "selector")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, stage, dataset, raw, ours in rows:
            record = {"row": row, "stage": stage, "dataset": dataset}
            for level, a, b in zip(LEVELS, raw, ours):
                record[f"f1_{level}_raw"], record[f"f1_{level}_selector"] = a, b
            writer.writerow(record)


def test_tables_use_test_rows_and_mark_missing_cells(tmp_path) -> None:
    raw, ours = (0.9, 0.7, 0.6, 0.5), (0.9, 0.8, 0.8, 0.75)
    rows = [("dinov3", "test", name, raw, ours) for name in ("LEVIR-CD", "DSIFN-CD", "WHU-CD")]
    rows += [("bit", "test", "LEVIR-CD", (0.88, 0.6, 0.5, 0.4), (0.88, 0.7, 0.7, 0.7)),
             ("bit", "selector_eval", "LEVIR-CD", (0.1,) * 4, (0.1,) * 4),
             ("cdlamba_whu", "test", "WHU-CD", (0.86, 0.74, 0.56, 0.41), (0.86, 0.82, 0.81, 0.77))]
    _summary(tmp_path / "summary.csv", rows)
    diffregcd = tmp_path / "diffregcd"
    diffregcd.mkdir()
    for name in ("LEVIR-CD", "DSIFN-CD"):
        f1 = {level: {"diffregcd": 0.95 - 0.2 * int(level), "diffregcd_oracle": 0.5} for level in LEVELS}
        (diffregcd / f"{name}.json").write_text(json.dumps({"f1": f1}), encoding="utf-8")

    out = tmp_path / "tables"
    sources = build(tmp_path / "summary.csv", diffregcd, out)
    assert "BIT / WHU-CD" in sources["missing"] and "DiffRegCD / WHU-CD" in sources["missing"]
    assert "BIT / LEVIR-CD" not in sources["missing"] and set(sources["diffregcd"]) == {"LEVIR-CD", "DSIFN-CD"}

    with (out / "table_p27_main_comparison.csv").open(encoding="utf-8") as handle:
        main = list(csv.DictReader(handle))
    assert len(main) == 3 * 7
    levir = {r["method"]: r for r in main if r["dataset"] == "LEVIR-CD"}
    augmented = levir["DINOv3 detector + misalignment augmentation"]
    assert augmented["handling"] == "augmentation" and augmented["f1_clean"] == ""
    assert [r["handling"] for r in main[:7]] == ["none"] * 4 + ["augmentation", "joint registration",
                                                               "localizer + selector"]
    assert float(levir["BIT"]["f1_level1"]) == pytest.approx(0.6)  # the test row, not selector_eval
    assert float(levir["DiffRegCD"]["f1_clean"]) == pytest.approx(0.95)
    assert levir["CD-Lamba"]["f1_clean"] == ""
    proposed = levir["Ours (DINOv3 + localizer + selector)"]
    assert float(proposed["f1_mean_l1_l3"]) == pytest.approx((0.8 + 0.8 + 0.75) / 3)

    markdown = (out / "table_p27_main_comparison.md").read_text(encoding="utf-8")
    assert "**95.00**" in markdown and "**75.00**" in markdown
    assert "| WHU-CD | none | BIT | - | - | - | - | - |" in markdown

    with (out / "table_p27_generality.csv").open(encoding="utf-8") as handle:
        general = list(csv.DictReader(handle))
    change = next(r for r in general if r["detector"] == "CD-Lamba" and r["dataset"] == "WHU-CD"
                  and r["setting"] == "change")
    assert float(change["f1_level3"]) == pytest.approx(0.36)
    assert {r["detector"] for r in general} >= {"DINOv3 detector + misalignment augmentation"}

    # Rows run on another machine come in a second summary; a repeated row is refused.
    _summary(tmp_path / "other.csv", [("dinov3aug", "test", "LEVIR-CD", (0.88, 0.8, 0.7, 0.6), (0.88,) * 4)])
    build([tmp_path / "summary.csv", tmp_path / "other.csv"], diffregcd, out)
    with (out / "table_p27_main_comparison.csv").open(encoding="utf-8") as handle:
        merged = {(r["dataset"], r["method"]): r for r in csv.DictReader(handle)}
    assert float(merged[("LEVIR-CD", "DINOv3 detector + misalignment augmentation")]["f1_level3"]) == \
        pytest.approx(0.6)
    with pytest.raises(SystemExit, match="repeats"):
        build([tmp_path / "summary.csv", tmp_path / "summary.csv"], diffregcd, out)
    assert "+36.00" in (out / "table_p27_generality.md").read_text(encoding="utf-8")
