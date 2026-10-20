"""Build the P27 paper tables from finished results (no model, no data).

Inputs: the CSV written by ``tools.p27_summary --output`` and the DiffRegCD
result files of ``tools.run_p27_diffregcd_f1``. Only test-stage rows are used.

  main comparison  per dataset, grouped by how misalignment is handled: every
                   detector without correction (none); the DINOv3 detector trained
                   with misalignment augmentation (P29); DiffRegCD (joint
                   registration); the proposed method (DINOv3 detector + localizer
                   + selector)
  generality       per detector and dataset: without correction and with the
                   localizer and selector, and the change

Cells whose run has not finished are written as "-". Writes Markdown and CSV,
plus sources.json with the SHA256 of every input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
LEVELS = ("0", "1", "2", "3")
HEADER = ("Clean", "Level 1", "Level 2", "Level 3", "Mean (L1-L3)")
DETECTORS = (  # (name, summary row for LEVIR-CD / DSIFN-CD, summary row for WHU-CD)
    ("BIT", "bit", "bit_whu"),
    ("CD-Lamba", "cdlamba", "cdlamba_whu"),
    ("CDxLSTM", "cdxlstm", "cdxlstm_whu"),
    ("DINOv3 detector", "dinov3", "dinov3"),
)
AUGMENTED = ("DINOv3 detector + misalignment augmentation", "dinov3aug", "dinov3aug")
PROPOSED = "Ours (DINOv3 + localizer + selector)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(row, dataset) -> {"<level>_<path>": F1} for the test stage."""
    table = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record["stage"] != "test":
                continue
            values = {key[3:]: float(value) for key, value in record.items() if key.startswith("f1_") and value != ""}
            table[(record["row"], record["dataset"])] = values
    return table


def load_diffregcd(directory: Path) -> dict[str, dict[str, float]]:
    result = {}
    for dataset in DATASETS:
        path = directory / f"{dataset}.json"
        if path.is_file():
            f1 = json.loads(path.read_text(encoding="utf-8"))["f1"]
            result[dataset] = {level: float(f1[level]["diffregcd"]) for level in LEVELS}
    return result


def path_of(values: dict[str, float] | None, path: str) -> dict[str, float] | None:
    return None if values is None else {level: values[f"{level}_{path}"] for level in LEVELS
                                        if f"{level}_{path}" in values}


def scores(by_level: dict[str, float] | None) -> list[float | None]:
    """F1 at every level, then the mean of the perturbed levels."""
    row = [None if by_level is None else by_level.get(level) for level in LEVELS]
    perturbed = row[1:]
    mean = None if any(v is None for v in perturbed) else sum(perturbed) / len(perturbed)
    return [*row, mean]


def summary_row(summary: dict, row: str, row_whu: str, dataset: str) -> dict[str, float] | None:
    return summary.get((row_whu if dataset == "WHU-CD" else row, dataset))


def main_comparison(summary: dict, diffregcd: dict) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for name, row, row_whu in DETECTORS:
            values = summary_row(summary, row, row_whu, dataset)
            rows.append({"dataset": dataset, "method": name, "handling": "none",
                         "scores": scores(path_of(values, "raw"))})
        name, row, row_whu = AUGMENTED
        rows.append({"dataset": dataset, "method": name, "handling": "augmentation",
                     "scores": scores(path_of(summary_row(summary, row, row_whu, dataset), "raw"))})
        rows.append({"dataset": dataset, "method": "DiffRegCD", "handling": "joint registration",
                     "scores": scores(diffregcd.get(dataset))})
        rows.append({"dataset": dataset, "method": PROPOSED, "handling": "localizer + selector",
                     "scores": scores(path_of(summary.get(("dinov3", dataset)), "selector"))})
    return rows


def generality(summary: dict) -> list[dict[str, Any]]:
    rows = []
    for name, row, row_whu in (*DETECTORS, AUGMENTED):
        for dataset in DATASETS:
            values = summary_row(summary, row, row_whu, dataset)
            raw, ours = scores(path_of(values, "raw")), scores(path_of(values, "selector"))
            change = [None if a is None or b is None else b - a for a, b in zip(raw, ours)]
            rows.append({"detector": name, "dataset": dataset, "raw": raw, "ours": ours, "change": change})
    return rows


def cell(value: float | None, *, signed: bool = False, bold: bool = False) -> str:
    if value is None:
        return "-"
    if signed and round(value * 100, 2) == 0:
        value = 0.0
    text = f"{value * 100:+.2f}" if signed and value != 0 else f"{value * 100:.2f}"
    return f"**{text}**" if bold else text


def best_indices(rows: Sequence[dict[str, Any]]) -> set[tuple[int, int]]:
    best = set()
    for column in range(len(HEADER)):
        present = [(r["scores"][column], i) for i, r in enumerate(rows) if r["scores"][column] is not None]
        if present:
            top = max(v for v, _ in present)
            best |= {(i, column) for v, i in present if abs(v - top) < 5e-7}
    return best


def render_main(rows: list[dict[str, Any]]) -> str:
    lines = ["Table: F1 (%) on the test sets under misalignment. Handling \"none\": the detector is applied to the "
             "misaligned pair as is; \"augmentation\": the same DINOv3 detector trained with the test perturbation; "
             "the proposed method uses the DINOv3 detector. Bold: best in each column of a dataset.", "",
             "| Dataset | Misalignment handling | Method | " + " | ".join(HEADER) + " |",
             "|---|---|---|" + "---:|" * len(HEADER)]
    for dataset in DATASETS:
        block = [r for r in rows if r["dataset"] == dataset]
        best = best_indices(block)
        for i, r in enumerate(block):
            cells = [cell(v, bold=(i, c) in best) for c, v in enumerate(r["scores"])]
            lines.append(f"| {dataset if i == 0 else ''} | {r['handling']} | {r['method']} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_generality(rows: list[dict[str, Any]]) -> str:
    lines = ["Table: F1 (%) of each detector without correction and with the proposed localizer and selector "
             "(test sets); the change is in percentage points.", "",
             "| Detector | Dataset | Setting | " + " | ".join(HEADER) + " |",
             "|---|---|---|" + "---:|" * len(HEADER)]
    for name, _, _ in (*DETECTORS, AUGMENTED):
        for j, r in enumerate(x for x in rows if x["detector"] == name):
            label = name if j == 0 else ""
            lines.append(f"| {label} | {r['dataset']} | Without correction | " +
                         " | ".join(cell(v) for v in r["raw"]) + " |")
            lines.append("|  |  | + localizer + selector | " + " | ".join(cell(v) for v in r["ours"]) + " |")
            lines.append("|  |  | Change | " + " | ".join(cell(v, signed=True) for v in r["change"]) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else (f"{v:.6f}" if isinstance(v, float) else v) for v in row])


def build(summary_path: Path | Sequence[Path], diffregcd_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Several summary CSVs (rows run on different machines) are merged; a row may come from one only."""
    summary_paths = [summary_path] if isinstance(summary_path, Path) else list(summary_path)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for path in summary_paths:
        rows = load_summary(path)
        clash = sorted(set(rows) & set(summary))
        if clash:
            raise SystemExit(f"{path} repeats rows already read: {clash}")
        summary.update(rows)
    diffregcd = load_diffregcd(diffregcd_dir)
    main_rows, general_rows = main_comparison(summary, diffregcd), generality(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table_p27_main_comparison.md").write_bytes(render_main(main_rows).encode("utf-8"))
    (output_dir / "table_p27_generality.md").write_bytes(render_generality(general_rows).encode("utf-8"))
    levels = ["f1_clean", "f1_level1", "f1_level2", "f1_level3", "f1_mean_l1_l3"]
    write_csv(output_dir / "table_p27_main_comparison.csv", ["dataset", "handling", "method", *levels],
              [[r["dataset"], r["handling"], r["method"], *r["scores"]] for r in main_rows])
    write_csv(output_dir / "table_p27_generality.csv", ["detector", "dataset", "setting", *levels],
              [[r["detector"], r["dataset"], setting, *r[key]]
               for r in general_rows for setting, key in (("raw", "raw"), ("ours", "ours"), ("change", "change"))])
    missing = sorted({f"{r['method']} / {r['dataset']}" for r in main_rows if r["scores"][0] is None})
    sources = {"summary_csv": [{"path": path.as_posix(), "sha256": sha256(path)} for path in summary_paths],
               "diffregcd": {name: {"path": (diffregcd_dir / f"{name}.json").as_posix(),
                                    "sha256": sha256(diffregcd_dir / f"{name}.json")} for name in diffregcd},
               "missing": missing}
    (output_dir / "sources.json").write_bytes(json.dumps(sources, indent=2).encode("utf-8"))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", default=["reports/p27_localizer/summary.csv"],
                        help="p27_summary CSVs; several when rows ran on different machines")
    parser.add_argument("--diffregcd", default="reports/p27_localizer/diffregcd/test")
    parser.add_argument("--ablations", default="reports/p27_localizer/ablations",
                        help="ablation runs for the localizer, selector and efficiency tables (skipped if absent)")
    parser.add_argument("--output-dir", default="paper/data/p27")
    args = parser.parse_args()
    sources = build([Path(p) for p in args.summary], Path(args.diffregcd), Path(args.output_dir))
    print((Path(args.output_dir) / "table_p27_main_comparison.md").read_text(encoding="utf-8"))
    print("not finished yet: " + (", ".join(sources["missing"]) or "none"))
    from tools.p27_ablation_tables import build_ablation_tables

    print("ablation tables: " + (", ".join(build_ablation_tables(Path(args.ablations), Path(args.output_dir)))
                                 or "none"))


if __name__ == "__main__":
    main()
