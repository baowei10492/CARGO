"""P27 ablation and efficiency tables from the finished ablation runs (no model, no data).

Reads reports/p27_localizer/ablations/ (scripts/run_p27_ablations.sh):

  localizer   localizer/test/<variant>.json: the P27 localizer and its reduced
              versions, and the RoMa and SIFT baselines, in front of the DINOv3
              detectors; every pair is corrected (no selector), so the table
              isolates the localizer
  selector    selector/<row>/<latest>/selector_ablation.json: per detector and
              dataset, always correct / localizer gate / the selector and its
              retrained feature subsets; per-cell values and the mean over cells
  efficiency  efficiency/<latest>.json: parameters and per-pair timings

Clean change = F1 with the method minus F1 without correction on unperturbed
pairs (negative = worse); gains are F1 changes at levels 1-3; all in points.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
LOCALIZERS = (
    ("p27_full", "P27 localizer (full)"),
    ("p27_one_fine_step", "one refinement step"),
    ("p27_coarse_only", "coarse matching only"),
    ("p27_no_reliability", "no reliability weighting"),
    ("roma_similarity", "RoMa + similarity fit"),
    ("sift_similarity", "SIFT + similarity fit"),
)
SELECTOR_ROWS = (  # (summary row, detector, datasets)
    ("dinov3", "DINOv3", DATASETS),
    ("bit", "BIT", ("LEVIR-CD", "DSIFN-CD")),
    ("cdlamba", "CD-Lamba", ("LEVIR-CD", "DSIFN-CD")),
    ("cdxlstm", "CDxLSTM", ("LEVIR-CD", "DSIFN-CD")),
    ("bit_whu", "BIT", ("WHU-CD",)),
    ("cdlamba_whu", "CD-Lamba", ("WHU-CD",)),
    ("cdxlstm_whu", "CDxLSTM", ("WHU-CD",)),
)
SELECTOR_VARIANTS = (
    ("always", "always correct"),
    ("gate", "localizer gate"),
    ("full", "selector (19 features)"),
    ("registration_only", "registration features only (6)"),
    ("task_only", "task features only (13)"),
    ("without_forward_backward", "without forward-backward consistency (18)"),
)
UNSAFE_CLEAN_DROP = 0.005


def latest(directory: Path) -> Path | None:
    runs = sorted(p for p in directory.iterdir() if p.is_dir()) if directory.is_dir() else []
    return runs[-1] if runs else None


def pts(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "-"
    value = round(value * 100, 2)
    if value == 0:
        return "0.00"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


# ---------------------------------------------------------------- localizer

def localizer_rows(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for variant, label in LOCALIZERS:
        path = directory / f"{variant}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))["datasets"]
        for dataset in DATASETS:
            entry = result.get(dataset)
            if entry is None:
                continue
            success = entry["localization_success"]
            rows.append({
                "dataset": dataset, "variant": variant, "label": label,
                "success": [success.get(level) for level in ("1", "2", "3")],
                "clean_change": -entry["clean_drop"]["estimated"],
                "gain": [entry["gain"][level]["estimated"] for level in ("1", "2", "3")],
            })
    return rows


def render_localizer(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["Table: localizer ablation and matching baselines (test sets, DINOv3 detectors, every pair corrected). "
             "Success: share of crops aligned within 2 px (%); F1 changes in points.", "",
             "| Dataset | Localizer | Success L1 | Success L2 | Success L3 | Clean change | Gain L1 | Gain L2 | Gain L3 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for i, r in enumerate(x for x in rows if x["dataset"] == dataset):
            success = [pts(s, signed=False) for s in r["success"]]
            lines.append(f"| {dataset if i == 0 else ''} | {r['label']} | " + " | ".join(success) + " | "
                         + " | ".join(pts(v) for v in (r["clean_change"], *r["gain"])) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- selector

def selector_cells(directory: Path, rows: Sequence[tuple[str, str, Sequence[str]]] = SELECTOR_ROWS) -> list[dict[str, Any]]:
    cells = []
    for row, detector, datasets in rows:
        run = latest(directory / row)
        if run is None:
            continue
        result = json.loads((run / "selector_ablation.json").read_text(encoding="utf-8"))["datasets"]
        for dataset in datasets:
            entry = result[dataset]
            if not entry["full_reproduces_main"]["tau_equal"]:
                raise SystemExit(f"{row} / {dataset}: the retrained selector does not reproduce the frozen one")
            variants = {"always": entry["reference"]["estimated"], "gate": entry["reference"]["gated"],
                        **{name: entry["variants"][name] for name, _ in SELECTOR_VARIANTS[2:]}}
            cells.append({"row": row, "detector": detector, "dataset": dataset, "source": run.as_posix(),
                          "values": {name: {"clean_change": -v["clean_drop"],
                                            "gain": [v[f"gain_{k}"] for k in (1, 2, 3)],
                                            "tau": v.get("tau")}
                                     for name, v in variants.items()}})
    return cells


def selector_summary(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for name, label in SELECTOR_VARIANTS:
        values = [c["values"][name] for c in cells]
        drops = [-v["clean_change"] for v in values]
        summary.append({
            "variant": name, "label": label, "cells": len(values),
            "mean_gain": [sum(v["gain"][k] for v in values) / len(values) for k in range(3)],
            "mean_clean_change": -sum(drops) / len(drops),
            "worst_clean_change": -max(drops),
            "unsafe_cells": sum(d > UNSAFE_CLEAN_DROP for d in drops),
        })
    return summary


def render_selector(cells: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]]) -> str:
    lines = [f"Table: selector ablation (test sets), mean over {len(cells)} detector-dataset cells. "
             f"Unsafe: cells whose clean F1 drops by more than {UNSAFE_CLEAN_DROP * 100:.1f} points.", "",
             "| Decision | Clean change (mean) | Clean change (worst) | Unsafe cells | Gain L1 | Gain L2 | Gain L3 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['label']} | {pts(s['mean_clean_change'])} | {pts(s['worst_clean_change'])} | "
                     f"{s['unsafe_cells']}/{s['cells']} | " + " | ".join(pts(g) for g in s["mean_gain"]) + " |")
    lines += ["", "Per cell (clean change / gain at level 3):", "",
              "| Detector | Dataset | " + " | ".join(label for _, label in SELECTOR_VARIANTS) + " |",
              "|---|---|" + "---:|" * len(SELECTOR_VARIANTS)]
    for c in cells:
        lines.append(f"| {c['detector']} | {c['dataset']} | " + " | ".join(
            f"{pts(c['values'][name]['clean_change'])} / {pts(c['values'][name]['gain'][2])}"
            for name, _ in SELECTOR_VARIANTS) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- efficiency

def render_efficiency(result: dict[str, Any]) -> tuple[str, list[list[Any]]]:
    localizer = result["localizer_parameters"]
    lines = [f"Table: parameters and time per {result['crop_size']} px pair on {result['device']} (median). "
             f"Localizer: {localizer['total'] / 1e6:.2f} M parameters ({localizer['trainable'] / 1e6:.2f} M trainable); "
             f"RoMa + similarity fit: {result['roma_similarity_ms']['median_ms']:.1f} ms.", "",
             "| Detector | Detector parameters (M) | Selector parameters | Localizer, both directions (ms) | "
             "Warp (ms) | Selector (ms) | Detector, two passes (ms) | Total (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for name, entry in result["rows"].items():
        timing = entry["timing"]
        values = [entry["detector_parameters"]["total"] / 1e6, entry["selector_parameters"],
                  timing["localizer_both_directions"]["median_ms"], timing["warp"]["median_ms"],
                  timing["selector"]["median_ms"], timing["detector_two_passes"]["median_ms"],
                  timing["total_median_ms"]]
        rows.append([name, *values])
        lines.append(f"| {name} | {values[0]:.1f} | {values[1]} | {values[2]:.1f} | {values[3]:.2f} | "
                     f"{values[4]:.1f} | {values[5]:.1f} | {values[6]:.1f} |")
    return "\n".join(lines) + "\n", rows


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else (f"{v:.6f}" if isinstance(v, float) else v) for v in row])


def build_ablation_tables(ablations: Path, output_dir: Path) -> list[str]:
    """Writes whichever tables have inputs; returns their names."""
    written = []
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = localizer_rows(ablations / "localizer" / "test")
    if rows:
        (output_dir / "table_p27_localizer_ablation.md").write_bytes(render_localizer(rows).encode("utf-8"))
        write_csv(output_dir / "table_p27_localizer_ablation.csv",
                  ["dataset", "variant", "success_l1", "success_l2", "success_l3", "clean_change",
                   "gain_l1", "gain_l2", "gain_l3"],
                  [[r["dataset"], r["variant"], *r["success"], r["clean_change"], *r["gain"]] for r in rows])
        written.append("localizer")
    cells = selector_cells(ablations / "selector")
    if cells:
        summary = selector_summary(cells)
        (output_dir / "table_p27_selector_ablation.md").write_bytes(
            render_selector(cells, summary).encode("utf-8"))
        write_csv(output_dir / "table_p27_selector_ablation.csv",
                  ["detector", "dataset", "variant", "tau", "clean_change", "gain_l1", "gain_l2", "gain_l3", "source"],
                  [[c["detector"], c["dataset"], name, c["values"][name]["tau"], c["values"][name]["clean_change"],
                    *c["values"][name]["gain"], c["source"]] for c in cells for name, _ in SELECTOR_VARIANTS])
        written.append(f"selector ({len(cells)} cells)")
    efficiency = latest_file(ablations / "efficiency")
    if efficiency is not None:
        text, rows = render_efficiency(json.loads(efficiency.read_text(encoding="utf-8")))
        (output_dir / "table_p27_efficiency.md").write_bytes(text.encode("utf-8"))
        write_csv(output_dir / "table_p27_efficiency.csv",
                  ["detector", "detector_parameters_m", "selector_parameters", "localizer_ms", "warp_ms",
                   "selector_ms", "detector_two_passes_ms", "total_ms"], rows)
        written.append("efficiency")
    return written


def latest_file(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    return files[-1] if files else None
