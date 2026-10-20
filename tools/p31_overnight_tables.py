"""Tables and figures from the overnight session and the swapped-convention run (main record 192.74).

    dose      registration error against the area of pasted change, on self-pairs (figure + csv)
    bins      localization success by the crop's change fraction, and success against the tolerance (figure + csv)
    swapped   DiffRegCD's input convention: the frozen detectors, the augmentation baseline, this work and
              DiffRegCD on the same crops, draws and scoring pixels (md, tex, csv)
    seeds     the selector over five training seeds (md, csv)
    swap      the frozen detectors with the earlier image replaced, next to DiffRegCD's mechanism check (md, csv)

Reads reports/p31/overnight/run, reports/p31/swapped/run, the DiffRegCD re-run and the localizer ablation; writes
paper/data/p31/overnight/ and the figures next to the other paper figures.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
LEVELS = ("0", "1", "2", "3")
ROWS = (("dinov3", "DINOv3 detector"), ("bit", "BIT"), ("cdlamba", "CD-Lamba"), ("cdxlstm", "CDxLSTM"))
OVERNIGHT = Path("reports/p31/overnight/run")
SWAPPED = Path("reports/p31/swapped/run")
DIFFREGCD = Path("reports/p27_localizer/diffregcd/rerun")
VARIANT_LABELS = {"v2_full": "Ours (full)", "v2_no_reliability": "Ours without reliability weighting",
                  "v2_coarse_only": "Ours, coarse matching only", "v2_one_fine_step": "Ours, one refinement step",
                  "roma_similarity": "RoMa + RANSAC", "sift_similarity": "SIFT + RANSAC"}
BINS = ((0.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01))
TOLERANCES = (1, 2, 3, 4, 5, 6, 7, 8)
MECHANISM_VIEWS = ("normal", "same_date", "unrelated", "black")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown(caption: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    align = "|" + "|".join("---" if k == 0 else "---:" for k in range(len(header))) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return f"Table: {caption}\n\n| " + " | ".join(header) + " |\n" + align + "\n" + body + "\n"


def latex(caption: str, label: str, header: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> str:
    # The adjustwidth inside the float lets the table use the outer margin (MDPI class).
    lines = ["\\begin{table}[H]", "\\begin{adjustwidth}{-\\extralength}{0cm}",
             "\\caption{" + caption + "}", "\\label{" + label + "}", "\\centering",
             "\\begin{tabular}{" + align + "}", "\\toprule",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(v) for v in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustwidth}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * float(value):.1f}"


# ----------------------------------------------------------------------------------------------------- dose
def dose_rows() -> list[dict[str, Any]]:
    rows = [r for r in read_csv(OVERNIGHT / "dose" / "dose_summary.csv") if r["dataset"] == "all"]
    return [{"variant": r["variant"], "level": int(r["level"]), "fraction": float(r["fraction"]),
             "pairs": int(r["pairs"]), "success": float(r["success"]), "median_error": float(r["median_error"]),
             "mean_error": float(r["mean_error"])} for r in rows]


# ----------------------------------------------------------------------------------------------------- bins
def crop_errors(variant: str) -> list[dict[str, float]]:
    out = []
    for row in read_csv(OVERNIGHT / "errors" / f"errors_{variant}.csv"):
        direct = float(row["ate_raw"])
        composed = float(row["ate_diff"]) if row["ate_diff"] not in ("", "None") else None
        out.append({"dataset": row["dataset"], "level": int(row["level"]),
                    "change_fraction": float(row["change_fraction"]),
                    "error": direct if composed is None else min(direct, composed)})
    return out


def bin_of(fraction: float) -> str:
    for low, high in BINS:
        if (low == high == 0.0 and fraction == 0.0) or (low < fraction <= high):
            return "0" if low == high == 0.0 else f"{low:.2f}-{high:.2f}"
    return f"{BINS[-1][0]:.2f}-{BINS[-1][1]:.2f}"


def bin_table(variants: Sequence[str], level: int = 3, tolerance: float = 2.0) -> list[dict[str, Any]]:
    out = []
    for variant in variants:
        rows = [r for r in crop_errors(variant) if r["level"] == level]
        for low, high in BINS:
            name = "0" if low == high == 0.0 else f"{low:.2f}-{high:.2f}"
            members = [r for r in rows if bin_of(r["change_fraction"]) == name]
            if not members:
                continue
            errors = np.array([r["error"] for r in members])
            out.append({"variant": variant, "level": level, "change_fraction_bin": name, "crops": len(members),
                        "success": float((errors <= tolerance).mean()), "median_error": float(np.median(errors))})
    return out


def tolerance_table(variants: Sequence[str], level: int = 3) -> list[dict[str, Any]]:
    out = []
    for variant in variants:
        errors = np.array([r["error"] for r in crop_errors(variant) if r["level"] == level])
        out.append({"variant": variant, "crops": len(errors),
                    **{f"success_{t}px": float((errors <= t).mean()) for t in TOLERANCES}})
    return out


# -------------------------------------------------------------------------------------------------- swapped
def diffregcd_border_free() -> dict[str, dict[str, float]]:
    root = Path((DIFFREGCD / "LATEST").read_text(encoding="utf-8").strip())
    out = {}
    for name in DATASETS:
        f1 = json.loads((root / "border_free" / f"{name}.json").read_text(encoding="utf-8"))["f1"]
        out[name] = {level: float(f1[f"{level}/official_unmodified/scoring"]) for level in LEVELS}
    return out


def swapped_table() -> tuple[list[dict[str, Any]], list[Sequence[str]]]:
    diffreg = diffregcd_border_free()
    rows: list[dict[str, Any]] = []
    for row, label in ROWS:
        entry = json.loads((SWAPPED / f"{row}.json").read_text(encoding="utf-8"))["datasets"]
        for name in DATASETS:
            values = entry[name]["f1"]
            rows.append({"detector": label, "dataset": name, "setting": "Without correction",
                         **{level: values[level]["raw"] for level in LEVELS}})
            if row == "dinov3":
                augmented = entry[name]["p30_augmented_raw"]
                rows.append({"detector": label, "dataset": name, "setting": "Augmentation baseline",
                             **{level: augmented[level] for level in LEVELS}})
            rows.append({"detector": label, "dataset": name, "setting": "+ localizer + selector",
                         **{level: values[level]["selector"] for level in LEVELS}})
            rows.append({"detector": label, "dataset": name, "setting": "Oracle correction",
                         **{level: values[level]["oracle"] for level in LEVELS}})
            if row == "dinov3":
                rows.append({"detector": "DiffRegCD", "dataset": name, "setting": "Joint model",
                             **{level: diffreg[name][level] for level in LEVELS}})
    table = [[r["detector"], r["dataset"], r["setting"], *(pct(r[level]) for level in LEVELS)] for r in rows]
    return rows, table


# ---------------------------------------------------------------------------------------------------- seeds
def seed_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads((OVERNIGHT / "selector_seeds.json").read_text(encoding="utf-8"))
    rows = []
    for cell, entry in data["per_cell"].items():
        rows.append({"cell": cell, "clean_change_mean": -entry["clean_drop"]["mean"],
                     "clean_change_std": entry["clean_drop"]["std"],
                     **{f"gain_{k}_{key}": entry[f"gain_{k}"][key] for k in (1, 2, 3) for key in ("mean", "std")}})
    return rows, data["over_seeds"]


# ----------------------------------------------------------------------------------------------- input swap
def input_swap_rows() -> list[dict[str, Any]]:
    data = json.loads((OVERNIGHT / "input_swap.json").read_text(encoding="utf-8"))["rows"]
    root = Path((DIFFREGCD / "LATEST").read_text(encoding="utf-8").strip())
    rows = []
    for row, label in ROWS:
        for name in DATASETS:
            f1 = data[row][name]["f1"]
            rows.append({"model": label, "dataset": name, **{view: f1[view] for view in MECHANISM_VIEWS}})
    for name in DATASETS:
        f1 = json.loads((root / "mechanism" / f"{name}.json").read_text(encoding="utf-8"))["f1"]
        rows.append({"model": "DiffRegCD", "dataset": name,
                     **{view: float(f1[f"{view}/official_unmodified/scoring"]) for view in MECHANISM_VIEWS}})
    return rows


# -------------------------------------------------------------------------------------------------- figures
def figures(dose: Sequence[dict[str, Any]], bins: Sequence[dict[str, Any]], tolerances: Sequence[dict[str, Any]],
            output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42})
    styles = {"v2_full": ("tab:blue", "-"), "v2_no_reliability": ("tab:orange", "--"),
              "roma_similarity": ("tab:green", "-."), "sift_similarity": ("0.5", ":")}
    written = []

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for variant in ("v2_full", "v2_no_reliability", "roma_similarity"):
        colour, style = styles[variant]
        for axis, level, marker in ((axes[0], 3, "o"), (axes[1], 0, "s")):
            chosen = sorted((r for r in dose if r["variant"] == variant and r["level"] == level),
                            key=lambda r: r["fraction"])
            axis.plot([100 * r["fraction"] for r in chosen], [r["median_error"] for r in chosen], style, marker=marker,
                      ms=3, color=colour, lw=1.0, label=VARIANT_LABELS[variant])
    axes[0].set_title("Misaligned pairs (level 3)")
    axes[1].set_title("Aligned pairs")
    for axis in axes:
        axis.set_xlabel("Pasted change, share of the crop (%)")
        axis.set_yscale("log")
        axis.axhline(2.0, color="0.6", lw=0.6, ls="--")
    axes[0].set_ylabel("Median transfer error (px)")
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        path = output_dir / f"figure_p31_dose_response{suffix}"
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    order = [b for b in dict.fromkeys(r["change_fraction_bin"] for r in bins)]
    for variant in ("v2_full", "v2_no_reliability", "roma_similarity", "sift_similarity"):
        colour, style = styles[variant]
        chosen = [next((r for r in bins if r["variant"] == variant and r["change_fraction_bin"] == b), None)
                  for b in order]
        axes[0].plot(range(len(order)), [100 * r["success"] if r else np.nan for r in chosen], style, marker="o",
                     ms=3, color=colour, lw=1.0, label=VARIANT_LABELS[variant])
        entry = next(r for r in tolerances if r["variant"] == variant)
        axes[1].plot(TOLERANCES, [100 * entry[f"success_{t}px"] for t in TOLERANCES], style, marker="o", ms=3,
                     color=colour, lw=1.0)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(["none" if b == "0" else b.replace("-", "–") for b in order], fontsize=6.5)
    axes[0].set_xlabel("Change in the crop (share of pixels)")
    axes[0].set_ylabel("Localization success (%)")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].set_xlabel("Tolerance (px)")
    axes[1].set_ylabel("Localization success (%)")
    axes[1].axvline(2.0, color="0.6", lw=0.6, ls="--")
    figure.tight_layout()
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        path = output_dir / f"figure_p31_localization{suffix}"
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper/data/p31/overnight")
    parser.add_argument("--figure-dir", default="paper/figures")
    args = parser.parse_args()
    output_dir, figure_dir = Path(args.output_dir), Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    variants = ("v2_full", "v2_no_reliability", "v2_coarse_only", "v2_one_fine_step", "roma_similarity",
                "sift_similarity")

    dose = dose_rows()
    write_csv(output_dir / "table_dose.csv", dose)
    bins = bin_table(variants)
    write_csv(output_dir / "table_success_bins.csv", bins)
    tolerances = tolerance_table(variants)
    write_csv(output_dir / "table_success_tolerance.csv", tolerances)

    rows, table = swapped_table()
    write_csv(output_dir / "table_swapped.csv", rows)
    caption = ("F1 (\\%) under DiffRegCD's input convention (the earlier image is perturbed, the later one is the "
               "reference and carries the label frame), test sets, border-free perturbation, identical crops, draws "
               "and scoring pixels for every row.")
    header = ["Model", "Dataset", "Setting", "Aligned", "Level 1", "Level 2", "Level 3"]
    (output_dir / "table_swapped.md").write_text(markdown(caption.replace("\\%", "%"), header, table), encoding="utf-8")
    # Two LaTeX tables: the DiffRegCD comparison for the main text, the plug-in detectors for the appendix
    # (43 rows do not fit one page); a repeated model or dataset label is left blank.
    def compact(rows: Sequence[Sequence[str]]) -> list[list[str]]:
        out, last = [], ("", "")
        for row in rows:
            key = (row[0], row[1])
            out.append([row[0] if row[0] != last[0] else "", row[1] if key != last else "", *row[2:]])
            last = key
        return out
    main_rows = [row for row in table if row[0] in ("DINOv3 detector", "DiffRegCD")]
    plugin_rows = [row for row in table if row[0] not in ("DINOv3 detector", "DiffRegCD")]
    (output_dir / "table_swapped.tex").write_text(
        latex(caption, "tab:swapped", header, compact(main_rows), "lllrrrr"), encoding="utf-8")
    (output_dir / "table_swapped_plugins.tex").write_text(
        latex(caption.replace("for every row.", "for every row; the plug-in detectors, with the selectors trained "
                              "under the other convention and applied unchanged."),
              "tab:swapped_plugins", header, compact(plugin_rows), "lllrrrr"), encoding="utf-8")

    seeds, over = seed_rows()
    write_csv(output_dir / "table_seeds.csv", seeds)
    (output_dir / "table_seeds.md").write_text(markdown(
        "The selector retrained with five classifier seeds (test sets, border-free misalignment); "
        "F1 change in points, mean and standard deviation over seeds.",
        ["Cell", "Aligned", "Level 1", "Level 2", "Level 3"],
        [[r["cell"], f"{100 * r['clean_change_mean']:+.2f} ± {100 * r['clean_change_std']:.2f}",
          *(f"{100 * r[f'gain_{k}_mean']:+.2f} ± {100 * r[f'gain_{k}_std']:.2f}" for k in (1, 2, 3))]
         for r in seeds]), encoding="utf-8")

    swap = input_swap_rows()
    write_csv(output_dir / "table_input_swap.csv", swap)
    (output_dir / "table_input_swap.md").write_text(markdown(
        "F1 (%) when the earlier image is replaced (aligned test crops for the frozen detectors; DiffRegCD's "
        "mechanism check, where its earlier image also carries the level-3 perturbation, for reference).",
        ["Model", "Dataset", "Normal", "Later image", "Another tile", "Zeros"],
        [[r["model"], r["dataset"], *(pct(r[view]) for view in MECHANISM_VIEWS)] for r in swap]), encoding="utf-8")

    for path in figures(dose, bins, tolerances, figure_dir):
        print(path)
    print((output_dir / "table_swapped.md").read_text(encoding="utf-8"))
    print(json.dumps(over, indent=1))


if __name__ == "__main__":
    main()
