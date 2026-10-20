"""Paper tables under the border-free protocol (test split), read from the frozen result files.

    main        per dataset: detectors without correction, the border-free augmentation baseline,
                DiffRegCD (its input convention) and the proposed pipeline
    generality  the four detectors x three datasets, without / with the localizer and selector
    localizer   localizer variants and matching baselines, every pair corrected
    selector    decision rules and feature subsets, mean over the 12 cells and per cell
    border      methods under the per-crop zero-border protocol and under the border-free protocol

Sources (LATEST pointers): reports/p30/test, reports/p31/<row>/test, reports/p27_localizer/diffregcd/rerun,
reports/p31/localizer_ablation, reports/p31/selector_ablation, paper/data/p27/table_p27_main_comparison.csv.
Values are F1 in percent with one decimal; changes are in points.

    python -m tools.p31_paper_tables --output-dir paper/data/p31
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from tools.p27_ablation_tables import SELECTOR_VARIANTS, selector_cells, selector_summary
from tools.p31_selector_ablation import BORDER_FREE_ROWS

DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
LEVELS = ("0", "1", "2", "3")
PLUGINS = (("bit", "BIT"), ("cdlamba", "CD-Lamba"), ("cdxlstm", "CDxLSTM"),
           ("snunet", "SNUNet"), ("bifa", "BiFA"))
# P32 added the last two under the same protocol; their reports live elsewhere
# so that the frozen P31 rows keep their directories (docs/P32_PLUGIN_EXTENSION.md).
PLUGIN_ROOTS = {"snunet": Path("reports/p32"), "bifa": Path("reports/p32")}
DIFFREG = "official_unmodified/scoring"
LOCALIZERS = (("v2_full", "Ours (full)"), ("v2_one_fine_step", "one refinement step"),
              ("v2_coarse_only", "coarse matching only"), ("v2_no_reliability", "no reliability weighting"),
              ("roma_similarity", "RoMa + similarity fit"), ("sift_similarity", "SIFT + similarity fit"))


# ------------------------------------------------------------------ sources
def pointed(path: str | Path) -> Path:
    return Path(Path(path).read_text(encoding="utf-8").strip())


def load_json(path: Path, used: dict[str, str]) -> dict[str, Any]:
    used[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def sources(used: dict[str, str]) -> dict[str, Any]:
    p30 = load_json(pointed("reports/p30/test/LATEST") / "p30_test.json", used)["datasets"]
    plugins = {row: load_json(PLUGIN_ROOTS.get(row, Path("reports/p31")) / row / "test" / "p31_test.json",
                              used)["datasets"]
               for row, _ in PLUGINS}
    rerun = pointed("reports/p27_localizer/diffregcd/rerun/LATEST")
    diffreg = {part: {name: load_json(rerun / part / f"{name}.json", used)["f1"] for name in DATASETS}
               for part in ("border_free", "bordered")}
    ablation = pointed("reports/p31/localizer_ablation/LATEST")
    localizer = {key: load_json(ablation / f"{key}.json", used) for key, _ in LOCALIZERS}
    final = sorted(p for p in Path("reports/p31/final").iterdir() if p.is_dir())[-1]
    border_counts = load_json(final / "border_counts" / "summary.json", used)
    if not all(entry["anchor"]["passed"] for entry in border_counts["datasets"].values()):
        raise SystemExit(f"{final}: the border counts did not reproduce the P30 test rows")
    timing = load_json(final / "diffregcd_timing.json", used)
    efficiency = load_json(sorted(Path("reports/p27_localizer/ablations/efficiency").glob("*.json"))[-1], used)
    return {"p30": p30, "plugins": plugins, "diffreg": diffreg, "localizer": localizer,
            "border_counts": border_counts, "diffreg_timing": timing, "efficiency": efficiency}


def levels_of(values: dict[str, float]) -> list[float]:
    return [float(values[level]) for level in LEVELS]


def diffreg_levels(f1: dict[str, float]) -> list[float]:
    return [float(f1[f"{level}/{DIFFREG}"]) for level in LEVELS]


def with_mean(values: Sequence[float]) -> list[float]:
    return [*values, sum(values[1:]) / 3.0]


# ------------------------------------------------------------------ format
def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def signed(value: float | None) -> str:
    if value is None:
        return "-"
    rounded = round(value * 100, 1)
    return "0.0" if rounded == 0 else f"{rounded:+.1f}"


def best_marks(block: Sequence[Sequence[float | None]]) -> list[list[bool]]:
    columns = len(block[0])
    best = [max(row[c] for row in block if row[c] is not None) for c in range(columns)]
    return [[row[c] is not None and abs(row[c] - best[c]) < 5e-7 for c in range(columns)] for row in block]


def markdown(caption: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    text_columns = sum(1 for cell in header if cell in ("Dataset", "Handling", "Method", "Detector", "Setting",
                                                         "Localizer", "Decision", "Strategy", "Retrains"))
    lines = [f"Table: {caption}", "", "| " + " | ".join(header) + " |",
             "|" + "|".join("---" if i < text_columns else "---:" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def latex(caption: str, label: str, header: Sequence[str], rows: Sequence[Sequence[str]], *,
          align: str, groups: Sequence[int] = (), spans: Sequence[tuple[str, int]] = ()) -> str:
    """A booktabs table (\\toprule / \\midrule / \\bottomrule, as in the MDPI template); groups = row indices
    before which a \\midrule is drawn. The adjustwidth inside the float lets a wide table use the outer
    margin, the way the MDPI class expects; it has no effect on a table that already fits."""
    body = []
    for index, row in enumerate(rows):
        if index in groups:
            body.append("\\midrule")
        body.append(" & ".join(cell.replace("%", "\\%") for cell in row) + " \\\\")
    # A bare % in a caption comments out the rest of the line; body cells are escaped below.
    caption = caption.replace("\\%", "%").replace("%", "\\%")
    # A spanning row ("Aligned" over three columns) when repeated column names would be ambiguous.
    top, rules, column = [], [], 1
    for title, width in spans:
        top.append(f"\\multicolumn{{{width}}}{{c}}{{\\textbf{{{title}}}}}" if title else "")
        if title and width > 1:
            rules.append(f"\\cmidrule(lr){{{column}-{column + width - 1}}}")
        column += width
    header_rows = [" & ".join(top) + " \\\\", " ".join(rules)] if spans else []
    return "\n".join([
        "\\begin{table}[H]", "\\begin{adjustwidth}{-\\extralength}{0cm}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering",
        f"\\begin{{tabular}}{{{align}}}", "\\toprule", *header_rows,
        " & ".join(f"\\textbf{{{cell}}}" for cell in header) + " \\\\", "\\midrule",
        *body, "\\bottomrule", "\\end{tabular}", "\\end{adjustwidth}", "\\end{table}", ""])


def bold(text: str, flag: bool, style: str) -> str:
    if not flag:
        return text
    return f"**{text}**" if style == "md" else f"\\textbf{{{text}}}"


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else (f"{v:.6f}" if isinstance(v, float) else v) for v in row])


# ------------------------------------------------------------------ tables
def alternatives_rows(data: dict[str, Any]) -> list[tuple[str, str, list[float], list[float]]]:
    """(strategy, does it retrain the detector, clean F1 per dataset, level-3 F1 per dataset).

    The four ways to face misalignment with one detector, so that what each one costs is visible in
    a single table: leave it, correct every pair, retrain the detector with misalignment augmentation,
    or attach the proposed front end. The DINOv3 row is used because it is the only detector we
    retrained ourselves; Table~\\ref{tab:generality} carries the frozen detectors.
    """
    def per_dataset(pick) -> list[float]:
        return [float(pick(data["p30"][name])) for name in DATASETS]

    return [
        ("No correction", "--",
         per_dataset(lambda e: e["original"]["0"]), per_dataset(lambda e: e["original"]["3"])),
        ("Every pair corrected (our localizer)", "no",
         per_dataset(lambda e: e["proposed"]["f1"]["0"]["estimated"]),
         per_dataset(lambda e: e["proposed"]["f1"]["3"]["estimated"])),
        ("Detector retrained with misalignment augmentation", "yes",
         per_dataset(lambda e: e["p30_augmented"]["0"]), per_dataset(lambda e: e["p30_augmented"]["3"])),
        ("Ours: frozen detector + localizer + selector", "no",
         per_dataset(lambda e: e["proposed"]["f1"]["0"]["selector"]),
         per_dataset(lambda e: e["proposed"]["f1"]["3"]["selector"])),
    ]


def render_alternatives(rows: list[tuple[str, str, list[float], list[float]]], style: str) -> str:
    header = (["Strategy", "Retrains", "Aligned LEVIR", "Aligned DSIFN", "Aligned WHU",
               "L3 LEVIR", "L3 DSIFN", "L3 WHU"] if style == "md" else
              ["Strategy", "Retrains", "LEVIR", "DSIFN", "WHU", "LEVIR", "DSIFN", "WHU"])
    clean_marks = best_marks([clean for _, _, clean, _ in rows])
    level_marks = best_marks([level for _, _, _, level in rows])
    out = [[label, retrain,
            *(bold(pct(v), m, style) for v, m in zip(clean, clean_marks[i])),
            *(bold(pct(v), m, style) for v, m in zip(level, level_marks[i]))]
           for i, (label, retrain, clean, level) in enumerate(rows)]
    caption = ("F1 (%) of the four ways to face misalignment, on the DINOv3 detector: the first three columns are "
               "aligned pairs, the last three level 3. ``Retrains'' says whether the detector's weights change, which "
               "is what a plug-in must avoid. Correcting every pair recovers level 3 but costs aligned accuracy; "
               "retraining needs the training data and a run per detector and dataset, and on DSIFN it costs 6.4 "
               "points of aligned F1; the proposed front end keeps the detector frozen. Bold: best in each column.")
    if style == "md":
        return markdown(caption.replace("``", '"').replace("''", '"'), header, out)
    return latex(caption, "tab:alternatives", header, out, align="llrrrrrr",
                 spans=(("", 1), ("", 1), ("Aligned pairs", 3), ("Level 3", 3)))


def main_rows(data: dict[str, Any]) -> list[tuple[str, str, str, list[float]]]:
    """(dataset, handling, method, [clean, L1, L2, L3, mean])."""
    rows = []
    for name in DATASETS:
        p30 = data["p30"][name]
        for row, label in PLUGINS:
            rows.append((name, "none", label, with_mean(levels_of(data["plugins"][row][name]["without_correction"]))))
        rows.append((name, "none", "DINOv3 detector", with_mean(levels_of(p30["original"]))))
        rows.append((name, "augmentation", "DINOv3 + augmentation",
                     with_mean(levels_of(p30["p30_augmented"]))))
        rows.append((name, "localizer + selector", "Ours (DINOv3 + front end)",
                     with_mean([float(p30["proposed"]["f1"][level]["selector"]) for level in LEVELS])))
    return rows


def render_main(rows: list[tuple[str, str, str, list[float]]], style: str) -> str:
    header = ["Dataset", "Method", "Clean", "Level 1", "Level 2", "Level 3", "Mean (L1-L3)"]
    out, groups = [], []
    for name in DATASETS:
        block = [r for r in rows if r[0] == name]
        marks = best_marks([r[3] for r in block])
        if out:
            groups.append(len(out))
        for i, (dataset, _handling, method, values) in enumerate(block):
            out.append([dataset if i == 0 else "", method,
                        *(bold(pct(v), m, style) for v, m in zip(values, marks[i]))])
    caption = ("F1 (%) on the test sets under border-free misalignment. Every row uses the same convention: the later "
               "image is warped and the label stays in the earlier image's frame. The six detectors are applied to "
               "the misaligned pair as they are; ``DINOv3 + augmentation'' is the same DINOv3 detector retrained with "
               "border-free misalignment; the proposed front end keeps the DINOv3 detector frozen. DiffRegCD requires "
               "the opposite convention and is compared separately in Table~\\ref{tab:swapped}. Bold: best in each "
               "column of a dataset.")
    if style == "md":
        return markdown(caption.replace("``", '"').replace("''", '"'), header, out)
    return latex(caption, "tab:main", header, out, align="llrrrrr", groups=groups)


def generality_rows(data: dict[str, Any]) -> list[tuple[str, str, list[float], list[float]]]:
    rows = []
    for name in DATASETS:
        p30 = data["p30"][name]
        rows.append(("DINOv3 detector", name, with_mean(levels_of(p30["original"])),
                     with_mean([float(p30["proposed"]["f1"][level]["selector"]) for level in LEVELS])))
    for row, label in PLUGINS:
        for name in DATASETS:
            entry = data["plugins"][row][name]
            rows.append((label, name, with_mean(levels_of(entry["without_correction"])),
                         with_mean(levels_of(entry["with_selector"]))))
    return rows


def render_generality(rows: list[tuple[str, str, list[float], list[float]]], style: str) -> str:
    header = ["Detector", "Dataset", "Setting", "Clean", "Level 1", "Level 2", "Level 3", "Mean (L1-L3)"]
    out, groups, last = [], [], None
    for detector, name, without, with_ in rows:
        if detector != last and out:
            groups.append(len(out))
        first = detector != last
        last = detector
        out.append([detector if first else "", name, "Without correction", *(pct(v) for v in without)])
        out.append(["", "", "+ localizer + selector", *(pct(v) for v in with_)])
        out.append(["", "", "Change", *(signed(b - a) for a, b in zip(without, with_))])
    caption = ("F1 (%) of each frozen detector without correction and with the proposed localizer and selector "
               "(test sets, border-free misalignment); the change is in points.")
    if style == "md":
        return markdown(caption, header, out)
    return latex(caption, "tab:generality", header, out, align="lllrrrrr", groups=groups)


def localizer_rows(data: dict[str, Any]) -> list[tuple[str, str, list[float], float, list[float]]]:
    rows = []
    for name in DATASETS:
        for key, label in LOCALIZERS:
            entry = data["localizer"][key]["datasets"][name]
            success = [float(entry["localization_success"][level]) for level in ("1", "2", "3")]
            rows.append((name, label, success, -float(entry["clean_drop"]["estimated"]),
                         [float(entry["gain"][level]["estimated"]) for level in ("1", "2", "3")]))
    return rows


def render_localizer(rows: list[tuple[str, str, list[float], float, list[float]]], style: str) -> str:
    header = ["Dataset", "Localizer", "Succ. L1", "Succ. L2", "Succ. L3", "Clean", "Gain L1", "Gain L2",
              "Gain L3"]
    out, groups = [], []
    for name in DATASETS:
        if out:
            groups.append(len(out))
        for i, (dataset, label, success, clean, gains) in enumerate(r for r in rows if r[0] == name):
            out.append([dataset if i == 0 else "", label, *(pct(s) for s in success), signed(clean),
                        *(signed(g) for g in gains)])
    caption = ("Localizer ablation and matching baselines (test sets, border-free misalignment, DINOv3 detector, "
               "every pair corrected). Succ.: share of crops aligned within 2 px (%); Clean: F1 change on "
               "aligned pairs and Gain: F1 change at each level, in points relative to no correction.")
    if style == "md":
        return markdown(caption, header, out)
    return latex(caption, "tab:localizer", header, out, align="llrrrrrrr", groups=groups)


def render_selector(summary: list[dict[str, Any]], style: str) -> str:
    header = ["Decision", "Clean (mean)", "Clean (worst)", "Unsafe", "Gain L1", "Gain L2", "Gain L3"]
    out = [[s["label"], signed(s["mean_clean_change"]), signed(s["worst_clean_change"]),
            f"{s['unsafe_cells']}/{s['cells']}", *(signed(g) for g in s["mean_gain"])] for s in summary]
    caption = ("Decision rules and selector features (test sets, border-free misalignment), mean over the 12 "
               "detector-dataset cells of the four original detectors. Clean: F1 change on aligned pairs "
               "(points), mean and worst cell; Unsafe: cells whose clean F1 drops by more than 0.5 points; "
               "Gain: F1 change at each misalignment level (points).")
    if style == "md":
        return markdown(caption, header, out)
    return latex(caption, "tab:selector", header, out, align="lrrrrrr", groups=(2,))


BORDER_GROUPS = (("original", "DINOv3 detector (no misalignment training)"),
                 ("p29_augmented", "DINOv3 + augmentation, zero-border training"),
                 ("p30_augmented", "DINOv3 + augmentation, border-free training"))


def border_rows(data: dict[str, Any]) -> list[tuple[str, str, float | None, float]]:
    """(method, dataset, level-3 F1 with a per-crop zero border, level-3 F1 border-free), on the SAME crops
    and draws (the P30 test crops). Grid-crop numbers of the per-crop protocol are not mixed in: the crop
    sets differ (DSIFN clean F1 of DiffRegCD 68.4 on grid crops, 61.9 on the P30 crops).

    Detectors: tools/p31_border_counts.py (final session); DiffRegCD: bordered_3 of its border-free part."""
    counts = data["border_counts"]["datasets"]
    rows = []
    for group, label in BORDER_GROUPS:
        for name in DATASETS:
            f1 = counts[name]["f1"][group]
            rows.append((label, name, float(f1["zero_border/3"]), float(f1["border_free/3"])))
    for name in DATASETS:
        f1 = data["diffreg"]["border_free"][name]
        rows.append(("DiffRegCD", name, float(f1[f"bordered_3/{DIFFREG}"]), float(f1[f"3/{DIFFREG}"])))
    return rows


def render_border(rows: list[tuple[str, str, float | None, float]], style: str) -> str:
    header = ["Method", "Dataset", "Zero border", "Border-free", "Change"]
    out, groups = [], []
    methods = list(dict.fromkeys(r[0] for r in rows))
    for method in methods:
        if out:
            groups.append(len(out))
        for i, (m, name, bordered, free) in enumerate(r for r in rows if r[0] == method):
            out.append([m if i == 0 else "", name, pct(bordered), pct(free),
                        "-" if bordered is None else signed(free - bordered)])
    caption = ("Level-3 F1 (%) with a per-crop zero border and without it, on the same test crops and draws. "
               "Zero border: the 256 px crop warped with a constant zero border; border-free: the whole image "
               "warped before the crop is cut (crops 64 px from the tile edge).")
    if style == "md":
        return markdown(caption, header, out)
    return latex(caption, "tab:border", header, out, align="llrrr", groups=groups)


EFFICIENCY_ROWS = (("DINOv3 (ours)", "Ours: DINOv3 detector + localizer + selector"),
                   ("BIT", "BIT + localizer + selector"), ("CD-Lamba", "CD-Lamba + localizer + selector"),
                   ("CDxLSTM", "CDxLSTM + localizer + selector"))


def efficiency_rows(data: dict[str, Any]) -> list[tuple[str, float, float, float, str]]:
    """(method, parameters M, trained parameters M, ms per 256 px pair, detector retrained). Same GPU model
    (RTX 4090), batch 1, medians. The proposed modules add the localizer (its trainable part) and the selector."""
    result = data["efficiency"]
    localizer = result["localizer_parameters"]
    rows = []
    for name, label in EFFICIENCY_ROWS:
        entry = result["rows"][name]
        total = (entry["detector_parameters"]["total"] + localizer["total"] + entry["selector_parameters"]) / 1e6
        trained = (localizer["trainable"] + entry["selector_parameters"]) / 1e6
        rows.append((label, total, trained, float(entry["timing"]["total_median_ms"]), "no"))
    timing = data["diffreg_timing"]
    used = timing["parameters"]["diffusion_backbone"] + timing["parameters"]["change_head"]
    trained = timing["parameters"]["change_head"] + timing["parameters"]["registration_head"]
    rows.append(("DiffRegCD (released test path)", used / 1e6, trained / 1e6, float(timing["timing"]["median_ms"]),
                 "jointly, per dataset"))
    return rows


def render_efficiency(rows: list[tuple[str, float, float, float, str]], style: str) -> str:
    header = ["Method", "Params (M)", "Trained (M)", "ms per pair", "Detector retrained"]
    out = [[label, f"{total:.1f}", f"{trained:.2f}", f"{ms:.1f}", retrained] for label, total, trained, ms, retrained in rows]
    caption = ("Parameters and time per 256 px pair (RTX 4090, batch 1, median). Parameters: those used at inference; "
               "trained parameters: those fitted for the task (for the proposed modules, the trainable part of the "
               "localizer and the selector; the detector is not retrained). DiffRegCD: diffusion backbone and change "
               "head of its released test path; its flow head (143.2 M, trained jointly) is not run at test time and "
               "is not timed.")
    if style == "md":
        return markdown(caption, header, out)
    return latex(caption, "tab:efficiency", header, out, align="lrrrl", groups=(len(out) - 1,))


# ------------------------------------------------------------------ main
def build(output_dir: Path) -> list[str]:
    used: dict[str, str] = {}
    data = sources(used)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def emit(stem: str, renderer, rows, header, csv_rows) -> None:
        (output_dir / f"{stem}.md").write_text(renderer(rows, "md"), encoding="utf-8")
        (output_dir / f"{stem}.tex").write_text(renderer(rows, "tex"), encoding="utf-8")
        write_csv(output_dir / f"{stem}.csv", header, csv_rows)
        written.append(stem)

    alternatives = alternatives_rows(data)
    emit("table_alternatives", render_alternatives, alternatives,
         ["strategy", "retrains_detector", "clean_levir", "clean_dsifn", "clean_whu",
          "level3_levir", "level3_dsifn", "level3_whu"],
         [[s, r, *c, *t] for s, r, c, t in alternatives])
    main = main_rows(data)
    emit("table_main", render_main, main, ["dataset", "handling", "method", "f1_clean", "f1_level1", "f1_level2",
                                          "f1_level3", "f1_mean_l1_l3"],
         [[d, h, m, *v] for d, h, m, v in main])
    general = generality_rows(data)
    emit("table_generality", render_generality, general,
         ["detector", "dataset", "setting", "f1_clean", "f1_level1", "f1_level2", "f1_level3", "f1_mean_l1_l3"],
         [row for det, name, a, b in general for row in ([det, name, "without", *a], [det, name, "with", *b])])
    loc = localizer_rows(data)
    emit("table_localizer", render_localizer, loc,
         ["dataset", "localizer", "success_l1", "success_l2", "success_l3", "clean_change", "gain_l1", "gain_l2",
          "gain_l3"], [[d, l, *s, c, *g] for d, l, s, c, g in loc])
    cells = selector_cells(Path("reports/p31/selector_ablation"), rows=BORDER_FREE_ROWS)
    summary = selector_summary(cells)
    emit("table_selector", render_selector, summary,
         ["variant", "label", "cells", "mean_clean_change", "worst_clean_change", "unsafe_cells", "gain_l1",
          "gain_l2", "gain_l3"],
         [[s["variant"], s["label"], s["cells"], s["mean_clean_change"], s["worst_clean_change"], s["unsafe_cells"],
           *s["mean_gain"]] for s in summary])
    write_csv(output_dir / "table_selector_cells.csv",
              ["detector", "dataset", *(f"{name}_{k}" for name, _ in SELECTOR_VARIANTS for k in ("clean", "gain_l3"))],
              [[c["detector"], c["dataset"], *(v for name, _ in SELECTOR_VARIANTS
                                               for v in (c["values"][name]["clean_change"], c["values"][name]["gain"][2]))]
               for c in cells])
    border = border_rows(data)
    emit("table_border", render_border, border, ["method", "dataset", "f1_level3_zero_border", "f1_level3_border_free"],
         [[m, d, b, f] for m, d, b, f in border])
    efficiency = efficiency_rows(data)
    emit("table_efficiency", render_efficiency, efficiency,
         ["method", "parameters_m", "trained_parameters_m", "ms_per_pair", "detector_retrained"],
         [list(row) for row in efficiency])
    (output_dir / "sources.json").write_text(json.dumps(used, indent=2), encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper/data/p31")
    args = parser.parse_args()
    for stem in build(Path(args.output_dir)):
        print(f"{args.output_dir}/{stem}.md|.tex|.csv")


if __name__ == "__main__":
    main()
