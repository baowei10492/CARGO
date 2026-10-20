"""Assemble the border-free qualitative figure so that the decision is visible.

    python -m tools.p31_figure_qualitative <qualitative dir> --output paper/figures/figure_p31_qualitative \
        --pick configs/p31_qualitative_figure.json

tools/p27_figure_qualitative.py draws three error maps: without correction,
localizer only, and "ours". The third is never a new image -- the selector
either adopts the corrected prediction or keeps the raw one, so that column
always duplicates one of its neighbours and the figure shows seven columns
with six distinct panels. Worse, the one thing the figure exists to show, the
per-pair decision, is only implied by which neighbour the duplicate matches.

This version drops the duplicate and marks the decision instead: the panel the
selector adopted is framed, the other is dimmed, and a column on the right
states the probability against the calibrated threshold and the F1 that
follows. A row is then readable as "these were the two options, this is what
the selector took, this is what it cost or saved".

Needs only matplotlib and Pillow: it reads the tiles rendered by
tools/p31_qualitative.py and runs no model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

COLUMNS = (
    ("t1", "T1"),
    ("t2_input", "T2 (input)"),
    ("t2_corrected", "T2 registered"),
    ("reference", "Ground truth"),
    ("without_correction", "Keep: no correction"),
    ("localizer_only", "Adopt: corrected"),
)
OPTIONS = ("without_correction", "localizer_only")
LEGEND = (("TP", (1.0, 1.0, 1.0)), ("FP", (220 / 255, 40 / 255, 40 / 255)),
          ("FN", (40 / 255, 90 / 255, 230 / 255)), ("Not scored", (0.5, 0.5, 0.5)))
ADOPTED = "#1a7f37"      # frame of the panel the selector took
REJECTED = "#8a8a8a"


def adopted_view(entry: dict[str, Any]) -> str:
    """Which of the two options the selector took, from its own probability and threshold."""
    return "localizer_only" if float(entry["selector_probability"]) >= float(entry["tau"]) else "without_correction"


def row_label(entry: dict[str, Any], letter: str) -> str:
    level = int(entry["level"])
    condition = "aligned" if level == 0 else f"level {level}"
    return f"({letter}) {entry['dataset']}\n{condition}"


def decision_lines(entry: dict[str, Any]) -> tuple[str, str, str]:
    """(what it did, the probability against the threshold, the two outcomes)."""
    taken = adopted_view(entry)
    f1 = entry["rendered_f1"]
    ours = f1["with_selector"]
    other = f1["localizer_only"] if taken == "without_correction" else f1["without_correction"]
    verb = "adopt" if taken == "localizer_only" else "keep"
    # Near the threshold two decimals would read as "0.50 < 0.50"; show three there.
    probability, tau = float(entry["selector_probability"]), float(entry["tau"])
    digits = 3 if abs(probability - tau) < 0.005 else 2
    comparison = (f"$p$ = {probability:.{digits}f} {'$\\geq$' if taken == 'localizer_only' else '$<$'} "
                  f"$\\tau$ = {tau:.2f}")
    outcome = f"F1 {ours:.2f}, other {other:.2f}"
    return verb, comparison, outcome


def entries(directory: Path, pick: Sequence[str] | None = None) -> list[dict[str, Any]]:
    index = json.loads((directory / "qualitative_index.json").read_text(encoding="utf-8"))
    if pick:
        by_file = {e["file"]: e for e in index}
        missing = [name for name in pick if name not in by_file]
        if missing:
            raise SystemExit(f"not in {directory}/qualitative_index.json: {missing}")
        return [by_file[name] for name in pick]
    return sorted(index, key=lambda e: e.get("order", 0))


def assemble(directory: Path, rows: Sequence[dict[str, Any]], output: Path, *, tile_inches: float = 0.95,
             font: Sequence[str] = ("Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif")) -> list[Path]:
    columns = [(view, title) for view, title in COLUMNS if view in rows[0]["tiles"]]
    plt.rcParams.update({"font.family": "serif", "font.serif": list(font), "font.size": 7, "pdf.fonttype": 42})
    label_inches, decision_inches, header_inches, legend_inches = 0.78, 1.62, 0.30, 0.32
    width = label_inches + tile_inches * len(columns) + decision_inches
    height = header_inches + tile_inches * len(rows) + legend_inches
    figure = plt.figure(figsize=(width, height))
    gap = 0.03
    for r, entry in enumerate(rows):
        taken = adopted_view(entry)
        bottom = (legend_inches + (len(rows) - 1 - r) * tile_inches + gap / 2) / height
        for c, (view, title) in enumerate(columns):
            left = (label_inches + c * tile_inches + gap / 2) / width
            axis = figure.add_axes((left, bottom, (tile_inches - gap) / width, (tile_inches - gap) / height))
            axis.imshow(np.asarray(Image.open(directory / entry["tiles"][view]).convert("RGB")),
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            chosen = view in OPTIONS and view == taken
            for spine in axis.spines.values():
                spine.set_linewidth(1.6 if chosen else 0.4)
                spine.set_edgecolor(ADOPTED if chosen else ("0.75" if view in OPTIONS else "0.2"))
            if r == 0:
                axis.set_title(title, fontsize=7, pad=3)
            if c == 0:
                axis.text(-0.08, 0.5, row_label(entry, chr(ord("a") + r)), transform=axis.transAxes,
                          ha="right", va="center", fontsize=7)
            if view in OPTIONS:
                axis.text(0.04, 0.045, f"{entry['rendered_f1'][view]:.2f}", transform=axis.transAxes,
                          fontsize=8.5, color="white", ha="left", va="bottom", fontweight="bold",
                          bbox=dict(boxstyle="square,pad=0.18", facecolor="black", edgecolor="none", alpha=0.75))
        verb, comparison, outcome = decision_lines(entry)
        axis = figure.add_axes(((label_inches + len(columns) * tile_inches + 0.06) / width, bottom,
                                (decision_inches - 0.1) / width, (tile_inches - gap) / height))
        axis.axis("off")
        if r == 0:
            axis.set_title("Selector", fontsize=7, pad=3)
        axis.add_patch(Rectangle((0.0, 0.60), 0.40, 0.20, transform=axis.transAxes, facecolor=ADOPTED,
                                 edgecolor="none", alpha=0.12))
        axis.text(0.03, 0.70, verb, transform=axis.transAxes, fontsize=8.5, va="center", color=ADOPTED,
                  fontweight="bold")
        axis.text(0.0, 0.44, comparison, transform=axis.transAxes, fontsize=7, va="center")
        axis.text(0.0, 0.21, outcome, transform=axis.transAxes, fontsize=7, va="center")
    handles = [Patch(facecolor=colour, edgecolor="black", linewidth=0.4, label=name) for name, colour in LEGEND]
    handles.append(Patch(facecolor="none", edgecolor=ADOPTED, linewidth=1.6, label="Adopted by the selector"))
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7,
                  bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        path = output.with_suffix(suffix)
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="a tools.p31_qualitative output directory (with tiles/)")
    parser.add_argument("--output", required=True, help="output path without suffix")
    parser.add_argument("--pick", default=None, help="JSON list of panel file names (index 'file'), in row order")
    args = parser.parse_args()
    directory = Path(args.directory)
    pick = json.loads(Path(args.pick).read_text(encoding="utf-8")) if args.pick else None
    rows = entries(directory, pick)
    for entry in rows:
        verb, probability, outcome = decision_lines(entry)
        print(f"{entry['file']:32s} {verb:6s} {probability:28s} {outcome}")
    for path in assemble(directory, rows, Path(args.output)):
        print(path)
    drift = max(float(e.get("rendered_drift", 0.0)) for e in rows)
    if drift > 0.01:
        print(f"note: rendered crop F1 differs from the test records by up to {drift:.3f}")


if __name__ == "__main__":
    main()
