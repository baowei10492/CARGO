"""Assemble the P27 qualitative figure from a tools.p27_qualitative output directory.

Needs only matplotlib and Pillow (no GPU, no model). Rows follow the "order"
field of qualitative_index.json; columns are the views present in the tiles.
The crop F1 of every error view is printed in its lower-left corner. Writes a
PDF (vector text) and a 300 dpi PNG.
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
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

COLUMNS = (
    ("t1", "T1"),
    ("t2_input", "T2 (input)"),
    ("t2_corrected", "T2 registered"),
    ("reference", "Ground truth"),
    ("without_correction", "No registration"),
    ("localizer_only", "Localizer only"),
    ("with_selector", "Ours"),
)
ERROR_VIEWS = {"without_correction", "localizer_only", "with_selector"}
LEGEND = (("TP", (1.0, 1.0, 1.0)), ("FP", (220 / 255, 40 / 255, 40 / 255)),
          ("FN", (40 / 255, 90 / 255, 230 / 255)), ("Not scored", (0.5, 0.5, 0.5)))


def row_label(entry: dict[str, Any], letter: str) -> str:
    level = int(entry["level"])
    condition = "no perturbation" if level == 0 else f"level {level}"
    return f"({letter}) {entry['dataset']}\n{condition}"


def entries(directory: Path, pick: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The selected rows; with ``pick`` (panel file names), those rows in that order."""
    index = json.loads((directory / "qualitative_index.json").read_text(encoding="utf-8"))
    if pick:
        by_file = {e["file"]: e for e in index}
        missing = [name for name in pick if name not in by_file]
        if missing:
            raise SystemExit(f"not in {directory}/qualitative_index.json: {missing}")
        return [by_file[name] for name in pick]
    chosen = [e for e in index if e.get("category") == "selected"] or index
    return sorted(chosen, key=lambda e: e.get("order", 0))


def assemble(directory: Path, rows: Sequence[dict[str, Any]], output: Path, *, tile_inches: float = 0.95,
             font: Sequence[str] = ("Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif")) -> list[Path]:
    columns = [(view, title) for view, title in COLUMNS if view in rows[0]["tiles"]]
    plt.rcParams.update({"font.family": "serif", "font.serif": list(font), "font.size": 7, "pdf.fonttype": 42})
    label_inches, header_inches, legend_inches = 0.82, 0.22, 0.3
    width = label_inches + tile_inches * len(columns)
    height = header_inches + tile_inches * len(rows) + legend_inches
    figure = plt.figure(figsize=(width, height))
    gap = 0.03
    for r, entry in enumerate(rows):
        for c, (view, title) in enumerate(columns):
            left = (label_inches + c * tile_inches + gap / 2) / width
            bottom = (legend_inches + (len(rows) - 1 - r) * tile_inches + gap / 2) / height
            axis = figure.add_axes((left, bottom, (tile_inches - gap) / width, (tile_inches - gap) / height))
            axis.imshow(np.asarray(Image.open(directory / entry["tiles"][view]).convert("RGB")), interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.4)
            if r == 0:
                axis.set_title(title, fontsize=7, pad=3)
            if c == 0:
                axis.text(-0.08, 0.5, row_label(entry, chr(ord("a") + r)), transform=axis.transAxes,
                          ha="right", va="center", fontsize=7)
            if view in ERROR_VIEWS:
                axis.text(0.03, 0.04, f"F1 {entry['rendered_f1'][view]:.2f}", transform=axis.transAxes,
                          fontsize=6, color="white", ha="left", va="bottom",
                          bbox=dict(boxstyle="square,pad=0.15", facecolor="black", edgecolor="none", alpha=0.7))
    handles = [Patch(facecolor=colour, edgecolor="black", linewidth=0.4, label=name) for name, colour in LEGEND]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7,
                  bbox_to_anchor=(0.5 + label_inches / width / 2, 0.0), handlelength=1.2, columnspacing=1.6)
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
    parser.add_argument("directory", help="a tools.p27_qualitative output directory (with tiles/)")
    parser.add_argument("--output", required=True, help="output path without suffix")
    parser.add_argument("--pick", default=None, help="JSON list of panel file names (index 'file'), in row order")
    args = parser.parse_args()
    directory = Path(args.directory)
    pick = json.loads(Path(args.pick).read_text(encoding="utf-8")) if args.pick else None
    rows = entries(directory, pick)
    for path in assemble(directory, rows, Path(args.output)):
        print(path)
    drift = max(float(e.get("rendered_drift", 0.0)) for e in rows)
    if drift > 0.01:
        print(f"note: rendered crop F1 differs from the test records by up to {drift:.3f}")


if __name__ == "__main__":
    main()
