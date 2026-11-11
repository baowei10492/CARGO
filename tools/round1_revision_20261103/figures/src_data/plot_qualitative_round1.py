"""Approved qualitative layout extended to four archived cases; no model inference.
Source images and rendered F1/selector scores are checked against archived records.
The first two cases retain their approved images; rows c/d add WHU-CD and native S2Looking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from plot_data_figures import BLUE, INK, MUTED, OUT, RED, plt, setup

RUN = Path(__file__).resolve().parents[1] / "qualitative_data"
QUALITATIVE = ("LEVIR-CD_corrected_1.png", "DSIFN-CD_kept_clean_3.png", "WHU-CD_corrected_2.png", "S2Looking_native_kept.png")
FAILURES = ("LEVIR-CD_failure_1.png", "DSIFN-CD_failure_2.png")

GREEN = "#1A7F37"  # the output frame of Figure 1
ERROR_COLOURS = (("TP", (255, 255, 255)), ("FP", (220, 40, 40)), ("FN", (40, 90, 230)), ("TN", (0, 0, 0)),
                 ("Not scored", (128, 128, 128)))
COLUMNS = (("t1", r"$I_1$"), ("t2_input", r"$I_2$"), ("t2_corrected", r"$I_2^{\hat{T}}$"),
           ("reference", "Ground truth"), ("without_correction", r"$Q_{\mathrm{raw}}$"),
           ("localizer_only", r"$Q_{\mathrm{reg}}$"))
PREDICTIONS = ("without_correction", "localizer_only")


def tile(entry: dict, view: str) -> np.ndarray:
    return np.asarray(Image.open(RUN / entry["tiles"][view]).convert("RGB"))


def check_f1(entry: dict) -> None:
    for view in PREDICTIONS:
        image = tile(entry, view)
        tp, fp, fn = (np.all(image == np.array(c), axis=-1).sum() for c in
                      ((255, 255, 255), (220, 40, 40), (40, 90, 230)))
        f1 = 2 * tp / (2 * tp + fp + fn)
        assert abs(f1 - entry["rendered_f1"][view]) < 1e-6, (entry["file"], view, f1)


def selected(entry: dict) -> str:
    return "localizer_only" if entry["selector_probability"] >= entry["tau"] else "without_correction"


def row_label(entry: dict, letter: str) -> str:
    condition = "unperturbed" if int(entry["level"]) == 0 else f"level {entry['level']}"
    return f"({letter}) {entry['dataset']}\n{condition}"


def draw(files, stem: str) -> None:
    index = {e["file"]: e for e in json.loads((RUN / "qualitative_index.json").read_text(encoding="utf-8"))}
    rows = [index[f] for f in files]
    for entry in rows:
        check_f1(entry)

    # layout in inches: row label | 6 tiles | decision column; title row on top, legend below
    width, label_w, tile_w, gap, decision_w = 7.2, 0.34, 0.93, 0.05, 0.86
    title_h, f1_h, row_gap, legend_h = 0.24, 0.17, 0.07, 0.27
    height = title_h + len(rows) * (tile_w + f1_h) + (len(rows) - 1) * row_gap + legend_h
    fig = plt.figure(figsize=(width, height))

    def axes_at(left, bottom, w, h):
        return fig.add_axes((left / width, bottom / height, w / width, h / height))

    x0 = label_w + 0.04
    decision_x = x0 + 6 * tile_w + 5 * gap + 0.12
    for r, entry in enumerate(rows):
        bottom = legend_h + (len(rows) - 1 - r) * (tile_w + f1_h + row_gap) + f1_h
        output = selected(entry)
        for c, (view, title) in enumerate(COLUMNS):
            left = x0 + c * (tile_w + gap)
            ax = axes_at(left, bottom, tile_w, tile_w)
            ax.imshow(tile(entry, view), interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            chosen = view == output
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.6 if chosen else 0.4)
                spine.set_edgecolor(GREEN if chosen else INK)
            if r == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if view in PREDICTIONS:
                ax.text(0.5, -0.04, f"F1 = {entry['rendered_f1'][view]:.2f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7, color=GREEN if chosen else INK,
                        fontweight="bold" if chosen else "normal")
        fig.text(0.17 / width, (bottom + tile_w / 2) / height, row_label(entry, chr(97+r)),
                 rotation=90, ha="center", va="center", fontsize=7.5, linespacing=1.15)

        # decision column: probability bar with the threshold (as in Figure 2b), then the decision
        p, tau = float(entry["selector_probability"]), float(entry["tau"])
        bar_h = 0.72
        ax = axes_at(decision_x, bottom + (tile_w - bar_h) / 2, decision_w, bar_h)
        ax.set_xlim(0, decision_w)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0.02, 0), 0.09, 1, facecolor="white", edgecolor=INK, linewidth=0.5))
        ax.add_patch(Rectangle((0.02, 0), 0.09, p, facecolor=BLUE, edgecolor="none"))
        ax.plot([0.0, 0.13], [tau, tau], color=RED, linewidth=1.1, solid_capstyle="butt")
        digits = 3 if abs(p - tau) < 0.005 else 2  # "0.498 < 0.50" rather than "0.50 < 0.50"
        relation = r"\geq" if output == "localizer_only" else "<"
        ax.text(0.2, 0.86, rf"$p\,=\,{p:.{digits}f}$", fontsize=7.5, va="center")
        ax.text(0.2, 0.62, rf"${relation}\,\tau\,=\,{tau:.2f}$", fontsize=7.5, va="center")
        name = r"$Q_{\mathrm{reg}}$" if output == "localizer_only" else r"$Q_{\mathrm{raw}}$"
        ax.text(0.2, 0.3, "output " + name, fontsize=7.5, va="center", color=GREEN, fontweight="bold")
        if r == 0:
            # same baseline as the column titles, which sit 3 pt above the tiles
            fig.text((decision_x + decision_w / 2) / width, (bottom + tile_w + 3 / 72) / height, "Selector",
                     ha="center", va="baseline", fontsize=8)

    handles = [Patch(facecolor=np.array(c) / 255, edgecolor=INK, linewidth=0.4, label=name)
               for name, c in ERROR_COLOURS]
    handles.append(Patch(facecolor="none", edgecolor=GREEN, linewidth=1.6, label=r"output $\hat{Y}$"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.3, handleheight=0.9, columnspacing=1.6)
    for suffix, options in ((".pdf", {"dpi": 300}), (".svg", {"dpi": 300}), (".png", {"dpi": 300})):
        fig.savefig(OUT / f"{stem}{suffix}", **options)
    plt.close(fig)


if __name__ == "__main__":
    setup()
    draw(QUALITATIVE, "figure_qualitative")
    print("figures/figure_qualitative.pdf written; archived samples and metrics preserved")
