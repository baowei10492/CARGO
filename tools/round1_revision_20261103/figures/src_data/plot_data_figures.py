"""Redraw the line-chart figures in the house style of Figures 1-2 (ieee-figure-table, python backend).

    python figures/src_data/plot_data_figures.py

figure_mixture.pdf        mean F1 of the 18 detector-dataset combinations as the share of misaligned
                          pairs goes from 0 to 100% (main text, mixed test).
figure_localization.pdf   registration success by changed area, and the share of patches whose
                          registration error exceeds x (main text, matcher baselines).
figure_tradeoff.pdf       level-3 gain (top) and level-0 F1 change (bottom) against the decision
                          threshold over the range searched by Eq. (8), one column per dataset (appendix).
figure_dose_response.pdf  median registration error against the area of pasted change on
                          same-date pairs (appendix).

Data are read, never edited:
  mixture   D:/work1/rs_cd_project/reports/p31/statistics/20260922T024916Z/mixture.json
            (the LATEST statistics run; identical to 20260922T024825Z)
  tradeoff  the same run: tradeoff.csv and tradeoff_operating.json
  success   D:/work1/rs_cd_project/paper/data/p31/overnight/table_success_bins.csv and the
            per-patch errors in reports/p31/overnight/run/errors/
  dose      D:/work1/rs_cd_project/paper/data/p31/overnight/table_dose.csv
Do not regenerate these figures with tools/p31_statistics.py or tools/p31_overnight_tables.py:
those scripts write the pre-renaming labels.

Style contract: serif text as in the manuscript (Palatino), white plotting area, CARGO and the
full registration module in the blue of Figures 1-2, every series paired with its own marker and
line style so the figure survives greyscale printing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RS = Path(r"D:\work1\rs_cd_project")
MIXTURE = RS / "reports/p31/statistics/20260922T024916Z/mixture.json"
BINS = Path(__file__).resolve().parent / "p33_success_bins.csv"
ERRORS = Path(__file__).resolve().parents[2] / "provenance/registration_errors"
STATISTICS = RS / "reports/p31/statistics/20260922T024916Z"
DOSE = RS / "paper/data/p31/overnight/table_dose.csv"
OUT = Path(__file__).resolve().parents[1]
CM = 1 / 2.54

INK, GRID, MUTED = "#172033", "#C9CED6", "#6B7280"
BLUE, BLUE_LIGHT, TEAL, RED, GREY = "#0B5CAD", "#6FA3D8", "#0F766E", "#C0392B", "#5F6673"
SHADE = "#DCE8F6"  # the step-box blue of Figures 1-2

# Line charts follow published examples (the advisor's SWIFT, Fig. 2; Compress-Align-Detect, Fig. 7):
# solid lines told apart by colour and marker shape, a marker on every data point, dashed light
# horizontal grid lines, open axes, an unframed legend, and a constant reference drawn as a dashed line.
STYLE = {  # label, colour, line style, marker, marker face
    "selector": ("CARGO", BLUE, "-", "o", BLUE),
    "estimated": ("Always registering", RED, "-", "s", RED),
    "raw": ("Never registering", GREY, "-", "^", GREY),
    "oracle": ("Oracle registration", INK, (0, (4, 2.5)), None, None),
    # Figure 4 compares registration modules; the selector does not change the estimated transform,
    # so CARGO enters this figure through its registration module.
    "v2_full": ("Registration module", BLUE, "-", "o", BLUE),
    "v2_no_reliability": ("No change-aware weights", BLUE_LIGHT, "--", "o", "white"),  # variant of ours
    "roma_ts_ransac": ("RoMa + RANSAC", TEAL, "-", "s", TEAL),
    "sift_ts_ransac": ("SIFT + RANSAC", GREY, "-", "^", GREY),
    "phasecorr_fm": ("Fourier–Mellin", RED, "-.", "D", RED),
    "ecc_ts": ("ECC", "#8B5E99", ":", "v", "#8B5E99"),
}


def hgrid(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
        "mathtext.fontset": "custom", "mathtext.rm": "Palatino Linotype", "mathtext.it": "Palatino Linotype:italic",
        "mathtext.bf": "Palatino Linotype:bold",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def line(ax, key, x, y, **extra):
    label, colour, style, marker, face = STYLE[key]
    kwargs = dict(color=colour, linestyle=style, linewidth=1.5 if key in ("selector", "v2_full") else 1.2,
                  label=label, zorder=4 if key in ("selector", "v2_full") else 3)
    kwargs.update(marker_style(key))
    kwargs.update(extra)
    return ax.plot(x, y, **kwargs)[0]


# Small markers (user request, 2026-10-07): roughly half the earlier area, sized per shape so that
# circles, squares and triangles look equally heavy (a square of the same size has 27% more area
# than a circle, a triangle about 36% less).
MARKER_SIZE = {"o": 3.0, "s": 2.6, "^": 3.3, "v": 3.3, "D": 2.6, "P": 3.2}

# Six detectors in the appendix trade-off figure: Okabe-Ito colours (colour-blind safe), each paired
# with its own marker shape at the calibrated operating point.
DETECTORS = (("dinov3", "DINOv3"), ("bit", "BIT"), ("cdlamba", "CD-Lamba"), ("cdxlstm", "CDxLSTM"),
             ("snunet", "SNUNet"), ("bifa", "BiFA"))
DETECTOR_STYLE = {"dinov3": ("#0072B2", "o"), "bit": ("#E69F00", "s"), "cdlamba": ("#009E73", "^"),
                  "cdxlstm": ("#D55E00", "D"), "snunet": ("#CC79A7", "v"), "bifa": ("#4D4D4D", "P")}
DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")


def marker_style(key: str) -> dict:
    label, colour, style, marker, face = STYLE[key]
    if not marker:
        return {}
    hollow = face == "white"
    if hollow:  # the coloured outline sits outside the path, so a hollow marker is drawn smaller
        return dict(marker=marker, markersize=0.8 * MARKER_SIZE[marker], markerfacecolor=face,
                    markeredgecolor=colour, markeredgewidth=0.6)
    return dict(marker=marker, markersize=MARKER_SIZE[marker], markerfacecolor=face,
                markeredgecolor="white", markeredgewidth=0.35)


def save(fig, stem: str, folder: Path = OUT) -> None:
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(folder / f"{stem}{suffix}", **options)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ mixture
def mixture_series() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = json.loads(MIXTURE.read_text(encoding="utf-8"))["mean_over_cells"]
    p = np.array([100 * r["p"] for r in rows])
    series = {k: np.array([100 * r[k] for r in rows]) for k in ("raw", "estimated", "selector", "oracle")}
    assert abs(series["selector"][0] - 76.9) < 0.05 and abs(series["selector"][-1] - 66.6) < 0.05
    return p, series


def crossing(p: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Where second - first changes sign (linear interpolation between grid points)."""
    diff = second - first
    k = int(np.argmax(diff > 0))
    return float(p[k - 1] + (p[k] - p[k - 1]) * (-diff[k - 1]) / (diff[k] - diff[k - 1]))


def mixture_gain_figure() -> None:
    """Archived alternative (src_data/figure_mixture_gain.pdf): gain of CARGO over each fixed strategy."""
    p, series = mixture_series()
    over_never = series["selector"] - series["raw"]
    over_always = series["selector"] - series["estimated"]
    over_best = np.minimum(over_never, over_always)
    peak = int(np.argmax(over_best))
    cross = crossing(p, series["raw"], series["estimated"])
    assert abs(over_best[peak] - 2.44) < 0.01 and p[peak] == 35

    fig, ax = plt.subplots(figsize=(10.4 * CM, 6.4 * CM), constrained_layout=True)
    ax.axhline(0, color=INK, linewidth=0.7, zorder=2)
    ax.fill_between(p, 0, over_best, where=over_best >= 0, interpolate=True, color=SHADE, linewidth=0, zorder=1)
    line(ax, "raw", p, over_never, markevery=2, label="over Never registering", linewidth=1.1)
    line(ax, "estimated", p, over_always, markevery=2, label="over Always registering", linewidth=1.1)
    ax.plot(p, over_best, color=BLUE, linewidth=1.9, label="over the better of the two", zorder=4)
    ax.plot([cross, cross], [0, np.interp(cross, p, over_never)], color=MUTED, linewidth=0.6,
            linestyle=(0, (2, 2)), zorder=2)
    ax.text(cross + 1.2, 0.18, f"never = always\nat {cross:.0f}%", fontsize=7, color=MUTED, va="bottom")
    ax.plot(p[peak], over_best[peak], marker="o", markersize=4.6, color=BLUE, zorder=5)
    # label in the free wedge between the rising and the falling curve
    ax.annotate(f"+{over_best[peak]:.1f} pp at {p[peak]:.0f}%", xy=(p[peak], over_best[peak]),
                xytext=(58, 2.95), fontsize=7.5, color=BLUE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6, shrinkA=1, shrinkB=3))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 7.4)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 8, 2))
    hgrid(ax)
    ax.set_xlabel("Misaligned pairs (%)")
    ax.set_ylabel("F1 gain of CARGO (pp)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ("over the better of the two", "over Never registering",
                                       "over Always registering")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", handlelength=2.6)
    save(fig, "figure_mixture_gain", Path(__file__).resolve().parent)


def mixture_figure() -> None:
    """Main-text version: absolute mean F1 of the four strategies, drawn like a plain published line chart."""
    p, series = mixture_series()

    fig, ax = plt.subplots(figsize=(10.4 * CM, 6.8 * CM), constrained_layout=True)
    line(ax, "oracle", p, series["oracle"], linewidth=1.1)
    for key in ("raw", "estimated", "selector"):
        line(ax, key, p, series[key])
    ax.set_xlim(-2, 102)
    ax.set_ylim(58, 79)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(60, 80, 5))
    hgrid(ax)
    ax.set_xlabel("Misaligned pairs (%)")
    ax.set_ylabel("Mean F1 of 18 combinations (%)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(STYLE[k][0]) for k in ("selector", "estimated", "raw", "oracle")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside upper center", ncol=2,
               handlelength=2.4, columnspacing=1.6)
    save(fig, "figure_mixture")


# ------------------------------------------------------------------------------------- localization
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def crop_errors(variant: str, level: int) -> np.ndarray:
    """Per-patch registration error of Eq. (10): min(direct, composed with the dataset residual)."""
    out = []
    for row in read_csv(ERRORS / f"errors_{variant}.csv"):
        if int(row["level"]) != level:
            continue
        direct = float(row["ate_raw"])
        composed = float(row["ate_diff"]) if row["ate_diff"] not in ("", "None") else None
        out.append(direct if composed is None else min(direct, composed))
    return np.array(out)


def tail_panel(ax, level: int, variants, marks: dict[str, float]) -> None:
    """Share of patches whose error exceeds x, on a log x axis."""
    grid = np.logspace(0, np.log10(1500), 400)
    anchors = np.array([1, 2, 5, 10, 25, 50, 100, 250, 500, 1000])
    for v in variants:
        e = crop_errors(v, level)
        assert len(e) == 4792
        tail = np.array([100 * (e > x).mean() for x in grid])
        label, colour, style, marker, face = STYLE[v]
        ax.plot(grid, tail, color=colour, linestyle=style, linewidth=1.5 if v == "v2_full" else 1.2,
                zorder=4 if v == "v2_full" else 3)
        ax.plot(anchors, [100 * (e > x).mean() for x in anchors], linestyle="none", **marker_style(v),
                zorder=4 if v == "v2_full" else 3)
        if v in marks:
            got = 100 * (e > 25).mean()
            assert abs(got - marks[v]) < 0.05, (v, level, got)
    for x in (2, 25):  # the success threshold and the large-error threshold of Section 4.6.2
        ax.axvline(x, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    roma, full = marks["roma_ts_ransac"], marks["v2_full"]
    ax.annotate(f"{roma:.1f}", xy=(25, roma), xytext=(42, roma + 8), fontsize=7.5, color=TEAL, va="center",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.6, shrinkA=0, shrinkB=1.5))
    ax.annotate(f"{full:.1f}", xy=(25, full), xytext=(48, full + 7), fontsize=7.5, color=BLUE, va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6, shrinkA=0, shrinkB=1.5))
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.set_xticks([1, 2, 10, 25, 100, 1000])
    ax.set_xticklabels(["1", "2", "10", "25", "100", "1000"])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Error threshold $x$ (px)")
    ax.set_ylabel("Crops with error $> x$ (%)")  # "crop" as in the English text (Section 4.1.1)


def localization_figure() -> None:
    bins = read_csv(BINS)
    order = list(dict.fromkeys(r["change_fraction_bin"] for r in bins))
    names = {"0": "0", "0.00-0.02": "0–2", "0.02-0.05": "2–5", "0.05-0.10": "5–10", "0.10-0.20": "10–20",
             "0.20-1.01": ">20"}
    counts = {b: next(r["crops"] for r in bins if r["change_fraction_bin"] == b) for b in order}
    variants = ("v2_full", "v2_no_reliability", "roma_ts_ransac", "sift_ts_ransac", "phasecorr_fm", "ecc_ts")
    success = {v: np.array([100 * float(next(r["success"] for r in bins if r["variant"] == v
                                              and r["change_fraction_bin"] == b)) for b in order]) for v in variants}
    assert round(success["roma_ts_ransac"][-1]) == 19 and round(success["v2_full"][-1]) == 14

    fig, axes = plt.subplots(1, 3, figsize=(18.3 * CM, 7.2 * CM), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1, 1]})
    x = np.arange(len(order))
    for v in variants:
        line(axes[0], v, x, success[v])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{names[b]}\n(n={counts[b]})" for b in order], fontsize=6.5)
    axes[0].tick_params(axis="x", pad=3)
    axes[0].set_xlabel("Changed pixels in the crop (%)", labelpad=8)
    for v, i, dx, dy, ha in (("v2_full", 0, 0.0, -4.0, "center"), ("roma_ts_ransac", 5, 0.12, 2.2, "left"),
                             ("v2_full", 5, 0.12, -3.6, "left"), ("v2_no_reliability", 0, 0.12, -4.0, "left")):
        axes[0].text(i + dx, success[v][i] + dy, f"{success[v][i]:.0f}", fontsize=7, color=STYLE[v][1], ha=ha,
                     va="center")

    axes[0].set_ylim(0, 72)
    axes[0].set_yticks(range(0, 71, 10))
    axes[0].set_ylabel("Recovery rate (%)")
    axes[0].set_title("Level 3, by changed area", fontsize=8, pad=4)

    # (b) level 3 and (c) unperturbed: how often the error exceeds x; numbers checked against Section 4.6.2
    tail_panel(axes[1], 3, variants, {"v2_full": 6.5, "roma_ts_ransac": 25.4})
    axes[1].set_title("Level 3", fontsize=8, pad=4)
    tail_panel(axes[2], 0, variants, {"v2_full": 1.9, "roma_ts_ransac": 26.4})
    axes[2].set_title("Unperturbed pairs", fontsize=8, pad=4)

    for ax, tag in zip(axes, "abc"):
        hgrid(ax)
        ax.text(-0.16 if tag == "a" else -0.2, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold",
                fontsize=8.5, va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, handlelength=2.6, columnspacing=1.8)
    save(fig, "figure_localization")


def panel_tag(ax, tag: str, x: float = -0.2) -> None:
    ax.text(x, 1.02, f"({tag})", transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom")


# ----------------------------------------------------------------------------------------- tradeoff
def tradeoff_curves_figure() -> None:
    """Archived alternative (src_data/figure_tradeoff_curves.pdf): level-0 change against level-3 gain with
    the threshold hidden along each curve. Replaced on 2026-10-07: the curves fold back and the part
    below tau = 0.5, which Eq. (8) never searches, takes most of the drawing."""
    rows = read_csv(STATISTICS / "tradeoff.csv")
    operating = json.loads((STATISTICS / "tradeoff_operating.json").read_text(encoding="utf-8"))
    # the operating points are the level-0 change and level-3 gain of Table 2 (DINOv3 on LEVIR-CD: -0.3, +15.3)
    assert round(100 * operating["dinov3/LEVIR-CD"]["clean_change"], 1) == -0.3
    assert round(100 * operating["dinov3/LEVIR-CD"]["gain_3"], 1) == 15.3

    fig, axes = plt.subplots(1, 3, figsize=(18.3 * CM, 6.2 * CM), constrained_layout=True)
    for ax, dataset, tag in zip(axes, DATASETS, "abc"):
        for row, label in DETECTORS:
            cell = f"{row}/{dataset}"
            sweep = [r for r in rows if r["cell"] == cell]  # thresholds 0.00 ... 1.00, then never adopting
            assert len(sweep) == 102, (cell, len(sweep))
            colour, marker = DETECTOR_STYLE[row]
            ax.plot([100 * float(r["clean_change"]) for r in sweep], [100 * float(r["gain_3"]) for r in sweep],
                    color=colour, linewidth=1.1, zorder=3)
            point = operating[cell]
            ax.plot(100 * point["clean_change"], 100 * point["gain_3"], linestyle="none", marker=marker,
                    markersize=1.25 * MARKER_SIZE[marker], markerfacecolor=colour, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=4)
            ax.plot([], [], color=colour, linewidth=1.1, marker=marker, markersize=1.25 * MARKER_SIZE[marker],
                    markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.4, label=label)
        ax.axvline(-0.5, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        hgrid(ax)
        ax.set_title(dataset, fontsize=8, pad=4)
        ax.set_xlabel("F1 change, unperturbed pairs\n(percentage points)")
        panel_tag(ax, tag, -0.17 if tag == "a" else -0.14)
    axes[0].set_ylabel("F1 gain at level 3\n(percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6, handlelength=2.2, columnspacing=1.6)
    save(fig, "figure_tradeoff_curves", Path(__file__).resolve().parent)


def tradeoff_figure() -> None:
    """Appendix figure (figures/figure_tradeoff.pdf): the threshold sweep with tau on the x axis.

    Only the range searched by Eq. (8), tau = 0.50 ... 1.00, is drawn. Top row: level-3 gain;
    bottom row: level-0 change; markers at the calibrated threshold of each combination.
    """
    rows = read_csv(STATISTICS / "tradeoff.csv")
    operating = json.loads((STATISTICS / "tradeoff_operating.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(18.3 * CM, 10.2 * CM), constrained_layout=True, sharex=True)
    for col, dataset in enumerate(DATASETS):
        for row, label in DETECTORS:
            cell = f"{row}/{dataset}"
            sweep = [r for r in rows if r["cell"] == cell and r["tau"] not in ("", "None")
                     and float(r["tau"]) >= 0.5 - 1e-9]
            assert len(sweep) == 51, (cell, len(sweep))
            tau = [float(r["tau"]) for r in sweep]
            colour, marker = DETECTOR_STYLE[row]
            point = operating[cell]
            for r_ax, key in ((0, "gain_3"), (1, "clean_change")):
                ax = axes[r_ax, col]
                ax.plot(tau, [100 * float(r[key]) for r in sweep], color=colour, linewidth=1.1, zorder=3,
                        drawstyle="steps-post", label=label if (r_ax, col) == (0, 0) else None)
                ax.plot(point["tau"], 100 * point[key], linestyle="none", marker=marker,
                        markersize=1.25 * MARKER_SIZE[marker], markerfacecolor=colour, markeredgecolor="white",
                        markeredgewidth=0.4, zorder=4)
        axes[0, col].set_title(dataset, fontsize=8, pad=4)
        axes[1, col].axhline(0, color=INK, linewidth=0.6, zorder=2)
        axes[1, col].axhline(-0.5, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=2)
        axes[1, col].set_xlabel(r"Decision threshold $\tau$")
        for r_ax in (0, 1):
            hgrid(axes[r_ax, col])
            axes[r_ax, col].set_xlim(0.49, 1.01)
            axes[r_ax, col].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        panel_tag(axes[0, col], "abc"[col], -0.17 if col == 0 else -0.14)
        panel_tag(axes[1, col], "def"[col], -0.17 if col == 0 else -0.14)
    axes[0, 0].set_ylabel("F1 gain at level 3\n(percentage points)")
    axes[1, 0].set_ylabel("F1 change, unperturbed pairs\n(percentage points)")
    handles = [plt.Line2D([], [], color=DETECTOR_STYLE[r][0], linewidth=1.1, marker=DETECTOR_STYLE[r][1],
                          markersize=1.25 * MARKER_SIZE[DETECTOR_STYLE[r][1]], markerfacecolor=DETECTOR_STYLE[r][0],
                          markeredgecolor="white", markeredgewidth=0.4) for r, _ in DETECTORS]
    fig.legend(handles, [label for _, label in DETECTORS], loc="outside upper center", ncol=6, handlelength=2.2,
               columnspacing=1.6)
    save(fig, "figure_tradeoff")


# --------------------------------------------------------------------------------------------- dose
def dose_figure() -> None:
    rows = read_csv(DOSE)
    variants = ("v2_full", "v2_no_reliability", "roma_ts_ransac")
    fig, axes = plt.subplots(1, 2, figsize=(18.3 * CM, 5.9 * CM), constrained_layout=True, sharey=True)
    for ax, level, title, tag in zip(axes, (3, 0), ("Level 3", "Unperturbed pairs"), "ab"):
        for v in variants:
            chosen = sorted((r for r in rows if r["variant"] == v and int(r["level"]) == level),
                            key=lambda r: float(r["fraction"]))
            assert len(chosen) == 6
            line(ax, v, [100 * float(r["fraction"]) for r in chosen], [float(r["median_error"]) for r in chosen])
        # solid, so the threshold is not mistaken for one of the dashed grid lines
        ax.axhline(2.0, color=MUTED, linewidth=0.7, zorder=1)
        ax.text(50.5, 2.3, "2 px", fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 5)
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
        ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
        ax.minorticks_off()
        ax.set_xlim(-2, 52)
        ax.set_xticks(range(0, 51, 10))
        hgrid(ax)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xlabel("Pasted change (% of the crop)")
        panel_tag(ax, tag, -0.13 if tag == "a" else -0.06)
    axes[0].set_ylabel("Median registration error (px)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, handlelength=2.6, columnspacing=1.8)
    save(fig, "figure_dose_response")


if __name__ == "__main__":
    setup()
    mixture_figure()
    mixture_gain_figure()
    localization_figure()
    tradeoff_figure()
    tradeoff_curves_figure()
    dose_figure()
    print("figures/figure_{mixture,localization,tradeoff,dose_response}.pdf and the archived alternatives "
          "src_data/figure_mixture_gain.pdf, src_data/figure_tradeoff_curves.pdf written")
