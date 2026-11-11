"""Plot only the original, sparsely saved selector losses; no refitting."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8")
    original = json.loads(args.summary.read_text(encoding="utf-8"))
    pattern = re.compile(r"classifier=selector_(\S+) epoch=(\d+) fit_loss=([\d.eE+-]+) early_stop_loss=([\d.eE+-]+) best_epoch=(\d+)")
    records = [{"dataset": n, "epoch": int(e), "fit_loss": float(f), "early_stop_loss": float(v),
                "running_best_epoch": int(b)} for n, e, f, v, b in pattern.findall(text)]
    stops = {n: {"stopped_epoch": int(e), "best_epoch": int(b)} for n, e, b in
             re.findall(r"classifier=selector_(\S+) early stop at epoch=(\d+) \(best_epoch=(\d+)\)", text)}
    names = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
    for name in names:
        if not any(r["dataset"] == name for r in records) or name not in stops:
            raise ValueError(f"missing original log data: {name}")
        if stops[name]["best_epoch"] != original["datasets"][name]["best_epoch"]:
            raise ValueError(f"log/bundle-summary disagreement: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "original_loss_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75), layout="constrained")
    for index, (ax, name) in enumerate(zip(axes, names)):
        rows = [r for r in records if r["dataset"] == name]
        x = [r["epoch"] for r in rows]
        ax.plot(x, [r["fit_loss"] for r in rows], color="#0072B2", marker="o", markersize=3.5,
                linewidth=1.25, label="Training")
        ax.plot(x, [r["early_stop_loss"] for r in rows], color="#D55E00", marker="s", markersize=3.2,
                linewidth=1.25, linestyle="--", label="Validation (early stop)")
        best, stopped = stops[name]["best_epoch"], stops[name]["stopped_epoch"]
        ax.axvline(best, color="#555555", linewidth=0.9, linestyle=":")
        ax.set_title(f"({chr(97+index)}) {name}")
        ax.set_xlabel("Epoch")
        if index == 0:
            ax.set_ylabel("Weighted binary cross-entropy")
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(0.25, 0.57)
        ax.grid(axis="y", alpha=0.17, linewidth=0.6)
        ax.text(0.97, 0.96, f"Selected: {best}\nStopped: {stopped}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"selector_convergence_original.{extension}", dpi=300)
    plt.close(fig)
    gray = plt.imread(args.output / "selector_convergence_original.png")
    if gray.ndim == 3:
        grayscale = gray[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        plt.imsave(args.output / "selector_convergence_grayscale_QA.png", grayscale, cmap="gray", vmin=0, vmax=1)
    caption = (
        "Original training and validation loss records for the DINOv3 selectors on the three main datasets. "
        "Markers show the values actually saved at epoch 1 and every 25 epochs; connecting lines are guides. "
        "The validation split is the tile-disjoint early-stopping split, distinct from threshold calibration and testing. "
        "Dotted vertical lines mark the selected epochs (51, 45, and 19); training stopped at epochs 111, 105, and 79, "
        "respectively, after 60 epochs without validation improvement. The maximum budget was 300 epochs. "
        "Training losses were recorded during optimization with dropout; validation losses were evaluated with dropout disabled. "
        "No additional training was performed to produce this figure."
    )
    (args.output / "caption_en.txt").write_text(caption + "\n", encoding="utf-8")
    evidence = {"source": "original training log, not a diagnostic rerun", "log": str(args.log.resolve()),
                "log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
                "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                "stopping": stops, "loss_samples": len(records), "caption": caption}
    (args.output / "provenance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(records), "epochs": stops}, ensure_ascii=False))


if __name__ == "__main__":
    main()
