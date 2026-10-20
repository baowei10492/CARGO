"""P31 statistics from the existing border-free test records (CPU only; no model is run on images).

    ci             tile bootstrap (tiles resampled with replacement, all crops and levels of a tile together):
                   95% intervals of the clean change and the level gains of the localizer + selector in every
                   cell, and of ours minus the border-free augmentation baseline (DINOv3 row, same crops)
    decomposition  per cell and level: without correction, selector, always corrected, per-crop best of the two
                   predictions (fewer wrong pixels), oracle correction (true transform)
    transfer       the frozen selectors applied unchanged to other datasets (same detector) and to other
                   detectors (same dataset), each with its own calibrated tau
    tradeoff       clean change and gains over tau, with the calibrated operating point
    mixture        F1 when a fraction p of the pairs is misaligned (levels 1-3 equally often)

Before any number is written the records must reproduce the P30 / P31 test tables (raw and selector F1 per level,
selector probabilities). Output: reports/p31/statistics/<tag>/ (json, csv, figures).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rs_cd.p27_selector import FEATURE_NAMES, selector_probability
from tools import run_p30
from tools.p31_selector_ablation import BORDER_FREE_ROWS, DATASETS, row_root, sources

PATHS = ("raw", "oracle", "estimated")
KEYS = ("tp", "fp", "fn")
LEVELS = (0, 1, 2, 3)
EPS = 1e-12                  # rs_cd.metrics.binary.BinaryChangeMetrics
UNSAFE = -0.005              # clean change below this is unsafe (the 0.5 point line)
ANCHOR_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-4
AUGMENTATION_TOLERANCE = 1e-3
FINAL = "reports/p31/final"
# P32 added two detectors under the same protocol (docs/P32_PLUGIN_EXTENSION.md);
# their records live under reports/p32 and are read through row_root.
P32_ROWS = (("snunet", "SNUNet", DATASETS), ("bifa", "BiFA", DATASETS))
ALL_ROWS = BORDER_FREE_ROWS + P32_ROWS
TAUS = [round(0.01 * k, 2) for k in range(101)] + [math.inf]
MIXTURE = [round(0.05 * k, 2) for k in range(21)]


def f1(counts: np.ndarray) -> np.ndarray:
    """F1 of (..., 3) tp/fp/fn counts, with BinaryChangeMetrics' formula."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2.0 * precision * recall / (precision + recall + EPS)


class Cell:
    """The test records of one detector (row) on one dataset."""

    def __init__(self, row: str, label: str, dataset: str, records: Path, bundle: dict[str, Any]) -> None:
        with records.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["dataset"] == dataset]
        if not rows:
            raise SystemExit(f"{records}: no {dataset} records")
        self.row, self.label, self.dataset, self.bundle = row, label, dataset, bundle
        self.tau = math.inf if bundle["tau"] is None else float(bundle["tau"])
        self.crop = np.array([r["crop_id"] for r in rows])
        self.level = np.array([int(r["level"]) for r in rows])
        names = sorted({r["tile_id"] for r in rows})
        index = {name: k for k, name in enumerate(names)}
        self.tile = np.array([index[r["tile_id"]] for r in rows])
        self.tiles = len(names)
        self.counts = {p: np.array([[float(r[f"{p}_{k}"]) for k in KEYS] for r in rows]) for p in PATHS}
        self.probability = np.array([float(r["selector_probability"]) for r in rows])
        self.features = {name: np.array([float(r[name]) for r in rows]) for name in FEATURE_NAMES}

    @property
    def name(self) -> str:
        return f"{self.row}/{self.dataset}"

    def selected(self, probability: np.ndarray | None = None, tau: float | None = None) -> np.ndarray:
        probability = self.probability if probability is None else probability
        tau = self.tau if tau is None else tau
        return np.where((probability >= tau)[:, None], self.counts["estimated"], self.counts["raw"])

    def best(self) -> np.ndarray:
        """Per crop, the prediction with fewer wrong pixels (ties: without correction)."""
        wrong = {p: self.counts[p][:, 1] + self.counts[p][:, 2] for p in ("raw", "estimated")}
        return np.where((wrong["estimated"] < wrong["raw"])[:, None], self.counts["estimated"], self.counts["raw"])

    def by_level(self, counts: np.ndarray) -> dict[int, float]:
        return {level: float(f1(counts[self.level == level].sum(axis=0))) for level in LEVELS}

    def tile_sums(self, counts: np.ndarray) -> np.ndarray:
        """(levels, tiles, 3) count sums."""
        out = np.zeros((len(LEVELS), self.tiles, 3))
        for i, level in enumerate(LEVELS):
            keep = self.level == level
            np.add.at(out[i], self.tile[keep], counts[keep])
        return out

    def recomputed_probability(self, bundle: dict[str, Any] | None = None) -> np.ndarray:
        bundle = self.bundle if bundle is None else bundle
        matrix = np.stack([self.features[name] for name in bundle["feature_names"]], axis=1)
        return selector_probability(bundle, torch.tensor(matrix, dtype=torch.float32)).numpy().astype(float)


def summary_of(values: dict[int, float], reference: dict[int, float]) -> dict[str, float]:
    return {"clean_change": values[0] - reference[0], **{f"gain_{k}": values[k] - reference[k] for k in LEVELS[1:]}}


# --------------------------------------------------------------------------------------------------- loading
def anchor_table(row: str) -> dict[str, dict[str, dict[str, float]]]:
    """{dataset: {level: {rule: F1}}} of the frozen test run of a row."""
    if row == "dinov3":
        data = json.loads((run_p30.pointed("reports/p30/test/LATEST") / "p30_test.json").read_text(encoding="utf-8"))
        return {name: entry["proposed"]["f1"] for name, entry in data["datasets"].items()}
    data = json.loads((row_root(row) / "test" / "p31_test.json").read_text(encoding="utf-8"))
    return {name: entry["f1"] for name, entry in data["datasets"].items()}


def load_cells(rows: Sequence[tuple[str, str, Sequence[str]]] = BORDER_FREE_ROWS) -> list[Cell]:
    cells = []
    for row, label, names in rows:
        train_root, records = sources(row)
        anchors = anchor_table(row)
        for name, path in zip(DATASETS, records):
            if name not in names:
                continue
            bundle = torch.load(train_root / "bundles" / f"selector_{name}.pth", map_location="cpu", weights_only=False)
            cell = Cell(row, label, name, path, bundle)
            check_cell(cell, anchors[name])
            cells.append(cell)
    return cells


def check_cell(cell: Cell, anchor: dict[str, dict[str, float]]) -> None:
    """The records must reproduce the frozen test table; otherwise nothing is computed."""
    got = {"raw": cell.by_level(cell.counts["raw"]), "selector": cell.by_level(cell.selected()),
           "estimated": cell.by_level(cell.counts["estimated"]), "oracle": cell.by_level(cell.counts["oracle"])}
    for rule, values in got.items():
        for level, value in values.items():
            if abs(value - float(anchor[str(level)][rule])) > ANCHOR_TOLERANCE:
                raise SystemExit(f"{cell.name} level {level} {rule}: records give {value:.6f}, "
                                 f"the test table {anchor[str(level)][rule]:.6f}; instrument error, stop")
    difference = float(np.abs(cell.recomputed_probability() - cell.probability).max())
    if difference > PROBABILITY_TOLERANCE:
        raise SystemExit(f"{cell.name}: the frozen selector does not reproduce the recorded probabilities "
                         f"(max difference {difference:.2e})")


def augmentation_counts(cell: Cell, group: str = "p30_augmented") -> np.ndarray:
    """Per-crop counts of an augmentation baseline on the cell's crops and levels (border-free views)."""
    root = sorted(p for p in Path(FINAL).iterdir() if p.is_dir())[-1]      # as tools.p31_paper_tables
    path = root / "border_counts" / f"counts_{cell.dataset}.csv"
    table = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle):
            if r["group"] == group and r["view"] == ("clean" if r["level"] == "0" else "border_free"):
                table[(r["crop_id"], int(r["level"]))] = [float(r[k]) for k in KEYS]
    keys = list(zip(cell.crop.tolist(), cell.level.tolist()))
    missing = [key for key in keys if key not in table]
    if missing or len(table) != len(keys):
        raise SystemExit(f"{path}: the {group} crops differ from the {cell.name} records ({len(missing)} missing)")
    return np.array([table[key] for key in keys])


def check_augmentation(cell: Cell, counts: np.ndarray, group: str = "p30_augmented") -> dict[str, float]:
    data = json.loads((run_p30.pointed("reports/p30/test/LATEST") / "p30_test.json").read_text(encoding="utf-8"))
    reference = data["datasets"][cell.dataset][group]
    values = cell.by_level(counts)
    largest = max(abs(values[level] - float(reference[str(level)])) for level in LEVELS)
    if largest > AUGMENTATION_TOLERANCE:
        raise SystemExit(f"{cell.name}: the {group} counts do not reproduce the P30 test row ({largest:.4f})")
    return {str(level): value for level, value in values.items()}


# ------------------------------------------------------------------------------------------------------ ci
def draws(tiles: int, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, tiles, tiles), minlength=tiles) for _ in range(repeats)]).astype(float)


def interval(samples: np.ndarray, point: float) -> dict[str, Any]:
    low, high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))
    return {"value": point, "low": low, "high": high, "excludes_zero": bool(low > 0 or high < 0)}


def resampled(weights: np.ndarray, sums: np.ndarray) -> np.ndarray:
    """(repeats, 3) counts; torch's matmul, as numpy's MKL next to torch aborts on Windows (duplicate OpenMP)."""
    return (torch.from_numpy(weights) @ torch.from_numpy(sums)).numpy()


def paired(cell: Cell, first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> dict[str, dict[str, Any]]:
    """F1(first) - F1(second) per level, point value and tile-bootstrap interval."""
    a, b = cell.tile_sums(first), cell.tile_sums(second)
    out = {}
    for i, level in enumerate(LEVELS):
        point = float(f1(a[i].sum(axis=0)) - f1(b[i].sum(axis=0)))
        out[str(level)] = interval(f1(resampled(weights, a[i])) - f1(resampled(weights, b[i])), point)
    return out


def run_ci(cells: Sequence[Cell], repeats: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {"repeats": repeats, "seed": seed, "unit": "tile", "cells": {}, "versus_augmentation": {}}
    for cell in cells:
        weights = draws(cell.tiles, repeats, seed + zlib.crc32(cell.name.encode()) % 100000)
        result["cells"][cell.name] = {"tiles": cell.tiles, "crops": int((cell.level == 0).sum()),
                                      "selector_minus_raw": paired(cell, cell.selected(), cell.counts["raw"], weights)}
        if cell.row == "dinov3":
            augmented = augmentation_counts(cell)
            result["versus_augmentation"][cell.dataset] = {
                "augmentation_f1": check_augmentation(cell, augmented),
                "ours_minus_augmentation": paired(cell, cell.selected(), augmented, weights)}
    entries = [entry["selector_minus_raw"] for entry in result["cells"].values()]
    result["counts"] = {
        "level_3_gain_excludes_zero": sum(e["3"]["excludes_zero"] and e["3"]["value"] > 0 for e in entries),
        "clean_change_significantly_negative": sum(e["0"]["high"] < 0 for e in entries),
        "clean_change_significantly_positive": sum(e["0"]["low"] > 0 for e in entries),
        "cells": len(entries)}
    return result


# -------------------------------------------------------------------------------------------- decomposition
def run_decomposition(cells: Sequence[Cell]) -> dict[str, Any]:
    out: dict[str, Any] = {"cells": {}}
    for cell in cells:
        paths = {"raw": cell.counts["raw"], "selector": cell.selected(), "estimated": cell.counts["estimated"],
                 "best_of_two": cell.best(), "oracle": cell.counts["oracle"]}
        out["cells"][cell.name] = {name: {str(k): v for k, v in cell.by_level(c).items()} for name, c in paths.items()}
    mean: dict[str, dict[str, float]] = {}
    for level in ("1", "2", "3"):
        values = {name: float(np.mean([entry[name][level] for entry in out["cells"].values()]))
                  for name in ("raw", "selector", "estimated", "best_of_two", "oracle")}
        recoverable = values["oracle"] - values["raw"]
        mean[level] = {**values, "recoverable": recoverable, "realized": values["selector"] - values["raw"],
                       "lost_to_localizer": values["oracle"] - values["best_of_two"],
                       "lost_to_selector": values["best_of_two"] - values["selector"],
                       "realized_share": (values["selector"] - values["raw"]) / recoverable if recoverable else None}
    out["mean_over_cells"] = mean
    return out


# ------------------------------------------------------------------------------------------------- transfer
def transfer_entry(kind: str, source: Cell, target: Cell) -> dict[str, Any]:
    probability = target.probability if source is target else target.recomputed_probability(source.bundle)
    values = target.by_level(target.selected(probability, source.tau))
    raw = target.by_level(target.counts["raw"])
    return {"kind": kind, "source": source.name, "target": target.name, "tau": source.tau,
            "selected_clean": float((probability[target.level == 0] >= source.tau).mean()),
            **summary_of(values, raw)}


def run_transfer(cells: Sequence[Cell]) -> dict[str, Any]:
    by_name = {cell.name: cell for cell in cells}
    entries = [transfer_entry("native", cell, cell) for cell in cells]
    for cell in cells:
        for other in cells:
            if other is cell:
                continue
            if other.row == cell.row:
                entries.append(transfer_entry("other_dataset", cell, other))
            elif other.dataset == cell.dataset:
                entries.append(transfer_entry("other_detector", cell, other))
    summary = {}
    for kind in ("native", "other_dataset", "other_detector"):
        chosen = [e for e in entries if e["kind"] == kind]
        summary[kind] = {"pairs": len(chosen),
                         "clean_change_mean": float(np.mean([e["clean_change"] for e in chosen])),
                         "clean_change_worst": float(min(e["clean_change"] for e in chosen)),
                         "unsafe": sum(e["clean_change"] < UNSAFE for e in chosen),
                         **{f"gain_{k}_mean": float(np.mean([e[f"gain_{k}"] for e in chosen])) for k in LEVELS[1:]}}
    # Per target cell: the native selector against the mean of the selectors borrowed from elsewhere.
    per_target = {}
    for name in by_name:
        native = next(e for e in entries if e["kind"] == "native" and e["target"] == name)
        row = {"native_clean_change": native["clean_change"], "native_gain_3": native["gain_3"]}
        for kind in ("other_dataset", "other_detector"):
            borrowed = [e for e in entries if e["kind"] == kind and e["target"] == name]
            row[f"{kind}_clean_change_mean"] = float(np.mean([e["clean_change"] for e in borrowed]))
            row[f"{kind}_gain_3_mean"] = float(np.mean([e["gain_3"] for e in borrowed]))
        per_target[name] = row
    return {"entries": entries, "summary": summary, "per_target": per_target}


# ------------------------------------------------------------------------------------------------- tradeoff
def run_tradeoff(cells: Sequence[Cell]) -> dict[str, Any]:
    rows = []
    for cell in cells:
        raw = cell.by_level(cell.counts["raw"])
        for tau in TAUS:
            values = cell.by_level(cell.selected(tau=tau))
            rows.append({"cell": cell.name, "tau": None if math.isinf(tau) else tau, **summary_of(values, raw)})
    operating = {cell.name: {"tau": cell.tau, **summary_of(cell.by_level(cell.selected()), cell.by_level(cell.counts["raw"]))}
                 for cell in cells}
    mean = []
    for tau in TAUS:
        chosen = [r for r in rows if r["tau"] == (None if math.isinf(tau) else tau)]
        mean.append({"tau": chosen[0]["tau"], **{key: float(np.mean([r[key] for r in chosen]))
                                                 for key in ("clean_change", "gain_1", "gain_2", "gain_3")}})
    return {"rows": rows, "operating": operating, "mean_over_cells": mean}


# -------------------------------------------------------------------------------------------------- mixture
def mixture_f1(cell: Cell, counts: np.ndarray, p: float) -> float:
    total = (1.0 - p) * counts[cell.level == 0].sum(axis=0)
    for level in LEVELS[1:]:
        total = total + (p / 3.0) * counts[cell.level == level].sum(axis=0)
    return float(f1(total))


def run_mixture(cells: Sequence[Cell]) -> dict[str, Any]:
    methods = ("raw", "estimated", "selector", "oracle")
    rows = []
    for cell in cells:
        paths = {"raw": cell.counts["raw"], "estimated": cell.counts["estimated"], "selector": cell.selected(),
                 "oracle": cell.counts["oracle"]}
        for p in MIXTURE:
            rows.append({"cell": cell.name, "p": p, **{m: mixture_f1(cell, paths[m], p) for m in methods}})
    mean = [{"p": p, **{m: float(np.mean([r[m] for r in rows if r["p"] == p])) for m in methods}} for p in MIXTURE]
    per_cell = {}
    for cell in cells:
        chosen = [r for r in rows if r["cell"] == cell.name]
        crossing = next((r["p"] for r in chosen if r["estimated"] >= r["raw"]), None)
        margin = min(r["selector"] - max(r["raw"], r["estimated"]) for r in chosen)
        per_cell[cell.name] = {"always_correct_breaks_even_at": crossing, "selector_minus_best_fixed_worst": margin}
    return {"rows": rows, "mean_over_cells": mean, "per_cell": per_cell}


# -------------------------------------------------------------------------------------------------- figures
def figures(output_dir: Path, tradeoff: dict[str, Any], mixture: dict[str, Any], cells: Sequence[Cell]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
                         "font.size": 8, "pdf.fonttype": 42})
    written = []
    labels = {row: label for row, label, _ in ALL_ROWS}
    figure, axes = plt.subplots(1, len(DATASETS), figsize=(7.0, 2.4), sharey=False)
    for axis, name in zip(axes, DATASETS):
        for cell in (c for c in cells if c.dataset == name):
            chosen = [r for r in tradeoff["rows"] if r["cell"] == cell.name]
            line, = axis.plot([100 * r["clean_change"] for r in chosen], [100 * r["gain_3"] for r in chosen],
                              lw=0.9, label=labels[cell.row])
            point = tradeoff["operating"][cell.name]
            axis.plot(100 * point["clean_change"], 100 * point["gain_3"], "o", ms=4, color=line.get_color())
        axis.axvline(-0.5, color="0.5", lw=0.6, ls="--")
        axis.set_title(name)
        axis.set_xlabel("F1 change, aligned pairs (points)")
    axes[0].set_ylabel("F1 gain, level 3 (points)")
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        path = output_dir / f"figure_tradeoff{suffix}"
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.4, 2.4))
    styles = {"raw": ("Without correction", "0.4", "-"), "estimated": ("Always correct", "tab:red", "--"),
              "selector": ("Localizer + selector", "tab:blue", "-"), "oracle": ("Oracle correction", "0.6", ":")}
    for method, (label, colour, style) in styles.items():
        axis.plot([100 * r["p"] for r in mixture["mean_over_cells"]], [100 * r[method] for r in mixture["mean_over_cells"]],
                  style, color=colour, lw=1.0, label=label)
    axis.set_xlabel("Misaligned pairs (%)")
    axis.set_ylabel(f"F1 (%), mean of {len(cells)} cells")
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    for suffix, options in ((".pdf", {}), (".png", {"dpi": 300})):
        path = output_dir / f"figure_mixture{suffix}"
        figure.savefig(path, **options)
        written.append(path)
    plt.close(figure)
    return written


# ------------------------------------------------------------------------------------------------------ text
def pts(value: float) -> str:
    return f"{100 * value:+.1f}"


def render(ci: dict[str, Any], decomposition: dict[str, Any], transfer: dict[str, Any], mixture: dict[str, Any],
           cells: int) -> str:
    lines = ["[ci] selector minus raw, points, 95% tile-bootstrap interval"]
    for name, entry in ci["cells"].items():
        e = entry["selector_minus_raw"]
        lines.append(f"  {name:<20} tiles {entry['tiles']:>3} | clean {pts(e['0']['value'])} [{pts(e['0']['low'])}, "
                     f"{pts(e['0']['high'])}] | L3 {pts(e['3']['value'])} [{pts(e['3']['low'])}, {pts(e['3']['high'])}]"
                     f"{'' if e['3']['excludes_zero'] else '  (L3 interval includes 0)'}")
    lines.append(f"  counts: {ci['counts']}")
    lines.append("[ci] ours minus border-free augmentation baseline (DINOv3), clean / L1 / L2 / L3")
    for name, entry in ci["versus_augmentation"].items():
        e = entry["ours_minus_augmentation"]
        lines.append(f"  {name:<10} " + " | ".join(f"{pts(e[k]['value'])} [{pts(e[k]['low'])}, {pts(e[k]['high'])}]"
                                                   for k in ("0", "1", "2", "3")))
    lines.append(f"[decomposition] mean of {cells} cells, F1 %: raw / selector / always / best of two / oracle")
    for level, v in decomposition["mean_over_cells"].items():
        lines.append(f"  L{level}: {100 * v['raw']:.1f} / {100 * v['selector']:.1f} / {100 * v['estimated']:.1f} / "
                     f"{100 * v['best_of_two']:.1f} / {100 * v['oracle']:.1f}  | recoverable {pts(v['recoverable'])}, "
                     f"realized {pts(v['realized'])}, lost to localizer {pts(v['lost_to_localizer'])}, "
                     f"lost to selector {pts(v['lost_to_selector'])}")
    lines.append("[transfer] clean change mean / worst / unsafe | gain L3 mean")
    for kind, s in transfer["summary"].items():
        lines.append(f"  {kind:<15} n={s['pairs']:>2}  {pts(s['clean_change_mean'])} / {pts(s['clean_change_worst'])} / "
                     f"{s['unsafe']} | {pts(s['gain_3_mean'])}")
    lines.append(f"[mixture] mean of {cells} cells, F1 %: p -> raw / always / selector / oracle")
    for r in mixture["mean_over_cells"][::4]:
        lines.append(f"  p={r['p']:.2f}: {100 * r['raw']:.1f} / {100 * r['estimated']:.1f} / {100 * r['selector']:.1f} / "
                     f"{100 * r['oracle']:.1f}")
    return "\n".join(lines)


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="reports/p31/statistics")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--rows", nargs="+", default=[row for row, _, _ in ALL_ROWS],
                        help="detector rows to include; the default is all six")
    args = parser.parse_args()
    output_dir = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    cells = load_cells([row for row in ALL_ROWS if row[0] in args.rows])
    print(f"{len(cells)} cells reproduce the frozen test tables", flush=True)
    ci = run_ci(cells, args.repeats, args.seed)
    decomposition = run_decomposition(cells)
    transfer = run_transfer(cells)
    tradeoff = run_tradeoff(cells)
    mixture = run_mixture(cells)
    for stem, value in (("ci", ci), ("decomposition", decomposition), ("transfer", transfer)):
        (output_dir / f"{stem}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    (output_dir / "mixture.json").write_text(json.dumps({k: v for k, v in mixture.items() if k != "rows"}, indent=2),
                                             encoding="utf-8")
    (output_dir / "tradeoff_operating.json").write_text(json.dumps(tradeoff["operating"], indent=2), encoding="utf-8")
    write_rows(output_dir / "transfer.csv", transfer["entries"])
    write_rows(output_dir / "tradeoff.csv", tradeoff["rows"])
    write_rows(output_dir / "tradeoff_mean.csv", tradeoff["mean_over_cells"])
    write_rows(output_dir / "mixture.csv", mixture["rows"])
    for path in figures(output_dir, tradeoff, mixture, cells):
        print(path)
    text = render(ci, decomposition, transfer, mixture, len(cells))
    (output_dir / "statistics.txt").write_text(text, encoding="utf-8")
    Path(args.output_root, "LATEST").write_text(str(output_dir).replace("\\", "/") + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
