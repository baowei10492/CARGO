"""Change density as a covariate of global-translation localization failure.

Main record 157 found that on the WHU validation strip the frozen P7C
localizer fails where a large fraction of the analysis window changed, but
the association is confounded with location. These helpers back the two
preregistered follow-ups in docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md: a
train-region dose-response (A1) and a cross-dataset check on existing P21
per-sample localizer errors (A2). Everything here is deterministic and pure.
"""
from __future__ import annotations

import statistics
from typing import Any, Sequence

import numpy as np

DEFAULT_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.05),
    (0.05, 0.10),
    (0.10, 0.15),
    (0.15, 1.0),
)


def assign_bin(fraction: float, bins: Sequence[Sequence[float]] = DEFAULT_BINS) -> int:
    """Half-open bins; the last bin also includes its upper edge."""

    last = len(bins) - 1
    for index, (low, high) in enumerate(bins):
        if low <= fraction < high or (index == last and fraction == high):
            return index
    raise ValueError(f"change fraction {fraction} lies outside the bins")


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""

    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        stop = start
        while stop + 1 < len(array) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation of average ranks.

    Computed with element-wise sums rather than np.corrcoef: on Windows the
    MKL-backed numpy aborts inside BLAS once torch has loaded its own OpenMP
    runtime in the same process, and the A1 tool uses both.
    """

    if len(x) != len(y):
        raise ValueError("spearman needs paired sequences")
    if len(x) < 3:
        return None
    centred_x = average_ranks(x)
    centred_y = average_ranks(y)
    centred_x = centred_x - centred_x.mean()
    centred_y = centred_y - centred_y.mean()
    denominator = float(np.sqrt(np.sum(centred_x * centred_x) * np.sum(centred_y * centred_y)))
    if denominator == 0.0:
        return None
    return float(np.sum(centred_x * centred_y)) / denominator


def block_change_counts(
    label: np.ndarray, *, block: int, columns: int, rows: int, threshold: int
) -> np.ndarray:
    """Changed-pixel count of every ``block`` cell of the tile grid (rows x columns)."""

    region = label[: rows * block, : columns * block] > threshold
    if region.shape != (rows * block, columns * block):
        raise ValueError("label raster is smaller than the requested grid")
    return region.reshape(rows, block, columns, block).sum(axis=(1, 3), dtype=np.int64)


def window_change_fractions(
    counts: np.ndarray, *, blocks_per_window: int, block: int
) -> list[dict[str, Any]]:
    """Every grid-aligned window of ``blocks_per_window`` cells and its change fraction."""

    rows, columns = counts.shape
    size = int(blocks_per_window)
    if size > rows or size > columns:
        raise ValueError("window larger than the grid")
    area = float((size * block) ** 2)
    windows = []
    for row in range(rows - size + 1):
        for column in range(columns - size + 1):
            changed = int(counts[row : row + size, column : column + size].sum())
            windows.append(
                {"left": column * block, "top": row * block, "change_fraction": changed / area}
            )
    return windows


def select_stratified(
    windows: Sequence[dict[str, Any]],
    *,
    per_bin: int,
    min_spacing: int,
    bins: Sequence[Sequence[float]] = DEFAULT_BINS,
) -> list[dict[str, Any]]:
    """Greedy row-major pick per bin with a Chebyshev spacing between window origins."""

    chosen: list[dict[str, Any]] = []
    for index in range(len(bins)):
        members = sorted(
            (w for w in windows if assign_bin(float(w["change_fraction"]), bins) == index),
            key=lambda w: (int(w["top"]), int(w["left"])),
        )
        picked: list[dict[str, Any]] = []
        for window in members:
            if len(picked) >= per_bin:
                break
            if all(
                max(abs(int(window["left"]) - int(p["left"])), abs(int(window["top"]) - int(p["top"])))
                >= min_spacing
                for p in picked
            ):
                picked.append({**window, "bin": index})
        chosen.extend(picked)
    return chosen


def _rate(hits: int, n: int) -> float | None:
    return hits / n if n else None


def probe_bin_stats(
    windows: Sequence[dict[str, Any]], *, bins: Sequence[Sequence[float]] = DEFAULT_BINS
) -> list[dict[str, Any]]:
    """Per-bin counts from per-window outcomes (s2_hit, s3_hit, clean_q, s2_error, s3_error)."""

    stats = []
    for index, (low, high) in enumerate(bins):
        members = [w for w in windows if int(w["bin"]) == index]
        n = len(members)
        entry = {
            "bin": index,
            "range": [float(low), float(high)],
            "n": n,
            "s2_hits": sum(1 for w in members if w["s2_hit"]),
            "s3_hits": sum(1 for w in members if w["s3_hit"]),
            "clean_q_hits": sum(1 for w in members if w["clean_q"]),
            "mean_s2_error": statistics.fmean(float(w["s2_error"]) for w in members) if n else None,
            "mean_s3_error": statistics.fmean(float(w["s3_error"]) for w in members) if n else None,
        }
        entry["s2_pck1"] = _rate(entry["s2_hits"], n)
        entry["s3_pck1"] = _rate(entry["s3_hits"], n)
        entry["clean_q"] = _rate(entry["clean_q_hits"], n)
        stats.append(entry)
    return stats


def _merge(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    n = first["n"] + second["n"]
    merged = {
        "bin": f"{first['bin']}+{second['bin']}",
        "range": [first["range"][0], second["range"][1]],
        "n": n,
        "s3_hits": first["s3_hits"] + second["s3_hits"],
        "clean_q_hits": first["clean_q_hits"] + second["clean_q_hits"],
    }
    merged["s3_pck1"] = _rate(merged["s3_hits"], n)
    merged["clean_q"] = _rate(merged["clean_q_hits"], n)
    return merged


def probe_decision(stats: Sequence[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    """A1 verdict from the lowest and highest change bins (docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md)."""

    minimum = int(thresholds["min_windows_per_bin"])

    def works(entry: dict[str, Any]) -> bool:
        return (
            entry["n"] >= minimum
            and entry["s3_pck1"] >= thresholds["works_pck1_min"]
            and entry["clean_q"] <= thresholds["works_clean_q_max"]
        )

    def fails(entry: dict[str, Any]) -> bool:
        return entry["n"] >= minimum and (
            entry["s3_pck1"] <= thresholds["fails_pck1_max"]
            or entry["clean_q"] >= thresholds["fails_clean_q_min"]
        )

    low = stats[0]
    high = stats[-1]
    merged = False
    if high["n"] < minimum:
        high = _merge(stats[-2], stats[-1])
        merged = True
    if not works(low):
        verdict = "inconclusive_low_bin_failed"
    elif high["n"] < minimum:
        verdict = "inconclusive_insufficient_high_change_windows"
    elif fails(high):
        verdict = "reproduced_outside_val"
    elif works(high):
        verdict = "not_reproduced"
    else:
        verdict = "partial"
    return {"verdict": verdict, "merged_high_bin": merged, "low_bin": low, "high_bin": high}


def association_verdict(
    change: Sequence[float],
    error: Sequence[float],
    *,
    thresholds: dict[str, float],
    bins: Sequence[Sequence[float]] = DEFAULT_BINS,
) -> dict[str, Any]:
    """A2 verdict for one dataset: rank association plus a high-vs-low bin contrast."""

    rho = spearman(change, error)
    low = [float(e) for c, e in zip(change, error) if assign_bin(float(c), bins) == 0]
    high = [float(e) for c, e in zip(change, error) if assign_bin(float(c), bins) == len(bins) - 1]
    minimum = int(thresholds["min_samples_per_bin"])
    low_mean = statistics.fmean(low) if low else None
    high_mean = statistics.fmean(high) if high else None
    contrast_ok = (
        len(low) >= minimum
        and len(high) >= minimum
        and high_mean >= thresholds["high_over_low_min"] * low_mean
    )
    if rho is not None and rho >= thresholds["rho_min"] and contrast_ok:
        verdict = "associated"
    elif rho is not None and rho <= thresholds["rho_null_max"]:
        verdict = "not_associated"
    else:
        verdict = "weak_or_inconclusive"
    return {
        "verdict": verdict,
        "spearman": rho,
        "n": len(change),
        "low_bin_n": len(low),
        "high_bin_n": len(high),
        "low_bin_mean_error": low_mean,
        "high_bin_mean_error": high_mean,
    }


def overall_association(dsifn: str, gzcd: str) -> str:
    if dsifn == gzcd == "associated":
        return "cross_dataset_supported"
    if dsifn == gzcd == "not_associated":
        return "cross_dataset_not_supported"
    return "mixed"
