"""P27 F1 step: detector F1 on affinely perturbed crops, with and without correction.

Paths per crop and level (docs/P27_F1_AMENDMENT.md, P25 protocol section 4):
  raw              the perturbed T2 as it is
  oracle           the perturbed T2 sampled back with the true matrix
  estimated        the perturbed T2 sampled back with the localizer's estimate
  estimated_gated  estimated when the localizer says "misaligned", raw otherwise
Every path and level of a crop is scored on the same pixels: those whose image
under each level's true matrix lies inside the crop.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from rs_cd.affine_registration import CLEAN_LEVEL, align
from rs_cd.metrics.binary import BinaryChangeMetrics

PATHS = ("raw", "oracle", "estimated", "estimated_gated")
DETECTOR_PATHS = ("raw", "oracle", "estimated")   # estimated_gated reuses one of these


def scoring_mask(matrices: Sequence[np.ndarray], size: int) -> np.ndarray:
    """Pixels p with every matrix @ p inside the crop (element-wise, no BLAS)."""

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    inside = np.ones((size, size), dtype=bool)
    for m in matrices:
        u = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
        v = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
        inside &= (u >= 0.0) & (u <= size - 1.0) & (v >= 0.0) & (v <= size - 1.0)
    return inside


def path_images(perturbed: np.ndarray, truth: np.ndarray, estimate: np.ndarray) -> dict[str, np.ndarray]:
    """The three images the detector sees for one level (gating picks later)."""

    return {
        "raw": perturbed,
        "oracle": align(perturbed, truth),
        "estimated": align(perturbed, estimate),
    }


def gated_source(triggered: bool) -> str:
    return "estimated" if triggered else "raw"


@torch.no_grad()
def confusion_counts(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, threshold: float) -> list[list[int]]:
    """[tp, fp, fn, tn] per batch item, with BinaryChangeMetrics' rules.

    logits (B, 1, H, W); target and valid (1, 1, H, W) or (B, 1, H, W).
    """

    predictions = torch.sigmoid(logits.float()) >= threshold
    truth = torch.broadcast_to(target >= 0.5, predictions.shape)
    keep = torch.broadcast_to(valid >= 0.5, predictions.shape)
    positive, negative = predictions & keep, ~predictions & keep
    counts = torch.stack(
        [
            (positive & truth).flatten(1).sum(1),
            (positive & ~truth).flatten(1).sum(1),
            (negative & truth).flatten(1).sum(1),
            (negative & ~truth).flatten(1).sum(1),
        ],
        dim=1,
    )
    return counts.cpu().tolist()


def add_counts(metrics: BinaryChangeMetrics, counts: Sequence[int]) -> None:
    tp, fp, fn, tn = (int(value) for value in counts)
    metrics.true_positive += tp
    metrics.false_positive += fp
    metrics.false_negative += fn
    metrics.true_negative += tn


def _fraction(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) < 1e-12 else numerator / denominator


def f1_decision(
    f1: dict[str, dict[str, float]],
    *,
    levels: Sequence[int],
    primary: str,
    binding_levels: Sequence[int],
    min_gain: float,
    max_clean_drop: float,
    min_clean_raw_f1: float,
    oracle_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Verdict for one dataset from its F1 table {level: {path: f1}}.

    ``oracle_tolerance``: how far the clean oracle may sit from the clean raw path. The
    identity warp is exact, so the default demands equality; a detector whose inference
    draws per-sample random numbers (CD-Lamba) differs slightly between batch positions.
    """

    clean = f1[str(CLEAN_LEVEL)]
    per_level = {}
    for level in levels:
        row = f1[str(level)]
        per_level[str(level)] = {
            "harm": clean["raw"] - row["raw"],
            "ceiling": row["oracle"] - row["raw"],
            "gain": row[primary] - row["raw"],
            "gain_estimated": row["estimated"] - row["raw"],
            "recovery": _fraction(row[primary] - row["raw"], row["oracle"] - row["raw"]),
        }
    clean_drop = clean["raw"] - clean[primary]
    instrument = {
        "clean_raw_f1": clean["raw"],
        "clean_raw_f1_min": min_clean_raw_f1,
        "clean_oracle_difference": abs(clean["oracle"] - clean["raw"]),
        "clean_oracle_tolerance": oracle_tolerance,
        "clean_oracle_equals_raw": abs(clean["oracle"] - clean["raw"]) <= oracle_tolerance,
    }
    instrument["passed"] = instrument["clean_raw_f1"] >= min_clean_raw_f1 and instrument["clean_oracle_equals_raw"]
    checks = {
        **{f"gain_level_{level}": per_level[str(level)]["gain"] > min_gain for level in binding_levels},
        "clean_drop": clean_drop <= max_clean_drop,
    }
    if not instrument["passed"]:
        verdict = "instrument_failed"
    else:
        verdict = "positive_gain" if all(checks.values()) else "no_positive_gain"
    return {
        "verdict": verdict,
        "checks": checks,
        "clean_drop": clean_drop,
        "clean_drop_estimated": clean["raw"] - clean["estimated"],
        "levels": per_level,
        "instrument": instrument,
    }
