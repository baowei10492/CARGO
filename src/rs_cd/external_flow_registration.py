"""P26: DiffRegCD's per-dataset registration head as a similarity localizer.

The head predicts, on a coarse Hc x Wc grid, a normalized displacement
decoded from 32 x 32 displacement classes. Its training loss
(losses/roma_flow_cls_256.py) resizes the pixel ground-truth flow of the
unperturbed reference image to the coarse grid with align_corners=True and
normalizes it by 2 / (size - 1); the forward pass decodes on a slightly
different codebook. Which decoding, sign and grid reproduce a known
perturbation is therefore calibrated on self-pairs before any measurement
(docs/P26_DIFFREGCD_REGISTRATION_PROTOCOL.md). Everything here is pure numpy.
"""
from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

DECODINGS = ("loss", "model")
SIGNS = (1, -1)
GRIDS = ("reference", "perturbed")


def convention_key(decoding: str, sign: int, grid: str) -> str:
    return f"{decoding}|{'+' if sign > 0 else '-'}|{grid}"


def parse_convention(key: str) -> tuple[str, int, str]:
    decoding, sign, grid = key.split("|")
    if decoding not in DECODINGS or sign not in "+-" or grid not in GRIDS:
        raise ValueError(f"unknown convention {key!r}")
    return decoding, (1 if sign == "+" else -1), grid


ALL_CONVENTIONS = tuple(convention_key(d, s, g) for d, s, g in product(DECODINGS, SIGNS, GRIDS))


def coarse_points(coarse_height: int, coarse_width: int, height: int, width: int) -> np.ndarray:
    """Pixel position of every coarse cell under align_corners=True resizing, row-major."""

    ys = np.linspace(0.0, height - 1.0, coarse_height)
    xs = np.linspace(0.0, width - 1.0, coarse_width)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def normalized_to_pixels(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    """[Hc, Wc, 2] normalized (x, y) offsets -> [Hc * Wc, 2] pixels (align_corners=True)."""

    flow = np.asarray(flow, dtype=np.float64)
    pixels = np.empty((flow.shape[0] * flow.shape[1], 2), dtype=np.float64)
    pixels[:, 0] = flow[..., 0].ravel() * (width - 1) / 2.0
    pixels[:, 1] = flow[..., 1].ravel() * (height - 1) / 2.0
    return pixels


def correspondences(
    flow_pixels: np.ndarray, points: np.ndarray, *, sign: int, grid: str
) -> tuple[np.ndarray, np.ndarray]:
    """(reference points, perturbed points) under one convention.

    grid "reference": the field lives on the unperturbed image, q <-> q + s f.
    grid "perturbed": the field lives on the perturbed image, q + s f <-> q.
    """

    moved = points + sign * flow_pixels
    if grid == "reference":
        return points, moved
    if grid == "perturbed":
        return moved, points
    raise ValueError(f"unknown grid {grid!r}")


INPUT_ORDERS = ("A_perturbed", "A_reference")


def ordered_inputs(reference: Any, perturbed: Any, input_order: str) -> tuple[Any, Any]:
    """(A, B) for the registration head.

    P26 fed the perturbed image as A, following the README; the network
    expects the unperturbed reference as A (main record 191.3), which the
    P26-A amendment uses.
    """

    if input_order == "A_perturbed":
        return perturbed, reference
    if input_order == "A_reference":
        return reference, perturbed
    raise ValueError(f"unknown input order {input_order!r}")


def calibration_outcome(calibration: dict[str, Any], continue_when_imprecise: bool) -> str:
    """measure / descriptive / stop.

    An ambiguous calibration always stops. An unambiguous one that misses
    only the precision tolerance may still be measured, as a description,
    when the protocol says so (P26-A); its verdict is then never a pass.
    """

    if calibration["passed"]:
        return "measure"
    if calibration["unambiguous"] and continue_when_imprecise:
        return "descriptive"
    return "stop"


def choose_convention(
    worst_errors: dict[str, float | None], *, tolerance: float, ambiguity_ratio: float
) -> dict[str, Any]:
    """Preregistered pick from self-pair errors (worst crop per convention).

    The best convention must meet the tolerance and beat every convention of
    a different sign or grid by the ambiguity ratio; the two decodings of the
    same geometry differ only by a few percent in scale and are not compared.
    """

    cleaned = {
        key: (math.inf if value is None or not math.isfinite(value) else float(value))
        for key, value in worst_errors.items()
    }
    missing = sorted(set(ALL_CONVENTIONS) - set(cleaned))
    if missing:
        raise KeyError(f"self-check lacks conventions {missing}")
    best = min(sorted(cleaned), key=lambda key: cleaned[key])
    _, best_sign, best_grid = parse_convention(best)
    rivals = [
        value for key, value in cleaned.items()
        if parse_convention(key)[1:] != (best_sign, best_grid)
    ]
    rival = min(rivals)
    unambiguous = math.isfinite(cleaned[best]) and cleaned[best] <= ambiguity_ratio * rival
    within = cleaned[best] <= tolerance + 1e-9
    return {
        "selected": best,
        "worst_error": cleaned[best],
        "best_rival_error": rival,
        "tolerance_px": tolerance,
        "ambiguity_ratio": ambiguity_ratio,
        "unambiguous": unambiguous,
        "within_tolerance": within,
        "passed": unambiguous and within,
        "errors": {key: (None if math.isinf(value) else value) for key, value in cleaned.items()},
    }
