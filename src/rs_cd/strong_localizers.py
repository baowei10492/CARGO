"""P23: published global-translation localizers behind one interface.

P22 stopped at Gate 1 (main record 156): the frozen LEVIR localizer (P7C)
was exact on the WHU train region but drifted on the validation strip, and
the drift tracked the changed fraction of the analysis window (main record
157). P23 asks whether a stronger, published matcher closes that gap before
anything of ours changes. Every candidate answers the same question on the
same 3072 px window: which native-pixel translation (dx, dy) maps reference
position (x, y) to target position (x + dx, y + dy), the rs_cd.geometry
convention.

The helpers below (consensus, trigger, summaries, selection) are pure and
unit-tested. Matcher adapters import kornia and romatch lazily, so this
module loads without either installed. Protocol:
docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from rs_cd.change_density import DEFAULT_BINS, assign_bin
from rs_cd.physical_shift import CLEAN_KEY, box_downsample
from rs_cd.registration_controls import _parabolic_offset, to_grayscale
from rs_cd.robustness import _stable_seed

PHYSICAL_KEYS: tuple[str, ...] = tuple(
    f"physical_shift:s{severity}:t2" for severity in (1, 2, 3)
)
PCK_THRESHOLDS: tuple[float, ...] = (1.0, 2.0, 3.0, 6.0, 12.0)
REFERENCE_ROLE = "reference"


@dataclass(frozen=True)
class TranslationEstimate:
    """One native-pixel translation estimate.

    ``gate`` is the candidate's own zero-shift decision when it has one
    (P7C's q gate); generic matchers leave it ``None`` and the shared
    magnitude rule decides instead (see ``triggered``).
    """

    dx: float
    dy: float
    converged: bool
    support: int = -1
    inlier_fraction: float | None = None
    gate: bool | None = None

    def scaled(self, factor: int) -> "TranslationEstimate":
        return TranslationEstimate(
            dx=self.dx * factor,
            dy=self.dy * factor,
            converged=self.converged,
            support=self.support,
            inlier_fraction=self.inlier_fraction,
            gate=self.gate,
        )


def not_converged(support: int = 0) -> TranslationEstimate:
    """A failed estimate reports a zero offset; it never invents a displacement."""

    return TranslationEstimate(0.0, 0.0, False, int(support), None)


def translation_consensus(
    translations: torch.Tensor,
    *,
    threshold: float,
    min_matches: int,
    max_abs_shift: float | None,
    max_matches: int | None = None,
) -> TranslationEstimate:
    """Deterministic exhaustive RANSAC on a pure-translation model.

    The same rule as registration_controls.sift_translation_offset: every
    observed match is a hypothesis, the one with the most matches within
    ``threshold`` wins (first index on ties), and the estimate is the mean
    of its inliers. No random sampling, so no seed. Callers order matches by
    confidence before passing ``max_matches``; the search window
    ``max_abs_shift`` is the same fixed prior for every candidate.
    """

    points = translations.detach().reshape(-1, 2).to(torch.float64)
    if max_abs_shift is not None:
        limit = float(max_abs_shift)
        points = points[(points.abs() <= limit).all(dim=1)]
    if max_matches is not None and points.shape[0] > int(max_matches):
        points = points[: int(max_matches)]
    count = int(points.shape[0])
    if count < int(min_matches):
        return not_converged(count)
    distances = torch.cdist(
        points, points, compute_mode="donot_use_mm_for_euclid_dist"
    )
    within = distances <= float(threshold)
    best = int(torch.argmax(within.sum(dim=1)))
    inliers = points[within[best]]
    mean = inliers.mean(dim=0)
    return TranslationEstimate(
        dx=float(mean[0]),
        dy=float(mean[1]),
        converged=True,
        support=int(inliers.shape[0]),
        inlier_fraction=float(inliers.shape[0]) / float(count),
    )


def triggered(estimate: TranslationEstimate, threshold_native_px: float) -> bool:
    """Zero-shift decision: the candidate's own gate, else a magnitude rule.

    The magnitude rule (``|offset| > threshold``) uses one fixed threshold for
    every generic candidate: 6 native px, one P7C coarse pixel, the same
    tolerance P22 Gate 0 used for "clean stays within one coarse pixel".
    """

    if estimate.gate is not None:
        return bool(estimate.gate)
    if not estimate.converged:
        return False
    return math.hypot(estimate.dx, estimate.dy) > float(threshold_native_px)


def applied_offset(
    estimate: TranslationEstimate, *, rounded: bool
) -> tuple[float, float]:
    """Offset used for warping. Rounding keeps integer shifts exact under
    bilinear sampling, so a clean pair with a 0.3 px estimate is not blurred."""

    if rounded:
        return float(round(estimate.dx)), float(round(estimate.dy))
    return float(estimate.dx), float(estimate.dy)


def working_view(unit_window: torch.Tensor, factor: int) -> torch.Tensor:
    """3xHxW unit image at the candidate's working resolution (exact box mean)."""

    if factor == 1:
        return unit_window
    return box_downsample(unit_window.unsqueeze(0), factor)[0]


@torch.no_grad()
def windowed_phase_correlation(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: int,
    subpixel: bool = True,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """Vectorised equivalent of registration_controls.phase_correlation_offset.

    Same windowing, cross-power spectrum, row-major first-maximum rule inside
    the ``radius`` window and parabolic refinement; the reference function
    loops over the window in Python, which is too slow for a 96 px radius on
    3072 px windows. A parity test pins the two together.
    """

    reference_gray = to_grayscale(reference, "unit").float()
    target_gray = to_grayscale(target, "unit").float()
    if reference_gray.shape != target_gray.shape:
        raise ValueError("reference and target sizes must match")
    height, width = reference_gray.shape
    radius = int(radius)
    if radius < 0 or 2 * radius + 1 > min(height, width):
        raise ValueError("radius does not fit inside the image")
    window_y = torch.hann_window(
        height, periodic=False, device=reference_gray.device, dtype=reference_gray.dtype
    )
    window_x = torch.hann_window(
        width, periodic=False, device=reference_gray.device, dtype=reference_gray.dtype
    )
    window = window_y[:, None] * window_x[None, :]
    left = (reference_gray - reference_gray.mean()) * window
    right = (target_gray - target_gray.mean()) * window
    cross_power = torch.conj(torch.fft.rfft2(left)) * torch.fft.rfft2(right)
    cross_power = cross_power / cross_power.abs().clamp_min(eps)
    correlation = torch.fft.irfft2(cross_power, s=(height, width)).real
    centred = torch.roll(correlation, shifts=(radius, radius), dims=(0, 1))
    search = centred[: 2 * radius + 1, : 2 * radius + 1].cpu()
    flat_index = int(torch.argmax(search))
    row, column = divmod(flat_index, 2 * radius + 1)
    dy = row - radius
    dx = column - radius
    score = float(search[row, column])
    refined_dx, refined_dy = float(dx), float(dy)
    if subpixel:
        host = correlation.cpu()
        refined_dx += _parabolic_offset(
            host, dx, dy, axis="x", height=height, width=width
        )
        refined_dy += _parabolic_offset(
            host, dx, dy, axis="y", height=height, width=width
        )
    return refined_dx, refined_dy, score


class PhaseCorrelationLocalizer:
    """Classical Fourier phase correlation with parabolic sub-pixel refinement."""

    def __init__(self, *, working_factor: int, max_abs_shift_native: float) -> None:
        self.factor = int(working_factor)
        self.radius = int(math.floor(float(max_abs_shift_native) / self.factor))

    def estimate(
        self, t1: torch.Tensor, t2: torch.Tensor, *, sample_key: str
    ) -> TranslationEstimate:
        dx, dy, score = windowed_phase_correlation(
            working_view(t1, self.factor),
            working_view(t2, self.factor),
            radius=self.radius,
        )
        return TranslationEstimate(dx, dy, True, -1, None).scaled(self.factor)


def _uint8_gray(unit: torch.Tensor) -> np.ndarray:
    gray = to_grayscale(unit, "unit").detach().cpu().numpy()
    return np.ascontiguousarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8))


class SiftLocalizer:
    """SIFT keypoints, Lowe ratio test, translation consensus (OpenCV, CPU)."""

    def __init__(
        self,
        *,
        working_factor: int,
        max_features: int,
        ratio: float,
        consensus_threshold: float,
        min_matches: int,
        max_abs_shift_native: float,
    ) -> None:
        import cv2

        self.cv2 = cv2
        self.factor = int(working_factor)
        self.max_features = int(max_features)
        self.ratio = float(ratio)
        self.threshold = float(consensus_threshold)
        self.min_matches = int(min_matches)
        self.limit = float(max_abs_shift_native) / self.factor

    def estimate(
        self, t1: torch.Tensor, t2: torch.Tensor, *, sample_key: str
    ) -> TranslationEstimate:
        cv2 = self.cv2
        reference = _uint8_gray(working_view(t1, self.factor))
        target = _uint8_gray(working_view(t2, self.factor))
        sift = cv2.SIFT_create(nfeatures=self.max_features)
        keypoints_a, descriptors_a = sift.detectAndCompute(reference, None)
        keypoints_b, descriptors_b = sift.detectAndCompute(target, None)
        if (
            descriptors_a is None
            or descriptors_b is None
            or len(keypoints_a) < 2
            or len(keypoints_b) < 2
        ):
            return not_converged()
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_a, descriptors_b, k=2)
        scored: list[tuple[float, float, float]] = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            best, second = pair
            if second.distance <= 0 or best.distance >= self.ratio * second.distance:
                continue
            source = keypoints_a[best.queryIdx].pt
            destination = keypoints_b[best.trainIdx].pt
            scored.append(
                (
                    best.distance / second.distance,
                    destination[0] - source[0],
                    destination[1] - source[1],
                )
            )
        # Most distinctive matches first; stable for equal ratios.
        scored.sort(key=lambda item: item[0])
        translations = torch.tensor(
            [[item[1], item[2]] for item in scored], dtype=torch.float64
        ).reshape(-1, 2)
        return translation_consensus(
            translations,
            threshold=self.threshold,
            min_matches=self.min_matches,
            max_abs_shift=self.limit,
        ).scaled(self.factor)


class LightGlueDiskLocalizer:
    """DISK keypoints matched by LightGlue (kornia), translation consensus."""

    def __init__(
        self,
        *,
        working_factor: int,
        max_keypoints: int,
        consensus_threshold: float,
        min_matches: int,
        max_abs_shift_native: float,
        device: torch.device,
    ) -> None:
        import kornia.feature as KF

        self.KF = KF
        self.factor = int(working_factor)
        self.max_keypoints = int(max_keypoints)
        self.threshold = float(consensus_threshold)
        self.min_matches = int(min_matches)
        self.limit = float(max_abs_shift_native) / self.factor
        self.device = device
        self.disk = KF.DISK.from_pretrained("depth").to(device).eval()
        self.matcher = KF.LightGlueMatcher("disk").to(device).eval()

    @torch.inference_mode()
    def estimate(
        self, t1: torch.Tensor, t2: torch.Tensor, *, sample_key: str
    ) -> TranslationEstimate:
        KF = self.KF
        reference = working_view(t1, self.factor).float()
        target = working_view(t2, self.factor).float()
        features = self.disk(
            torch.stack([reference, target]).to(self.device),
            self.max_keypoints,
            pad_if_not_divisible=True,
        )
        keypoints_a, descriptors_a = features[0].keypoints, features[0].descriptors
        keypoints_b, descriptors_b = features[1].keypoints, features[1].descriptors
        if len(keypoints_a) < self.min_matches or len(keypoints_b) < self.min_matches:
            return not_converged()

        def lafs(points: torch.Tensor) -> torch.Tensor:
            scale = torch.ones(1, len(points), 1, 1, device=points.device)
            return KF.laf_from_center_scale_ori(points[None], scale)

        size = torch.tensor(reference.shape[-2:], device=self.device)
        _, indices = self.matcher(
            descriptors_a,
            descriptors_b,
            lafs(keypoints_a),
            lafs(keypoints_b),
            hw1=size,
            hw2=size,
        )
        if indices.numel() == 0:
            return not_converged()
        translations = keypoints_b[indices[:, 1]] - keypoints_a[indices[:, 0]]
        # LightGlue returns at most max_keypoints matches: consensus uses all of
        # them, so no confidence ordering is needed.
        return translation_consensus(
            translations,
            threshold=self.threshold,
            min_matches=self.min_matches,
            max_abs_shift=self.limit,
        ).scaled(self.factor)


def _to_pil(unit: torch.Tensor):
    from PIL import Image

    array = (
        (unit.detach().clamp(0.0, 1.0) * 255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


class RomaOutdoorLocalizer:
    """RoMa dense matcher (outdoor weights), sampled matches, translation consensus."""

    def __init__(
        self,
        *,
        working_factor: int,
        num_matches: int,
        consensus_threshold: float,
        min_matches: int,
        max_abs_shift_native: float,
        device: torch.device,
        seed: int,
        model_kwargs: dict[str, Any] | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if model_factory is None:
            from romatch import roma_outdoor as model_factory
        self.factor = int(working_factor)
        self.num_matches = int(num_matches)
        self.threshold = float(consensus_threshold)
        self.min_matches = int(min_matches)
        self.limit = float(max_abs_shift_native) / self.factor
        self.device = device
        self.seed = int(seed)
        self.model = model_factory(device=device, **(model_kwargs or {}))

    @torch.inference_mode()
    def estimate(
        self, t1: torch.Tensor, t2: torch.Tensor, *, sample_key: str
    ) -> TranslationEstimate:
        reference = working_view(t1, self.factor)
        target = working_view(t2, self.factor)
        height, width = (int(value) for value in reference.shape[-2:])
        warp, certainty = self.model.match(
            _to_pil(reference), _to_pil(target), device=self.device
        )
        # RoMa's balanced sampling draws from the global generator: seed it
        # per pair so every estimate is reproducible.
        torch.manual_seed(_stable_seed(self.seed, sample_key, "p23", 0, "roma"))
        matches, confidence = self.model.sample(warp, certainty, num=self.num_matches)
        points_a, points_b = self.model.to_pixel_coordinates(
            matches, height, width, height, width
        )
        order = torch.argsort(confidence.reshape(-1), descending=True, stable=True)
        translations = (points_b - points_a).reshape(-1, 2)[order]
        return translation_consensus(
            translations,
            threshold=self.threshold,
            min_matches=self.min_matches,
            max_abs_shift=self.limit,
        ).scaled(self.factor)


# --------------------------------------------------------------------------
# Screening summaries (label-free except the optional change-fraction column)
# --------------------------------------------------------------------------


def _rate(flags: Sequence[bool]) -> float | None:
    return (sum(1 for flag in flags if flag) / len(flags)) if flags else None


def row_error(row: dict[str, Any]) -> float:
    return math.hypot(
        float(row["est_dx"]) - float(row["gt_dx"]),
        float(row["est_dy"]) - float(row["gt_dy"]),
    )


def scenario_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    errors = [row_error(row) for row in rows]
    seconds = [float(row["seconds"]) for row in rows]
    entry: dict[str, Any] = {
        "n": len(rows),
        "native_mean_epe": statistics.fmean(errors),
        "native_median_epe": statistics.median(errors),
        "trigger_rate": _rate([_as_bool(row["trigger"]) for row in rows]),
        "nonconverged_rate": _rate([not _as_bool(row["converged"]) for row in rows]),
        "median_seconds": statistics.median(seconds),
    }
    for threshold in PCK_THRESHOLDS:
        entry[f"native_pck{threshold:g}"] = _rate([e <= threshold + 1e-6 for e in errors])
    return entry


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return bool(value)


def group_rows(
    rows: Sequence[dict[str, Any]], *keys: str
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    return grouped


def candidate_verdict(
    by_scenario: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, Any]:
    missing = [key for key in (*PHYSICAL_KEYS, CLEAN_KEY) if key not in by_scenario]
    if missing:
        raise KeyError(f"screen rows are missing scenarios: {missing}")
    pck2 = {key: by_scenario[key]["native_pck2"] for key in PHYSICAL_KEYS}
    clean_trigger = by_scenario[CLEAN_KEY]["trigger_rate"]
    return {
        "physical_pck2": pck2,
        "mean_physical_pck2": statistics.fmean(pck2.values()),
        "tracks": all(value >= thresholds["tracking_pck2_min"] for value in pck2.values()),
        "clean_trigger_rate": clean_trigger,
        "clean_passed": clean_trigger <= thresholds["clean_trigger_max"],
        "median_seconds": statistics.median(
            entry["median_seconds"] for entry in by_scenario.values()
        ),
    }


def select_candidate(
    verdicts: dict[str, dict[str, Any]],
    roles: dict[str, str],
    *,
    margin: float,
) -> str | None:
    """Preregistered pick among tracking, non-reference candidates.

    Highest mean physical PCK@2; every candidate within ``margin`` of that
    best is a tie, broken by the lower clean trigger rate, then the lower
    median runtime, then the name.
    """

    eligible = {
        name: verdict
        for name, verdict in verdicts.items()
        if verdict["tracks"] and roles.get(name) != REFERENCE_ROLE
    }
    if not eligible:
        return None
    best = max(verdict["mean_physical_pck2"] for verdict in eligible.values())
    tied = [
        name
        for name, verdict in eligible.items()
        if verdict["mean_physical_pck2"] >= best - float(margin) - 1e-12
    ]
    return min(
        tied,
        key=lambda name: (
            eligible[name]["clean_trigger_rate"],
            eligible[name]["median_seconds"],
            name,
        ),
    )


def change_bin_summary(
    rows: Sequence[dict[str, Any]],
    bins: Sequence[Sequence[float]] = DEFAULT_BINS,
) -> list[dict[str, Any]]:
    """Descriptive only: physical S3 PCK@2 and clean trigger per change-fraction bin."""

    summary = []
    for index, (low, high) in enumerate(bins):
        members = [
            row
            for row in rows
            if row.get("change_fraction") not in (None, "")
            and assign_bin(float(row["change_fraction"]), bins) == index
        ]
        s3 = [row for row in members if row["scenario"] == PHYSICAL_KEYS[-1]]
        clean = [row for row in members if row["scenario"] == CLEAN_KEY]
        summary.append(
            {
                "bin": index,
                "range": [float(low), float(high)],
                "tiles": len(clean),
                "physical_s3_pck2": _rate([row_error(row) <= 2.0 + 1e-6 for row in s3]),
                "physical_s3_mean_epe": (
                    statistics.fmean(row_error(row) for row in s3) if s3 else None
                ),
                "clean_trigger_rate": _rate([_as_bool(row["trigger"]) for row in clean]),
            }
        )
    return summary


def clean_agreement(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[str],
    *,
    tolerance: float = 3.0,
) -> dict[str, Any]:
    """Clean pairs on which at least two of ``candidates`` move by the same amount.

    Unrelated localizers (P7C and a tracking matcher) agreeing on a non-zero
    clean offset is evidence of a genuine residual misregistration in the
    data rather than a localizer error; reported, never used as a gate.
    """

    clean = [
        row
        for row in rows
        if row["scenario"] == CLEAN_KEY and row["candidate"] in set(candidates)
    ]
    by_id = group_rows(clean, "id")
    agreeing = []
    for (sample_id,), members in sorted(by_id.items()):
        fired = [row for row in members if _as_bool(row["trigger"])]
        if len(fired) < 2:
            continue
        xs = [float(row["est_dx"]) for row in fired]
        ys = [float(row["est_dy"]) for row in fired]
        if max(xs) - min(xs) <= tolerance and max(ys) - min(ys) <= tolerance:
            agreeing.append(
                {
                    "id": sample_id,
                    "candidates": sorted(str(row["candidate"]) for row in fired),
                    "median_dx": statistics.median(xs),
                    "median_dy": statistics.median(ys),
                }
            )
    return {
        "candidates": list(candidates),
        "tolerance_px": float(tolerance),
        "clean_pairs": len(by_id),
        "agreeing_nonzero": len(agreeing),
        "pairs": agreeing,
    }


def anchor_offsets(
    observed: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """P7C rows of this run against the authoritative P22 offsets file.

    Compared per (id, scenario): the conditional estimate within ``tolerance``
    and the q gate exactly. Every reference row must be observed.
    """

    def key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row["id"]), str(row["scenario"])

    seen = {key(row): row for row in observed}
    missing = [key(row) for row in reference if key(row) not in seen]
    mismatches = []
    worst = 0.0
    for row in reference:
        other = seen.get(key(row))
        if other is None:
            continue
        delta = max(
            abs(float(other["est_dx"]) - float(row["est_dx"])),
            abs(float(other["est_dy"]) - float(row["est_dy"])),
        )
        worst = max(worst, delta)
        if delta > tolerance or _as_bool(other["trigger"]) != _as_bool(row["q_gate"]):
            mismatches.append(
                {
                    "id": key(row)[0],
                    "scenario": key(row)[1],
                    "delta": delta,
                    "reference_gate": _as_bool(row["q_gate"]),
                    "observed_gate": _as_bool(other["trigger"]),
                }
            )
    return {
        "reference_rows": len(reference),
        "missing": len(missing),
        "mismatches": len(mismatches),
        "max_abs_delta": worst,
        "tolerance": float(tolerance),
        "examples": mismatches[:10],
        "passed": not missing and not mismatches and len(reference) > 0,
    }


def selfcheck_summary(
    rows: Sequence[dict[str, Any]], *, tolerance: float
) -> dict[str, Any]:
    """Sign, unit and scale check on a window shifted against itself."""

    by_candidate = group_rows(rows, "candidate")
    results = {}
    for (name,), members in by_candidate.items():
        worst = max(row_error(row) for row in members)
        results[name] = {
            "n": len(members),
            "max_native_error": worst,
            "all_converged": all(_as_bool(row["converged"]) for row in members),
            "passed": worst <= tolerance + 1e-6
            and all(_as_bool(row["converged"]) for row in members),
        }
    return {
        "tolerance_px": float(tolerance),
        "candidates": results,
        "passed": bool(results) and all(entry["passed"] for entry in results.values()),
    }


def compare_f1_tables(
    observed: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    path_map: dict[str, str],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """P23 Gate 1 anchor: the P7C paths of this run against the P22 Gate 1 table."""

    deltas: dict[str, float] = {}
    missing = []
    for scenario, by_path in reference.items():
        for reference_path, observed_path in path_map.items():
            name = f"{observed_path}:{scenario}"
            if scenario not in observed or observed_path not in observed[scenario]:
                missing.append(name)
                continue
            deltas[name] = float(observed[scenario][observed_path]) - float(
                by_path[reference_path]
            )
    return {
        "deltas": deltas,
        "missing": missing,
        "tolerance": float(tolerance),
        "max_abs_delta": max((abs(v) for v in deltas.values()), default=None),
        "passed": bool(deltas)
        and not missing
        and all(abs(value) <= tolerance for value in deltas.values()),
    }
