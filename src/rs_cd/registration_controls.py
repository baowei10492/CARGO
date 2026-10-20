from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class RegistrationControlResult:
    dx: float
    dy: float
    score: float
    converged: bool = True


def to_unit_image(image: torch.Tensor, normalization: str) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape CxHxW")
    if normalization == "imagenet":
        mean = IMAGENET_MEAN.to(image)
        std = IMAGENET_STD.to(image)
        return (image * std + mean).clamp(0.0, 1.0)
    if normalization in {"unit", "none"}:
        unit = image.float()
        if normalization == "none":
            unit = unit / 255.0
        return unit.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported normalization: {normalization}")


def to_grayscale(image: torch.Tensor, normalization: str) -> torch.Tensor:
    unit = to_unit_image(image, normalization)
    if unit.shape[0] == 1:
        return unit[0]
    if unit.shape[0] != 3:
        raise ValueError("Expected one or three image channels")
    weights = unit.new_tensor((0.2989, 0.5870, 0.1140))
    return (unit * weights[:, None, None]).sum(dim=0)


def _signed_index(index: int, size: int) -> int:
    return index if index <= size // 2 else index - size


@torch.no_grad()
def _parabolic_offset(
    correlation: torch.Tensor,
    dx: int,
    dy: int,
    *,
    axis: str,
    height: int,
    width: int,
) -> float:
    """Sub-pixel peak refinement by a three-point parabola fit.

    Returns 0.0 when the three samples are degenerate (flat or a local
    minimum), so a failed fit never invents a displacement.
    """

    def value(offset_x: int, offset_y: int) -> float:
        return float(
            correlation[(dy + offset_y) % height, (dx + offset_x) % width]
        )

    centre = value(0, 0)
    if axis == "x":
        lower, upper = value(-1, 0), value(1, 0)
    else:
        lower, upper = value(0, -1), value(0, 1)
    denominator = lower - 2.0 * centre + upper
    if abs(denominator) < 1e-12:
        return 0.0
    delta = 0.5 * (lower - upper) / denominator
    if not -1.0 < delta < 1.0:
        return 0.0
    return float(delta)


def phase_correlation_offset(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    normalization: str = "imagenet",
    max_abs_shift: int | None = None,
    eps: float = 1e-8,
    subpixel: bool = False,
) -> RegistrationControlResult:
    """Estimate reference-to-target translation with phase correlation.

    With ``subpixel=True`` the integer peak is refined by fitting a
    parabola through its two neighbours along each axis, the standard
    refinement for correlation peaks. This matters for fairness: the
    learned localizer decodes onto a 2 px grid and cannot express a
    sub-pixel answer, so comparing it against an *unrefined* phase
    correlation on sub-pixel perturbations would handicap the classical
    baseline on precisely the axis where it has a structural advantage.
    """

    reference_gray = to_grayscale(reference, normalization).float()
    target_gray = to_grayscale(target, normalization).float()
    if reference_gray.shape != target_gray.shape:
        raise ValueError("reference and target sizes must match")
    height, width = reference_gray.shape
    window_y = torch.hann_window(
        height,
        periodic=False,
        device=reference_gray.device,
        dtype=reference_gray.dtype,
    )
    window_x = torch.hann_window(
        width,
        periodic=False,
        device=reference_gray.device,
        dtype=reference_gray.dtype,
    )
    window = window_y[:, None] * window_x[None, :]
    left = (reference_gray - reference_gray.mean()) * window
    right = (target_gray - target_gray.mean()) * window
    left_fft = torch.fft.rfft2(left)
    right_fft = torch.fft.rfft2(right)
    cross_power = torch.conj(left_fft) * right_fft
    cross_power = cross_power / cross_power.abs().clamp_min(eps)
    correlation = torch.fft.irfft2(cross_power, s=(height, width)).real

    if max_abs_shift is None:
        flat_index = int(correlation.argmax())
        peak_y, peak_x = divmod(flat_index, width)
        dx = _signed_index(peak_x, width)
        dy = _signed_index(peak_y, height)
        score = float(correlation[peak_y, peak_x])
    else:
        radius = int(max_abs_shift)
        if radius < 0:
            raise ValueError("max_abs_shift must be non-negative")
        candidates: list[tuple[float, int, int]] = []
        for dy_candidate in range(-radius, radius + 1):
            iy = dy_candidate % height
            for dx_candidate in range(-radius, radius + 1):
                ix = dx_candidate % width
                candidates.append(
                    (
                        float(correlation[iy, ix]),
                        dx_candidate,
                        dy_candidate,
                    )
                )
        score, dx, dy = max(candidates, key=lambda item: item[0])
    refined_dx = float(dx)
    refined_dy = float(dy)
    if subpixel:
        refined_dx += _parabolic_offset(
            correlation,
            dx,
            dy,
            axis="x",
            height=height,
            width=width,
        )
        refined_dy += _parabolic_offset(
            correlation,
            dx,
            dy,
            axis="y",
            height=height,
            width=width,
        )
    return RegistrationControlResult(
        dx=refined_dx,
        dy=refined_dy,
        score=float(score),
    )


def ecc_translation_offset(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    normalization: str = "imagenet",
    initial_offset: tuple[float, float] = (0.0, 0.0),
    iterations: int = 100,
    epsilon: float = 1e-6,
) -> RegistrationControlResult:
    """Estimate reference-to-target translation with OpenCV ECC.

    OpenCV's returned warp is used with ``WARP_INVERSE_MAP`` to sample the
    target in reference coordinates, so its translation follows this
    project's ``reference(x) <-> target(x + dx)`` convention.
    """

    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "ECC requires opencv-python-headless; install the geometry extra"
        ) from error

    reference_gray = (
        to_grayscale(reference, normalization).detach().cpu().numpy()
    )
    target_gray = to_grayscale(target, normalization).detach().cpu().numpy()
    reference_np = np.ascontiguousarray(reference_gray, dtype=np.float32)
    target_np = np.ascontiguousarray(target_gray, dtype=np.float32)
    warp = np.array(
        [
            [1.0, 0.0, float(initial_offset[0])],
            [0.0, 1.0, float(initial_offset[1])],
        ],
        dtype=np.float32,
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(iterations),
        float(epsilon),
    )
    try:
        score, estimated = cv2.findTransformECC(
            reference_np,
            target_np,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            1,
        )
    except cv2.error:
        return RegistrationControlResult(
            dx=float(initial_offset[0]),
            dy=float(initial_offset[1]),
            score=float("-inf"),
            converged=False,
        )
    return RegistrationControlResult(
        dx=float(estimated[0, 2]),
        dy=float(estimated[1, 2]),
        score=float(score),
        converged=True,
    )


@torch.no_grad()
def sift_translation_offset(
    reference: torch.Tensor,
    target: torch.Tensor,
    *,
    normalization: str = "imagenet",
    ratio: float = 0.75,
    ransac_threshold: float = 2.0,
    minimum_matches: int = 4,
    max_abs_shift: float | None = None,
) -> RegistrationControlResult:
    """Estimate translation with SIFT keypoints + a RANSAC consensus.

    Feature-based registration is the classical alternative a reviewer
    will ask about, and it is structurally different from phase
    correlation and ECC: it matches sparse repeatable structures rather
    than dense intensity, so it degrades differently under change.

    The estimate follows this project's convention throughout:
    ``reference(x, y) <-> target(x + dx, y + dy)``, so the translation is
    (target keypoint - reference keypoint). RANSAC runs on the
    translation model only, matching what the learned localizer is
    allowed to express, so neither side gets extra degrees of freedom.

    Returns ``converged=False`` with a zero offset when there are too few
    matches; the caller decides the fallback, and no silent guess is made.
    """

    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "SIFT requires opencv-python-headless; install the geometry "
            "extra"
        ) from error

    def _uint8(image: torch.Tensor) -> np.ndarray:
        gray = to_grayscale(image, normalization).detach().cpu().numpy()
        return np.ascontiguousarray(
            np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
        )

    reference_np = _uint8(reference)
    target_np = _uint8(target)
    sift = cv2.SIFT_create()
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        reference_np,
        None,
    )
    target_keypoints, target_descriptors = sift.detectAndCompute(
        target_np,
        None,
    )
    if (
        reference_descriptors is None
        or target_descriptors is None
        or len(reference_keypoints) < minimum_matches
        or len(target_keypoints) < minimum_matches
    ):
        return RegistrationControlResult(
            dx=0.0,
            dy=0.0,
            score=0.0,
            converged=False,
        )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(
        reference_descriptors,
        target_descriptors,
        k=2,
    )
    translations: list[tuple[float, float]] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance >= float(ratio) * second.distance:
            continue
        source = reference_keypoints[best.queryIdx].pt
        destination = target_keypoints[best.trainIdx].pt
        translations.append(
            (destination[0] - source[0], destination[1] - source[1])
        )
    if max_abs_shift is not None:
        # Same search window every other estimator gets. Without it,
        # repetitive urban structure produces large consistent false
        # correspondences and the consensus locks onto a spurious
        # cluster (LEVIR measured ~200 px, main record 85.1). This is
        # not giving SIFT the answer: the learned localizer is
        # architecturally confined to the same window.
        limit = float(max_abs_shift)
        translations = [
            item
            for item in translations
            if abs(item[0]) <= limit and abs(item[1]) <= limit
        ]
    if len(translations) < minimum_matches:
        return RegistrationControlResult(
            dx=0.0,
            dy=0.0,
            score=float(len(translations)),
            converged=False,
        )

    # Deterministic RANSAC on the translation model: every candidate is
    # an observed match, so there is no random sampling and no seed.
    candidates = np.asarray(translations, dtype=np.float64)
    distances = np.linalg.norm(
        candidates[:, None, :] - candidates[None, :, :],
        axis=2,
    )
    inlier_counts = (distances <= float(ransac_threshold)).sum(axis=1)
    best_index = int(np.argmax(inlier_counts))
    inliers = candidates[
        distances[best_index] <= float(ransac_threshold)
    ]
    consensus = inliers.mean(axis=0)
    return RegistrationControlResult(
        dx=float(consensus[0]),
        dy=float(consensus[1]),
        score=float(len(inliers)) / float(len(candidates)),
        converged=True,
    )
