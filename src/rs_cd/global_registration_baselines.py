"""Classical global registration baselines in the registration module's family (reviewer request, 2026-10-08).

    phasecorr_fm   Fourier-Mellin phase correlation (Reddy and Chatterji, IEEE TIP 1996): the isotropic scale from the
                   phase correlation of the log-polar, high-pass filtered magnitude spectra (rotation fixed at 0; half
                   of the angle axis, since the magnitude spectrum repeats after 180 degrees), then the translation
                   from the phase correlation of the reference and the target with the scale undone
    ecc_ts         enhanced correlation coefficient maximisation (Evangelidis and Psarakis, IEEE TPAMI 2008): affine
                   motion on a three-level Gaussian pyramid started at the identity, projected onto translation +
                   isotropic scale by least squares on the 17 x 17 grid

Both take uint8 RGB crops and return a SimilarityEstimate whose matrix maps reference (T1) pixels to target (T2)
pixels, like every other localizer of the ablation (rs_cd.p27_baseline_localizers.SimilarityBaseline wraps them).
Scaling is about the crop centre (size / 2), as in rs_cd.affine_registration.perturbation_matrix.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from rs_cd.affine_registration import SimilarityEstimate, grid_points
from rs_cd.translation_scale_baselines import SCALE_LIMITS, weighted_scale_translation


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _scale_about_centre(scale: float, size: int) -> np.ndarray:
    return cv2.getRotationMatrix2D((size / 2.0, size / 2.0), 0.0, float(scale)).astype(np.float64)


def _highpass(rows: int, cols: int) -> np.ndarray:
    """Reddy and Chatterji's emphasis filter (1 - X)(2 - X), X = cos(pi xi) cos(pi eta), on the centred spectrum."""
    xi = np.linspace(-0.5, 0.5, rows, endpoint=False)[:, None]
    eta = np.linspace(-0.5, 0.5, cols, endpoint=False)[None, :]
    x = np.cos(np.pi * xi) * np.cos(np.pi * eta)
    return ((1.0 - x) * (2.0 - x)).astype(np.float32)


class FourierMellinScaleTranslation:
    """Fourier-Mellin phase correlation for translation + isotropic scale (3 DoF)."""

    def __init__(self, *, size: int = 256, radial_samples: int | None = None) -> None:
        self.size = int(size)
        self.window = cv2.createHanningWindow((self.size, self.size), cv2.CV_32F)
        self.highpass = _highpass(self.size, self.size)
        self.radial = int(radial_samples or 2 * self.size)          # log-radius samples
        self.max_radius = self.size / 2.0
        self.k_log = self.radial / math.log(self.max_radius)          # columns per unit of log radius

    def _log_polar(self, gray: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray * self.window)))
        magnitude = np.log1p(spectrum * self.highpass).astype(np.float32)
        polar = cv2.warpPolar(magnitude, (self.radial, self.size), (self.size / 2.0, self.size / 2.0),
                              self.max_radius, cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR)
        return np.ascontiguousarray(polar[: self.size // 2])          # angles 0 .. 180 degrees

    def __call__(self, reference: np.ndarray, target: np.ndarray, key: str = "") -> SimilarityEstimate:
        ref, tgt = _gray(reference), _gray(target)
        if ref.shape != (self.size, self.size) or tgt.shape != ref.shape:
            raise ValueError("FourierMellinScaleTranslation expects size x size crops")
        (dx, _), _ = cv2.phaseCorrelate(self._log_polar(ref), self._log_polar(tgt))
        scale = float(np.clip(math.exp(-dx / self.k_log), *SCALE_LIMITS))     # target = reference magnified by scale
        unscaled = cv2.warpAffine(tgt, _scale_about_centre(1.0 / scale, self.size), (self.size, self.size),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        (sx, sy), response = cv2.phaseCorrelate(ref, unscaled, self.window)  # unscaled = reference shifted by t / s
        matrix = _scale_about_centre(scale, self.size)
        matrix[:, 2] += scale * np.array([sx, sy])
        converged = bool(np.isfinite(matrix).all())
        return SimilarityEstimate(matrix=matrix if converged else np.eye(2, 3), converged=converged,
                                  inliers=0, inlier_fraction=float(response) if np.isfinite(response) else None)


class EccScaleTranslation:
    """ECC with affine motion on a Gaussian pyramid, projected onto translation + isotropic scale (3 DoF)."""

    def __init__(self, *, size: int = 256, levels: int = 3, iterations: int = 100, epsilon: float = 1e-5,
                 gauss_filter: int = 5, grid_count: int = 17) -> None:
        self.size = int(size)
        self.levels = int(levels)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), float(epsilon))
        self.gauss_filter = int(gauss_filter)
        self.points = grid_points(self.size, self.size, int(grid_count))

    def __call__(self, reference: np.ndarray, target: np.ndarray, key: str = "") -> SimilarityEstimate:
        pyramid = [(_gray(reference), _gray(target))]
        for _ in range(self.levels - 1):
            pyramid.append((cv2.pyrDown(pyramid[-1][0]), cv2.pyrDown(pyramid[-1][1])))
        warp = np.eye(2, 3, dtype=np.float32)
        solved = 0
        for step, (ref, tgt) in enumerate(reversed(pyramid)):
            if step > 0:
                warp[:, 2] *= 2.0                                          # translation to the finer level
            try:
                _, warp = cv2.findTransformECC(ref, tgt, warp, cv2.MOTION_AFFINE, self.criteria, None,
                                               self.gauss_filter)
                solved += 1
            except cv2.error:                                              # not converged at this level
                continue
        if solved == 0 or not np.isfinite(warp).all():
            return SimilarityEstimate()
        # The warp maps template (reference) pixels to input (target) pixels: template(x) ~ input(W x).
        source = self.points
        mapped = source @ warp[:, :2].astype(np.float64).T + warp[:, 2].astype(np.float64)
        fitted = weighted_scale_translation(source, mapped, np.ones(len(source)))
        if fitted is None:
            return SimilarityEstimate()
        scale, translation = fitted
        matrix = np.array([[scale, 0.0, translation[0]], [0.0, scale, translation[1]]], dtype=np.float64)
        return SimilarityEstimate(matrix=matrix, converged=True, inliers=solved, inlier_fraction=None)
