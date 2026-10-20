"""P30 border-free geometry (docs/P30_BORDER_FREE_PROTOCOL.md).

The P27 harness perturbs and corrects the 256 px crop itself, so the perturbed
and the corrected T2 carry a zero border. Here both act on the whole T2 tile and
the crop is cut out afterwards:

    perturbed tile   P = warp(tile, M_tile)               (the observed, misaligned T2)
    corrected crop   warp(P, A_tile, inverse)[crop]        (A: true matrix or estimate)

with M_tile, A_tile the crop-frame matrices moved to the tile frame. The
corrected image is resampled twice, as in P27. Crops keep enough margin to the
tile edge that the perturbed crop never needs padding; an estimate that reaches
beyond the perturbed tile leaves zero pixels, which are counted.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from rs_cd.affine_crops import in_frame, source_inside_tile


class BorderFreeGeometry:
    """Perturbation and correction of one crop, both applied to the whole T2 tile."""

    def __init__(self, tile_t2: np.ndarray, top: int, left: int, size: int) -> None:
        self.tile = tile_t2
        self.top, self.left, self.size = int(top), int(left), int(size)
        self._perturbed: dict[bytes, np.ndarray] = {}
        self.missing: dict[str, int] = {}

    def _crop(self, image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image[self.top:self.top + self.size, self.left:self.left + self.size])

    def perturbed_tile(self, matrix: np.ndarray) -> np.ndarray:
        key = np.asarray(matrix, dtype=np.float64).tobytes()
        if key not in self._perturbed:
            if np.allclose(matrix, np.eye(2, 3)):
                self._perturbed[key] = self.tile
            else:
                height, width = self.tile.shape[:2]
                self._perturbed[key] = cv2.warpAffine(
                    self.tile, in_frame(matrix, self.top, self.left), (width, height),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return self._perturbed[key]

    def perturb(self, matrix: np.ndarray) -> np.ndarray:
        """The misaligned crop: real content up to its edges. Stops if the crop would need padding."""
        if not source_inside_tile(matrix, self.top, self.left, self.size, self.tile.shape[0]):
            raise SystemExit(f"crop y{self.top:04d}x{self.left:04d}: the perturbation reaches outside the tile")
        return self._crop(self.perturbed_tile(matrix))

    def corrected(self, truth: np.ndarray, correction: np.ndarray) -> np.ndarray:
        observed = self.perturbed_tile(truth)
        height, width = observed.shape[:2]
        aligned = cv2.warpAffine(observed, in_frame(correction, self.top, self.left), (width, height),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return self._crop(aligned)

    def _uncovered(self, truth: np.ndarray, correction: np.ndarray) -> int:
        """Pixels of the corrected crop that sample outside the content of the perturbed tile."""
        height, width = self.tile.shape[:2]
        ones = np.ones((height, width), dtype=np.uint8)
        content = cv2.warpAffine(ones, in_frame(truth, self.top, self.left), (width, height),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        covered = cv2.warpAffine(content, in_frame(correction, self.top, self.left), (width, height),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return int((self._crop(covered) == 0).sum())

    def path_images(self, perturbed: np.ndarray, truth: np.ndarray, estimate: np.ndarray) -> dict[str, np.ndarray]:
        """Same keys and order as rs_cd.affine_f1.path_images; ``perturbed`` is ignored (it is the raw view)."""
        self.missing[_key(truth)] = self._uncovered(truth, estimate)
        return {"raw": self.perturb(truth), "oracle": self.corrected(truth, truth),
                "estimated": self.corrected(truth, estimate)}

    def missing_pixels(self, truth: np.ndarray) -> int:
        """Pixels of the estimated view without content (the estimate reached beyond the perturbed tile)."""
        return self.missing.get(_key(truth), 0)


def _key(matrix: Any) -> bytes:
    return np.asarray(matrix, dtype=np.float64).tobytes()
