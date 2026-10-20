"""Untrained coarse matching on frozen backbone tokens (P27 feasibility check).

Mutual nearest neighbours between the L2-normalised token grids of two
images, mapped to pixel positions, give correspondences for a robust
similarity fit. This is the untrained version of the coarse stage in
docs/P27_OWN_LOCALIZER_DESIGN.md; the check asks how far it survives
rotation before any training.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def token_centers(height: int, width: int, stride: int) -> np.ndarray:
    """Pixel centre of every token of a stride-``stride`` grid, row-major (x, y).

    ConvNeXt patchifies 4 x 4 and then halves the grid with 2 x 2 steps, so the
    token at column j covers input pixels [j * stride, (j + 1) * stride).
    """

    offset = (stride - 1) / 2.0
    xs = np.arange(width, dtype=np.float64) * stride + offset
    ys = np.arange(height, dtype=np.float64) * stride + offset
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


@torch.no_grad()
def mutual_nearest(reference: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual nearest neighbours of two C x H x W token grids.

    Returns (reference indices, target indices, cosine similarities), row-major.
    """

    if reference.ndim != 3 or reference.shape != target.shape:
        raise ValueError("expected two C x H x W tensors of the same shape")
    left = F.normalize(reference.float().flatten(1).T, dim=1)
    right = F.normalize(target.float().flatten(1).T, dim=1)
    similarity = left @ right.T
    forward = similarity.argmax(dim=1)
    backward = similarity.argmax(dim=0)
    indices = torch.arange(left.shape[0], device=similarity.device)
    mutual = backward[forward] == indices
    ref_idx = indices[mutual]
    tgt_idx = forward[mutual]
    scores = similarity[ref_idx, tgt_idx]
    return ref_idx.cpu().numpy(), tgt_idx.cpu().numpy(), scores.cpu().numpy()


def correspondences(
    ref_idx: np.ndarray, tgt_idx: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return centers[ref_idx], centers[tgt_idx]
