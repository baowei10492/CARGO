"""Differentiable 2-D transform tools for the P27 localizer (torch).

Conventions match rs_cd.affine_registration: pixel centres sit at integer
coordinates, a 2 x 3 matrix A maps a reference point p to the target point
A p, and warping a target "back" samples it at A p. Version 1 of the
localizer estimates scale + translation (no rotation, main record 192.2).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _batched(points: torch.Tensor, batch: int) -> torch.Tensor:
    if points.dim() == 2:
        return points.unsqueeze(0).expand(batch, -1, -1)
    return points


# Element-wise arithmetic only: matmul would be cast to float16 under
# autocast, and weighted sums of squared pixel coordinates overflow there.


def apply_affine(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """(B, 2, 3) matrices applied to (N, 2) or (B, N, 2) points -> (B, N, 2)."""

    points = _batched(points, matrix.shape[0]).to(matrix.dtype)
    x, y = points[..., 0], points[..., 1]
    m = matrix[:, :, None, :]
    return torch.stack(
        [m[:, 0, :, 0] * x + m[:, 0, :, 1] * y + m[:, 0, :, 2],
         m[:, 1, :, 0] * x + m[:, 1, :, 1] * y + m[:, 1, :, 2]],
        dim=-1,
    )


def compose(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    """p -> outer(inner(p))."""

    rows = []
    for r in range(2):
        a, b, c = outer[:, r, 0], outer[:, r, 1], outer[:, r, 2]
        rows.append(torch.stack([
            a * inner[:, 0, 0] + b * inner[:, 1, 0],
            a * inner[:, 0, 1] + b * inner[:, 1, 1],
            a * inner[:, 0, 2] + b * inner[:, 1, 2] + c,
        ], dim=1))
    return torch.stack(rows, dim=1)


def invert(matrix: torch.Tensor) -> torch.Tensor:
    a, b, c = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    d, e, f = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    det = a * e - b * d
    top = torch.stack([e / det, -b / det, (b * f - c * e) / det], dim=1)
    bottom = torch.stack([-d / det, a / det, (c * d - a * f) / det], dim=1)
    return torch.stack([top, bottom], dim=1)


def scale_translation(scale: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(scale)
    return torch.stack(
        [torch.stack([scale, zero, tx], dim=1), torch.stack([zero, scale, ty], dim=1)], dim=1
    )


def identity(batch: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.eye(2, 3, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)


def fit_scale_translation(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = 1e-6,
    scale_bounds: tuple[float, float] = (0.25, 4.0),
) -> torch.Tensor:
    """Weighted least squares for target = s * source + t, per batch item."""

    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    source_mean = (weights[..., None] * source).sum(dim=1) / total
    target_mean = (weights[..., None] * target).sum(dim=1) / total
    source_centred = source - source_mean[:, None]
    target_centred = target - target_mean[:, None]
    numerator = (weights * (source_centred * target_centred).sum(dim=-1)).sum(dim=1)
    denominator = (weights * (source_centred * source_centred).sum(dim=-1)).sum(dim=1).clamp_min(eps)
    scale = (numerator / denominator).clamp(*scale_bounds)
    shift = target_mean - scale[:, None] * source_mean
    return scale_translation(scale, shift[:, 0], shift[:, 1])


def robust_fit_scale_translation(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    iterations: int,
    huber_px: float,
) -> torch.Tensor:
    """Iteratively reweighted least squares with Huber weights on the residual."""

    matrix = fit_scale_translation(source, target, weights)
    for _ in range(int(iterations)):
        residual = (apply_affine(matrix, source) - target).norm(dim=-1)
        huber = torch.where(residual <= huber_px, torch.ones_like(residual), huber_px / residual.clamp_min(1e-6))
        matrix = fit_scale_translation(source, target, weights * huber)
    return matrix


def grid(height: int, width: int, count: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    ys = torch.linspace(0.0, height - 1.0, count, device=device, dtype=dtype)
    xs = torch.linspace(0.0, width - 1.0, count, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def transfer_distance(estimate: torch.Tensor, truth: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """(B, N) point distances between two transforms."""

    return (apply_affine(estimate, points) - apply_affine(truth, points)).norm(dim=-1)


def mean_displacement(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    points = _batched(points, matrix.shape[0]).to(matrix)
    return (apply_affine(matrix, points) - points).norm(dim=-1).mean(dim=1)


def pixel_centres(height: int, width: int, stride: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Centre of every stride-``stride`` token in input pixels, row-major (x, y)."""

    offset = (stride - 1) / 2.0
    ys = torch.arange(height, device=device, dtype=dtype) * stride + offset
    xs = torch.arange(width, device=device, dtype=dtype) * stride + offset
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)


def warp_image(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """out(p) = image(matrix @ p), bilinear, zero outside (cv2 WARP_INVERSE_MAP semantics)."""

    batch, _, height, width = image.shape
    points = pixel_centres(height, width, 1, device=image.device, dtype=matrix.dtype)
    mapped = apply_affine(matrix, points)
    normalized = torch.stack(
        [(2.0 * mapped[..., 0] + 1.0) / width - 1.0, (2.0 * mapped[..., 1] + 1.0) / height - 1.0], dim=-1
    ).view(batch, height, width, 2)
    return F.grid_sample(image, normalized.to(image.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
