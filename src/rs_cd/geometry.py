from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate_offset(tensor: torch.Tensor, offset: torch.Tensor) -> None:
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape BxCxHxW")
    if offset.shape != (tensor.shape[0], 2):
        raise ValueError(
            f"offset must have shape {(tensor.shape[0], 2)}, got "
            f"{tuple(offset.shape)}"
        )


def scale_input_offset(
    offset_pixels: torch.Tensor,
    image_size: tuple[int, int],
    feature_size: tuple[int, int],
) -> torch.Tensor:
    """Convert ``(dx, dy)`` from input pixels to feature-map pixels."""

    if offset_pixels.ndim != 2 or offset_pixels.shape[1] != 2:
        raise ValueError("offset_pixels must have shape Bx2")
    image_height, image_width = image_size
    feature_height, feature_width = feature_size
    if min(image_height, image_width, feature_height, feature_width) < 1:
        raise ValueError("image and feature sizes must be positive")
    stride = offset_pixels.new_tensor(
        [image_width / feature_width, image_height / feature_height]
    )
    return offset_pixels / stride


def warp_reference_from_target(
    target: torch.Tensor,
    offset: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "reflection",
) -> torch.Tensor:
    """Sample a target tensor in the coordinate frame of a reference tensor.

    The convention is deliberately explicit: an offset ``(dx, dy)`` means
    that reference position ``(x, y)`` corresponds to target position
    ``(x + dx, y + dy)``. Therefore a target image that was translated right
    by ``dx > 0`` is aligned back to the reference by passing the same
    positive ``dx`` to this function.

    Offsets are expressed in pixels of ``target`` and have shape ``Bx2``.
    """

    _validate_offset(target, offset)
    batch, _, height, width = target.shape
    y = (
        (torch.arange(height, device=target.device, dtype=target.dtype) + 0.5)
        * (2.0 / height)
        - 1.0
    )
    x = (
        (torch.arange(width, device=target.device, dtype=target.dtype) + 0.5)
        * (2.0 / width)
        - 1.0
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = base_grid.unsqueeze(0).expand(batch, -1, -1, -1).clone()
    typed_offset = offset.to(device=target.device, dtype=target.dtype)
    grid[..., 0] += typed_offset[:, 0, None, None] * (2.0 / width)
    grid[..., 1] += typed_offset[:, 1, None, None] * (2.0 / height)
    warped = F.grid_sample(
        target,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    # Preserve an exact no-op for clean samples. Besides making the coordinate
    # audit unambiguous, this prevents a nominally disabled alignment branch
    # from introducing grid construction round-off.
    zero_offset = typed_offset.abs().amax(dim=1) == 0
    if bool(zero_offset.any()):
        warped = torch.where(
            zero_offset[:, None, None, None],
            target,
            warped,
        )
    return warped


def warp_sampling_valid_mask(
    offset: torch.Tensor,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return pixels whose target sampling coordinate is inside the image."""

    if offset.ndim != 2 or offset.shape[1] != 2:
        raise ValueError("offset must have shape Bx2")
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    target_device = device if device is not None else offset.device
    typed = offset.to(device=target_device, dtype=torch.float32)
    y = torch.arange(height, device=target_device, dtype=torch.float32)
    x = torch.arange(width, device=target_device, dtype=torch.float32)
    query_y = y[None, :, None] + typed[:, 1, None, None]
    query_x = x[None, None, :] + typed[:, 0, None, None]
    valid_y = (query_y >= 0.0) & (query_y <= float(height - 1))
    valid_x = (query_x >= 0.0) & (query_x <= float(width - 1))
    return (valid_y & valid_x).unsqueeze(1).to(torch.float32)


def interior_valid_mask(
    batch_size: int,
    height: int,
    width: int,
    border: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a fixed common-support mask for fair registration comparison."""

    if batch_size < 1 or height < 1 or width < 1:
        raise ValueError("batch and spatial sizes must be positive")
    if border < 0:
        raise ValueError("border must be non-negative")
    if 2 * border >= min(height, width):
        raise ValueError("border removes the complete image")
    mask = torch.ones(
        batch_size,
        1,
        height,
        width,
        dtype=torch.float32,
        device=device,
    )
    if border:
        mask[:, :, :border, :] = 0.0
        mask[:, :, -border:, :] = 0.0
        mask[:, :, :, :border] = 0.0
        mask[:, :, :, -border:] = 0.0
    return mask
