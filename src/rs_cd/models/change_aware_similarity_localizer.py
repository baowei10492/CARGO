"""P27 version 1: change-aware scale + translation localizer.

docs/P27_OWN_LOCALIZER_DESIGN.md (main record 192-192.2). A frozen DINOv3
ConvNeXt-Tiny gives stride-4 and stride-8 features; small trainable
adapters embed them for registration.

coarse (stride 8)  all-pairs correlation, dual-softmax matching, a local
                   soft-argmax around each best match, a change-aware weight
                   per reference token, and an IRLS scale + translation fit
fine (stride 4)    the target is warped back with the current estimate and
                   re-encoded; a local correlation (radius 3 cells) gives a
                   residual displacement per token, fitted the same way and
                   composed with the current estimate; repeated
misalignment       a small head on the final estimate and match statistics

The output matrix maps a reference pixel to its target pixel (pixel centres
at integer coordinates), as in rs_cd.affine_registration.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rs_cd.similarity_geometry import (
    compose,
    grid,
    mean_displacement,
    pixel_centres,
    robust_fit_scale_translation,
    warp_image,
)


class FrozenDinoV3Features(nn.Module):
    """Stride-4 and stride-8 features of a frozen timm DINOv3 ConvNeXt."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            f"local-dir:{checkpoint_dir}", pretrained=True, features_only=True, out_indices=(0, 1)
        )
        if [int(v) for v in self.model.feature_info.reduction()] != [4, 8]:
            raise ValueError(f"unexpected strides {self.model.feature_info.reduction()}")
        self.channels = tuple(int(v) for v in self.model.feature_info.channels())
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenDinoV3Features":
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stride4, stride8 = self.model(images)
        return stride4, stride8


def _adapter(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )


def _neighbour_table(height: int, width: int, radius: int) -> torch.Tensor:
    """(H*W, (2r+1)^2) flat indices of the clamped window around every cell."""

    rows = torch.arange(height).view(-1, 1).expand(height, width).reshape(-1)
    cols = torch.arange(width).view(1, -1).expand(height, width).reshape(-1)
    offsets = torch.arange(-radius, radius + 1)
    dy = offsets.view(-1, 1).expand(-1, offsets.numel()).reshape(-1)
    dx = offsets.view(1, -1).expand(offsets.numel(), -1).reshape(-1)
    neighbour_rows = (rows[:, None] + dy[None]).clamp(0, height - 1)
    neighbour_cols = (cols[:, None] + dx[None]).clamp(0, width - 1)
    return neighbour_rows * width + neighbour_cols


class ChangeAwareScaleTranslationLocalizer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        stride4_channels: int,
        stride8_channels: int,
        coarse_dim: int = 128,
        fine_dim: int = 64,
        initial_temperature: float = 0.05,
        coarse_window: int = 1,
        fine_radius: int = 3,
        fine_iterations: int = 2,
        irls_iterations: int = 3,
        coarse_huber_px: float = 8.0,
        fine_huber_px: float = 2.0,
        misalignment_grid: int = 9,
        use_change_weights: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # False (P33 ablation, retrained): correspondences keep their matching confidence only, and the change
        # probability is not an input of the misalignment head; saved models without the key keep True.
        self.use_change_weights = bool(use_change_weights)
        self.coarse_adapter = _adapter(stride8_channels, coarse_dim)
        self.fine_adapter = _adapter(stride4_channels, fine_dim)
        self.change_head = nn.Sequential(nn.Linear(3 * coarse_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.misalignment_head = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 1))
        self.log_coarse_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.log_fine_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        self.coarse_window = int(coarse_window)
        self.fine_radius = int(fine_radius)
        self.fine_iterations = int(fine_iterations)
        self.irls_iterations = int(irls_iterations)
        self.coarse_huber_px = float(coarse_huber_px)
        self.fine_huber_px = float(fine_huber_px)
        self.misalignment_grid = int(misalignment_grid)
        side = 2 * self.fine_radius + 1
        offsets = torch.stack(
            torch.meshgrid(torch.arange(side) - self.fine_radius, torch.arange(side) - self.fine_radius, indexing="ij"),
            dim=-1,
        ).reshape(-1, 2).flip(-1).float()  # (dx, dy) per window cell, row-major
        self.register_buffer("fine_offsets", offsets, persistent=False)

    # ------------------------------------------------------------------ parts
    def _coarse(self, reference8: torch.Tensor, target8: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = reference8.shape
        ref = F.normalize(self.coarse_adapter(reference8).float(), dim=1)
        tgt = F.normalize(self.coarse_adapter(target8).float(), dim=1)
        ref_flat = ref.flatten(2).transpose(1, 2)  # (B, N, D)
        tgt_flat = tgt.flatten(2).transpose(1, 2)
        temperature = self.log_coarse_temperature.exp().clamp(0.005, 1.0)
        logits = ref_flat @ tgt_flat.transpose(1, 2) / temperature  # (B, N, N)
        dual = torch.softmax(logits, dim=2) * torch.softmax(logits, dim=1)
        confidence, best = dual.max(dim=2)  # (B, N)

        table = _neighbour_table(height, width, self.coarse_window).to(logits.device)  # (N, K)
        neighbours = table[best]  # (B, N, K)
        local_logits = torch.gather(logits, 2, neighbours)
        local_probs = torch.softmax(local_logits, dim=2)
        centres = pixel_centres(height, width, 8, device=logits.device)  # (N, 2)
        matched = (local_probs[..., None] * centres[neighbours]).sum(dim=2)  # (B, N, 2)

        matched_features = torch.gather(tgt_flat, 1, best[..., None].expand(-1, -1, tgt_flat.shape[-1]))
        change_logits = self.change_head(
            torch.cat([ref_flat, matched_features, (ref_flat - matched_features).abs()], dim=-1)
        )[..., 0]
        weights = confidence * torch.sigmoid(-change_logits) if self.use_change_weights else confidence
        source = centres.unsqueeze(0).expand(batch, -1, -1)
        matrix = robust_fit_scale_translation(
            source, matched, weights, iterations=self.irls_iterations, huber_px=self.coarse_huber_px
        )
        return {
            "matrix": matrix,
            "confidence": confidence.view(batch, height, width),
            "change_logits": change_logits.view(batch, height, width),
            "weights": weights.view(batch, height, width),
        }

    def _fine(self, reference4: torch.Tensor, aligned4: torch.Tensor, change_logits: torch.Tensor):
        batch, _, height, width = reference4.shape
        ref = F.normalize(self.fine_adapter(reference4).float(), dim=1)
        tgt = F.normalize(self.fine_adapter(aligned4).float(), dim=1)
        side = 2 * self.fine_radius + 1
        windows = F.unfold(tgt, kernel_size=side, padding=self.fine_radius)  # (B, D*K, N)
        windows = windows.view(batch, ref.shape[1], side * side, height * width)
        correlation = (ref.flatten(2).unsqueeze(2) * windows).sum(dim=1)  # (B, K, N)
        temperature = self.log_fine_temperature.exp().clamp(0.005, 1.0)
        probs = torch.softmax(correlation / temperature, dim=1)
        offsets = self.fine_offsets.to(probs.dtype)
        displacement = torch.stack(
            [(probs * offsets[None, :, 0, None]).sum(dim=1), (probs * offsets[None, :, 1, None]).sum(dim=1)], dim=-1
        ) * 4.0  # (B, N, 2) pixels
        confidence = probs.max(dim=1).values  # (B, N)
        change = F.interpolate(change_logits.unsqueeze(1), size=(height, width), mode="nearest")[:, 0]
        weights = confidence * torch.sigmoid(-change.flatten(1)) if self.use_change_weights else confidence
        centres = pixel_centres(height, width, 4, device=ref.device).unsqueeze(0).expand(batch, -1, -1)
        residual = robust_fit_scale_translation(
            centres, centres + displacement, weights, iterations=self.irls_iterations, huber_px=self.fine_huber_px
        )
        return residual, confidence.view(batch, height, width)

    # ---------------------------------------------------------------- forward
    def forward(self, reference: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
        if reference.shape != target.shape:
            raise ValueError("reference and target shapes must match")
        batch = reference.shape[0]
        features4, features8 = self.backbone(torch.cat([reference, target], dim=0))
        reference4, target8 = features4[:batch], features8[batch:]
        coarse = self._coarse(features8[:batch], target8)
        matrix = coarse["matrix"]
        stages = [matrix]
        fine_confidence = None
        for _ in range(self.fine_iterations):
            aligned = warp_image(target, matrix.detach())
            aligned4, _ = self.backbone(aligned)
            residual, fine_confidence = self._fine(reference4, aligned4, coarse["change_logits"])
            matrix = compose(matrix, residual)
            stages.append(matrix)

        points = grid(reference.shape[-2], reference.shape[-1], self.misalignment_grid, device=reference.device)
        scale = matrix[:, 0, 0]
        summary = torch.stack(
            [
                mean_displacement(matrix, points) / 16.0,
                (scale - 1.0) * 10.0,
                matrix[:, :, 2].norm(dim=1) / 16.0,
                coarse["confidence"].flatten(1).mean(dim=1),
                fine_confidence.flatten(1).mean(dim=1) if fine_confidence is not None else torch.zeros_like(scale),
                torch.sigmoid(coarse["change_logits"]).flatten(1).mean(dim=1) if self.use_change_weights
                else torch.zeros_like(scale),
            ],
            dim=1,
        )
        misaligned_logit = self.misalignment_head(summary.detach())[:, 0]
        return {
            "matrix": matrix,
            "stage_matrices": stages,
            "coarse_confidence": coarse["confidence"],
            "coarse_weights": coarse["weights"],
            "change_logits": coarse["change_logits"],
            "fine_confidence": fine_confidence,
            "misaligned_logit": misaligned_logit,
        }


def build_localizer(config: dict[str, Any], backbone: nn.Module | None = None) -> ChangeAwareScaleTranslationLocalizer:
    spec = dict(config)
    if backbone is None:
        backbone = FrozenDinoV3Features(spec.pop("checkpoint_dir"))
        channels = backbone.channels
    else:
        spec.pop("checkpoint_dir", None)
        channels = tuple(spec.pop("backbone_channels"))
    spec.pop("backbone_channels", None)
    return ChangeAwareScaleTranslationLocalizer(
        backbone, stride4_channels=int(channels[0]), stride8_channels=int(channels[1]), **spec
    )
