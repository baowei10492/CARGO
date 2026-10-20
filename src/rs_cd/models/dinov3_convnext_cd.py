from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class DifferenceFPNDecoder(nn.Module):
    """Top-down decoder for four absolute-difference feature maps."""

    def __init__(
        self,
        feature_channels: list[int],
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError("DifferenceFPNDecoder expects four feature levels")
        self.lateral = nn.ModuleList(
            [
                ConvNormAct(channels, decoder_channels, kernel_size=1)
                for channels in feature_channels
            ]
        )
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(decoder_channels, decoder_channels),
                    ConvNormAct(decoder_channels, decoder_channels),
                )
                for _ in range(3)
            ]
        )
        self.output = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        differences: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if len(differences) != 4:
            raise ValueError("Expected four temporal difference maps")
        projected = [
            lateral(feature)
            for lateral, feature in zip(
                self.lateral, differences, strict=True
            )
        ]
        x = projected[-1]
        for decoder_index, level in enumerate(range(2, -1, -1)):
            x = F.interpolate(
                x,
                size=projected[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = self.refine[decoder_index](x + projected[level])
        logits = self.output(x)
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class DinoV3ConvNeXtChangeDetector(nn.Module):
    """Shared DINOv3 ConvNeXt backbone with a difference-FPN decoder.

    This module is intentionally a clean backbone baseline. Temporal fusion is
    limited to commutative absolute differences; no proposed research module is
    included.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = True,
        trainable_stages: int | None = None,
        decoder_channels: int = 128,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "DINOv3 ConvNeXt requires timm>=1.0.20. "
                "Install with: python -m pip install 'timm>=1.0.20,<2'"
            ) from error

        self.backbone_name = str(backbone_name)
        if trainable_stages is None:
            trainable_stages = 0 if freeze_backbone else 4
        self.trainable_stages = int(trainable_stages)
        if not 0 <= self.trainable_stages <= 4:
            raise ValueError("trainable_stages must be between 0 and 4")
        self.freeze_backbone = self.trainable_stages == 0
        local_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path
            else None
        )
        if local_path is not None and not local_path.exists():
            raise FileNotFoundError(
                f"DINOv3 checkpoint path does not exist: {local_path}"
            )

        create_kwargs: dict[str, Any] = {
            "features_only": True,
            "out_indices": out_indices,
            "drop_path_rate": 0.0,
        }
        if local_path is not None and local_path.is_dir():
            model_name = f"local-dir:{local_path}"
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                **create_kwargs,
            )
            self.pretrained_source = str(local_path)
        else:
            if local_path is not None:
                create_kwargs["checkpoint_path"] = str(local_path)
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=bool(pretrained and local_path is None),
                **create_kwargs,
            )
            self.pretrained_source = (
                str(local_path)
                if local_path is not None
                else (
                    f"timm:{self.backbone_name}"
                    if pretrained
                    else "random_initialization"
                )
            )

        feature_channels = list(self.backbone.feature_info.channels())
        if len(feature_channels) != 4:
            raise ValueError(
                "The selected backbone must expose four feature levels, got "
                f"{feature_channels}"
            )
        self.decoder = DifferenceFPNDecoder(
            feature_channels=feature_channels,
            decoder_channels=int(decoder_channels),
        )
        self.backbone.requires_grad_(False)
        for stage_index in range(4 - self.trainable_stages, 4):
            self.backbone.get_submodule(
                f"stages_{stage_index}"
            ).requires_grad_(True)
        if self.freeze_backbone:
            self.backbone.eval()

    def train(self, mode: bool = True) -> DinoV3ConvNeXtChangeDetector:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            features = self.backbone(image)
        return list(features)

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, torch.Tensor | str | list[str] | None] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if image_t1.shape != image_t2.shape:
            raise ValueError(
                f"T1/T2 shapes must match, got {image_t1.shape} "
                f"and {image_t2.shape}"
            )
        features_t1 = self._encode(image_t1)
        features_t2 = self._encode(image_t2)
        differences = [
            torch.abs(left - right)
            for left, right in zip(features_t1, features_t2, strict=True)
        ]
        logits = self.decoder(differences, output_size=image_t1.shape[-2:])
        return {"change_logits": logits, "aux": None}
