from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        widths = [base_channels * (2**index) for index in range(4)]
        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, widths[0]),
                ConvBlock(widths[0], widths[1]),
                ConvBlock(widths[1], widths[2]),
                ConvBlock(widths[2], widths[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if index < len(self.blocks) - 1:
                x = self.pool(x)
        return features


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.decode = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.decode(torch.cat([x, skip], dim=1))


class FCSiamDiff(nn.Module):
    """Compact shared-weight FC-Siam-diff baseline.

    The output follows the project contract:
    ``{"change_logits": Tensor[B,1,H,W], "aux": None}``.
    Because all temporal fusion uses absolute differences and weights are
    shared, swapping T1/T2 is exactly invariant up to floating-point effects.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels * (2**index) for index in range(4)]
        self.encoder = SharedEncoder(in_channels, base_channels)
        self.up3 = UpBlock(widths[3], widths[2], widths[2])
        self.up2 = UpBlock(widths[2], widths[1], widths[1])
        self.up1 = UpBlock(widths[1], widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, torch.Tensor | str | list[str] | None] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if image_t1.shape != image_t2.shape:
            raise ValueError(
                f"T1/T2 shapes must match, got {image_t1.shape} and {image_t2.shape}"
            )
        features_t1 = self.encoder(image_t1)
        features_t2 = self.encoder(image_t2)
        differences = [
            torch.abs(left - right)
            for left, right in zip(features_t1, features_t2, strict=True)
        ]
        x = self.up3(differences[3], differences[2])
        x = self.up2(x, differences[1])
        x = self.up1(x, differences[0])
        return {"change_logits": self.head(x), "aux": None}
