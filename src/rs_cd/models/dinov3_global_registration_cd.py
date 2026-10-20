from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rs_cd.geometry import scale_input_offset, warp_reference_from_target

from .dinov3_convnext_cd import DinoV3ConvNeXtChangeDetector


def warp_feature(
    feature: torch.Tensor,
    offset: torch.Tensor,
    padding_mode: str = "reflection",
) -> torch.Tensor:
    """Sample ``feature[y + dy, x + dx]`` for each batch item.

    ``offset`` is expressed in the input feature map's pixel units and has
    shape ``Bx2`` with columns ``dx, dy``.
    """

    return warp_reference_from_target(
        feature,
        offset,
        padding_mode=padding_mode,
    )


class GlobalTranslationEstimator(nn.Module):
    """Differentiable global translation from shared DINO descriptors."""

    def __init__(
        self,
        channels: int,
        radius: int = 3,
        initial_logit_scale: float = 200.0,
        surface_decoder_channels: int = 0,
        normalize_score_surface: bool = False,
    ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be positive")
        if initial_logit_scale <= 0:
            raise ValueError("initial_logit_scale must be positive")
        self.radius = int(radius)
        self.normalize_score_surface = bool(normalize_score_surface)
        # Positive per-channel reliability weights preserve the original DINO
        # descriptor space and avoid the random projection failure seen in P5b.
        self.channel_logits = nn.Parameter(torch.zeros(channels))
        self.log_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(initial_logit_scale)))
        )
        decoder_channels = int(surface_decoder_channels)
        if decoder_channels < 0:
            raise ValueError("surface_decoder_channels must be non-negative")
        if decoder_channels > 0:
            self.surface_refiner: nn.Module | None = nn.Sequential(
                nn.Conv2d(1, decoder_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(decoder_channels, 1, kernel_size=1),
            )
            nn.init.zeros_(self.surface_refiner[-1].weight)
            nn.init.zeros_(self.surface_refiner[-1].bias)
            candidate_count = (2 * radius + 1) ** 2
            self.zero_offset_head: nn.Module | None = nn.Sequential(
                nn.Linear(candidate_count, decoder_channels),
                nn.GELU(),
                nn.Linear(decoder_channels, 1),
            )
            nn.init.zeros_(self.zero_offset_head[-1].weight)
            nn.init.zeros_(self.zero_offset_head[-1].bias)
        else:
            self.surface_refiner = None
            self.zero_offset_head = None
        self.register_buffer(
            "offsets",
            torch.tensor(
                [
                    [dx, dy]
                    for dy in range(-radius, radius + 1)
                    for dx in range(-radius, radius + 1)
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _overlap_slices(
        height: int,
        width: int,
        dx: int,
        dy: int,
    ) -> tuple[slice, slice, slice, slice]:
        reference_y0 = max(0, -dy)
        reference_y1 = min(height, height - dy)
        reference_x0 = max(0, -dx)
        reference_x1 = min(width, width - dx)
        return (
            slice(reference_y0, reference_y1),
            slice(reference_x0, reference_x1),
            slice(reference_y0 + dy, reference_y1 + dy),
            slice(reference_x0 + dx, reference_x1 + dx),
        )

    def forward(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if reference.shape != target.shape or reference.ndim != 4:
            raise ValueError(
                "reference and target must have identical BxCxHxW shapes"
            )
        _, channels, height, width = reference.shape
        if channels != self.channel_logits.numel():
            raise ValueError(
                f"Expected {self.channel_logits.numel()} channels, got "
                f"{channels}"
            )
        weights = F.softplus(self.channel_logits.float()) + 1e-4
        weights = weights / weights.mean()
        descriptor_scale = weights.sqrt().view(1, channels, 1, 1)
        normalized_reference = F.normalize(
            reference.float() * descriptor_scale,
            dim=1,
            eps=1e-6,
        )
        normalized_target = F.normalize(
            target.float() * descriptor_scale,
            dim=1,
            eps=1e-6,
        )
        scores = []
        for dx_float, dy_float in self.offsets.tolist():
            dx, dy = int(dx_float), int(dy_float)
            ref_y, ref_x, target_y, target_x = self._overlap_slices(
                height,
                width,
                dx,
                dy,
            )
            similarity = (
                normalized_reference[:, :, ref_y, ref_x]
                * normalized_target[:, :, target_y, target_x]
            ).sum(dim=1)
            scores.append(similarity.mean(dim=(-2, -1)))
        score_volume = torch.stack(scores, dim=1)
        score_input = score_volume
        if self.normalize_score_surface:
            score_input = score_input - score_input.mean(dim=1, keepdim=True)
            score_input = score_input / score_input.std(
                dim=1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-4)
        logit_scale = self.log_logit_scale.exp().clamp(1.0, 1000.0)
        logits = score_input * logit_scale
        if self.surface_refiner is not None:
            side = 2 * self.radius + 1
            surface = score_input.view(-1, 1, side, side)
            logits = logits + self.surface_refiner(surface).flatten(1)
            if self.zero_offset_head is None:
                raise RuntimeError("zero-offset head was not initialized")
            center_index = self.radius * side + self.radius
            center_adjustment = self.zero_offset_head(score_input)
            center_mask = F.one_hot(
                torch.tensor(
                    center_index,
                    device=logits.device,
                ),
                num_classes=logits.shape[1],
            ).to(logits)
            logits = logits + center_adjustment * center_mask[None]
        probabilities = torch.softmax(logits, dim=1)
        offsets = self.offsets.to(probabilities)
        expected_offset = probabilities @ offsets
        hard_indices = probabilities.argmax(dim=1)
        hard_offset = offsets[hard_indices]
        center_index = self.radius * (2 * self.radius + 1) + self.radius
        zero_probability = probabilities[:, center_index]
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1)
        entropy = entropy / math.log(float(self.offsets.shape[0]))
        confidence = 1.0 - entropy
        top_two = torch.topk(
            score_volume,
            k=min(2, score_volume.shape[1]),
            dim=1,
        ).values
        margin = (
            top_two[:, 0] - top_two[:, 1]
            if top_two.shape[1] > 1
            else torch.zeros_like(confidence)
        )
        return {
            "offset_feature": expected_offset,
            "hard_offset_feature": hard_offset,
            "zero_probability": zero_probability,
            "offset_probabilities": probabilities,
            "confidence": confidence,
            "score_margin": margin,
            "score_volume": score_volume,
            "logit_scale": logit_scale,
        }


class DinoV3GlobalRegistrationChangeDetector(
    DinoV3ConvNeXtChangeDetector
):
    """P3b with explicit global registration before multi-scale differencing."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = True,
        trainable_stages: int | None = None,
        decoder_channels: int = 128,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        search_radius: int = 3,
        initial_logit_scale: float = 200.0,
        surface_decoder_channels: int = 0,
        normalize_score_surface: bool = False,
        offset_decoding: str = "soft",
        alignment_gate: str = "none",
        zero_probability_threshold: float = 0.25,
        alignment_parameterization: str = "direct",
        alignment_logit_bias: float = -3.0,
        registration_only: bool = True,
        freeze_registration: bool = False,
        use_confidence_gate: bool = False,
        target_distribution_sigma: float = 0.5,
        offset_distribution_weight: float = 1.0,
        offset_regression_weight: float = 1.0,
        fusion_regularization_weight: float = 0.0,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            trainable_stages=trainable_stages,
            decoder_channels=decoder_channels,
            out_indices=out_indices,
        )
        feature_channels = list(self.backbone.feature_info.channels())
        self.registration = GlobalTranslationEstimator(
            channels=feature_channels[0],
            radius=int(search_radius),
            initial_logit_scale=float(initial_logit_scale),
            surface_decoder_channels=int(surface_decoder_channels),
            normalize_score_surface=bool(normalize_score_surface),
        )
        self.offset_decoding = str(offset_decoding)
        if self.offset_decoding not in {"soft", "hard"}:
            raise ValueError("offset_decoding must be 'soft' or 'hard'")
        self.alignment_gate = str(alignment_gate)
        if self.alignment_gate not in {
            "none",
            "soft_nonzero",
            "hard_nonzero",
            "threshold_nonzero",
        }:
            raise ValueError(
                "alignment_gate must be none, soft_nonzero, hard_nonzero, "
                "or threshold_nonzero"
            )
        self.zero_probability_threshold = float(
            zero_probability_threshold
        )
        if not 0.0 <= self.zero_probability_threshold <= 1.0:
            raise ValueError(
                "zero_probability_threshold must be in [0, 1]"
            )
        self.alignment_parameterization = str(alignment_parameterization)
        if self.alignment_parameterization not in {"direct", "sigmoid"}:
            raise ValueError(
                "alignment_parameterization must be direct or sigmoid"
            )
        self.alignment_logit_bias = float(alignment_logit_bias)
        self.alignment_strengths = nn.Parameter(torch.zeros(4))
        self.registration_only = bool(registration_only)
        self.freeze_registration = bool(freeze_registration)
        self.use_confidence_gate = bool(use_confidence_gate)
        self.target_distribution_sigma = float(target_distribution_sigma)
        self.offset_distribution_weight = float(
            offset_distribution_weight
        )
        self.offset_regression_weight = float(offset_regression_weight)
        self.fusion_regularization_weight = float(
            fusion_regularization_weight
        )
        if self.target_distribution_sigma <= 0:
            raise ValueError("target_distribution_sigma must be positive")

        # P5c always preserves the P3b feature extractor and decoder. Stage A
        # learns registration; Stage B learns only four fusion strengths.
        self.backbone.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.freeze_backbone = True
        self.alignment_strengths.requires_grad_(not self.registration_only)
        if self.freeze_registration:
            self.registration.requires_grad_(False)
        self.backbone.eval()
        self.decoder.eval()

    def train(
        self,
        mode: bool = True,
    ) -> DinoV3GlobalRegistrationChangeDetector:
        super().train(mode)
        self.backbone.eval()
        self.decoder.eval()
        if self.freeze_registration:
            self.registration.eval()
        return self

    def _target_offset(
        self,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        required = {
            "misregistration_target",
            "misregistration_dx",
            "misregistration_dy",
        }
        if not required.issubset(batch):
            return None
        names = list(batch["misregistration_target"])
        dx = torch.as_tensor(
            batch["misregistration_dx"],
            device=device,
            dtype=dtype,
        )
        dy = torch.as_tensor(
            batch["misregistration_dy"],
            device=device,
            dtype=dtype,
        )
        direction = torch.tensor(
            [
                1.0 if name == "t2" else -1.0 if name == "t1" else 0.0
                for name in names
            ],
            device=device,
            dtype=dtype,
        )
        return torch.stack((dx * direction, dy * direction), dim=1)

    def _registration_loss(
        self,
        registration_outputs: dict[str, torch.Tensor],
        batch: dict[str, Any] | None,
        image_size: tuple[int, int],
        feature_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        predicted_feature = registration_outputs["offset_feature"]
        zero = predicted_feature.new_zeros(())
        if batch is None:
            return zero, {}
        target_pixels = self._target_offset(
            batch,
            device=predicted_feature.device,
            dtype=predicted_feature.dtype,
        )
        if target_pixels is None:
            return zero, {}
        image_height, image_width = image_size
        feature_height, feature_width = feature_size
        stride = predicted_feature.new_tensor(
            [image_width / feature_width, image_height / feature_height]
        )
        target_feature = target_pixels / stride

        offsets = self.registration.offsets.to(predicted_feature)
        squared_distance = (
            offsets[None, :, 0] - target_feature[:, None, 0]
        ) ** 2 + (
            offsets[None, :, 1] - target_feature[:, None, 1]
        ) ** 2
        sigma = self.target_distribution_sigma
        target_distribution = torch.softmax(
            -0.5 * squared_distance / (sigma * sigma),
            dim=1,
        )
        probabilities = registration_outputs[
            "offset_probabilities"
        ].clamp_min(1e-8)
        distribution_loss = -(
            target_distribution * probabilities.log()
        ).sum(dim=1).mean()
        regression_loss = F.smooth_l1_loss(
            predicted_feature,
            target_feature,
        )
        predicted_pixels = predicted_feature * stride
        hard_predicted_pixels = (
            registration_outputs["hard_offset_feature"] * stride
        )
        epe_pixels = torch.linalg.vector_norm(
            predicted_pixels - target_pixels,
            dim=1,
        )
        hard_epe_pixels = torch.linalg.vector_norm(
            hard_predicted_pixels - target_pixels,
            dim=1,
        )
        auxiliary_loss = (
            self.offset_distribution_weight * distribution_loss
            + self.offset_regression_weight * regression_loss
        )
        strengths = self._alignment_strength_values()
        fusion_regularization = strengths.square().mean()
        auxiliary_loss = (
            auxiliary_loss
            + self.fusion_regularization_weight * fusion_regularization
        )
        metrics = {
            "aux": float(auxiliary_loss.detach()),
            "offset_distribution": float(distribution_loss.detach()),
            "offset_regression": float(regression_loss.detach()),
            "offset_epe_pixels": float(epe_pixels.mean().detach()),
            "hard_offset_epe_pixels": float(
                hard_epe_pixels.mean().detach()
            ),
            "offset_within_2px": float(
                (epe_pixels <= 2.0).float().mean().detach()
            ),
            "offset_within_4px": float(
                (epe_pixels <= 4.0).float().mean().detach()
            ),
            "registration_confidence": float(
                registration_outputs["confidence"].mean().detach()
            ),
            "registration_margin": float(
                registration_outputs["score_margin"].mean().detach()
            ),
            "registration_logit_scale": float(
                registration_outputs["logit_scale"].detach()
            ),
            "alignment_strength_mean": float(strengths.mean().detach()),
        }
        return auxiliary_loss, metrics

    def _alignment_strength_values(self) -> torch.Tensor:
        if self.alignment_parameterization == "sigmoid":
            return torch.sigmoid(
                self.alignment_strengths + self.alignment_logit_bias
            )
        return self.alignment_strengths.clamp(0.0, 1.0)

    @torch.no_grad()
    def forward_with_forced_offset(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        offset_pixels: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[tuple[float, float]]]:
        """Audit the feature-alignment path with a known input-pixel offset.

        This deliberately bypasses the learned registration estimator. It
        returns three causal variants from the same encoded features:

        * ``raw_change_logits``: the unchanged P3b path;
        * ``aligned_change_logits``: fully aligned differences;
        * ``blended_change_logits``: checkpoint alignment strengths with an
          oracle non-zero gate and confidence one.
        """

        if image_t1.shape != image_t2.shape:
            raise ValueError("T1/T2 shapes must match")
        if offset_pixels.shape != (image_t1.shape[0], 2):
            raise ValueError(
                f"offset_pixels must have shape {(image_t1.shape[0], 2)}"
            )
        features_t1 = self._encode(image_t1)
        features_t2 = self._encode(image_t2)
        image_size = image_t1.shape[-2:]
        raw_differences: list[torch.Tensor] = []
        aligned_differences: list[torch.Tensor] = []
        blended_differences: list[torch.Tensor] = []
        feature_strides: list[tuple[float, float]] = []
        strengths = self._alignment_strength_values()
        nonzero_gate = (
            torch.linalg.vector_norm(offset_pixels.float(), dim=1) > 0
        ).to(image_t1)
        if self.alignment_gate == "none":
            nonzero_gate = torch.ones_like(nonzero_gate)

        for level, (feature_t1, feature_t2) in enumerate(
            zip(features_t1, features_t2, strict=True)
        ):
            feature_size = feature_t1.shape[-2:]
            level_offset = scale_input_offset(
                offset_pixels.to(feature_t2),
                image_size=image_size,
                feature_size=feature_size,
            )
            aligned_t2 = warp_feature(feature_t2, level_offset)
            raw_difference = torch.abs(feature_t1 - feature_t2)
            aligned_difference = torch.abs(feature_t1 - aligned_t2)
            blend = strengths[level] * nonzero_gate[:, None, None, None]
            raw_differences.append(raw_difference)
            aligned_differences.append(aligned_difference)
            blended_differences.append(
                raw_difference
                + blend * (aligned_difference - raw_difference)
            )
            feature_strides.append(
                (
                    image_size[1] / feature_size[1],
                    image_size[0] / feature_size[0],
                )
            )

        return {
            "raw_change_logits": self.decoder(
                raw_differences,
                output_size=image_size,
            ),
            "aligned_change_logits": self.decoder(
                aligned_differences,
                output_size=image_size,
            ),
            "blended_change_logits": self.decoder(
                blended_differences,
                output_size=image_size,
            ),
            "oracle_offset_pixels": offset_pixels,
            "oracle_sample_gate": nonzero_gate,
            "alignment_strengths": strengths,
            "feature_strides_xy": feature_strides,
        }

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float] | None]:
        del targets
        if image_t1.shape != image_t2.shape:
            raise ValueError(
                f"T1/T2 shapes must match, got {image_t1.shape} and "
                f"{image_t2.shape}"
            )
        features_t1 = self._encode(image_t1)
        features_t2 = self._encode(image_t2)
        registration_outputs = self.registration(
            features_t1[0],
            features_t2[0],
        )
        offset_level0 = registration_outputs["offset_feature"]
        hard_offset_level0 = registration_outputs["hard_offset_feature"]
        warp_offset_level0 = (
            hard_offset_level0
            if self.offset_decoding == "hard"
            else offset_level0
        )
        feature0_height, feature0_width = features_t1[0].shape[-2:]
        raw_differences = []
        fused_differences = []
        strengths = self._alignment_strength_values()
        confidence = registration_outputs["confidence"]
        zero_probability = registration_outputs["zero_probability"]
        if self.alignment_gate == "soft_nonzero":
            sample_gate = 1.0 - zero_probability
        elif self.alignment_gate == "hard_nonzero":
            sample_gate = (
                torch.linalg.vector_norm(hard_offset_level0, dim=1) > 0
            ).to(offset_level0)
        elif self.alignment_gate == "threshold_nonzero":
            sample_gate = (
                zero_probability < self.zero_probability_threshold
            ).to(offset_level0)
        else:
            sample_gate = torch.ones_like(zero_probability)
        for level, (feature_t1, feature_t2) in enumerate(
            zip(features_t1, features_t2, strict=True)
        ):
            height, width = feature_t1.shape[-2:]
            scale = warp_offset_level0.new_tensor(
                [width / feature0_width, height / feature0_height]
            )
            level_offset = warp_offset_level0 * scale
            aligned_t2 = warp_feature(feature_t2, level_offset)
            raw_difference = torch.abs(feature_t1 - feature_t2)
            aligned_difference = torch.abs(feature_t1 - aligned_t2)
            raw_differences.append(raw_difference)
            strength = strengths[level]
            if self.use_confidence_gate:
                blend = (
                    strength
                    * sample_gate[:, None, None, None]
                    * confidence[:, None, None, None]
                )
            else:
                blend = strength * sample_gate[:, None, None, None]
            fused_differences.append(
                raw_difference + blend * (aligned_difference - raw_difference)
            )

        raw_logits = self.decoder(
            raw_differences,
            output_size=image_t1.shape[-2:],
        )
        if self.registration_only:
            final_logits = raw_logits
        else:
            final_logits = self.decoder(
                fused_differences,
                output_size=image_t1.shape[-2:],
            )
        stride = offset_level0.new_tensor(
            [
                image_t1.shape[-1] / feature0_width,
                image_t1.shape[-2] / feature0_height,
            ]
        )
        auxiliary_loss, auxiliary_metrics = self._registration_loss(
            registration_outputs=registration_outputs,
            batch=batch,
            image_size=image_t1.shape[-2:],
            feature_size=features_t1[0].shape[-2:],
        )
        return {
            "change_logits": final_logits,
            "raw_change_logits": raw_logits,
            "predicted_offset_feature": offset_level0,
            "predicted_offset_pixels": offset_level0 * stride,
            "predicted_hard_offset_feature": hard_offset_level0,
            "predicted_hard_offset_pixels": hard_offset_level0 * stride,
            "registration_confidence": confidence,
            "registration_margin": registration_outputs["score_margin"],
            "registration_zero_probability": zero_probability,
            "alignment_sample_gate": sample_gate,
            "alignment_strengths": strengths,
            "aux_loss": auxiliary_loss,
            "aux_metrics": auxiliary_metrics,
        }
