from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rs_cd.geometry import warp_reference_from_target

from .dinov3_convnext_cd import DinoV3ConvNeXtChangeDetector


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _DescriptorBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class TaskAwareInputRegistrationEstimator(nn.Module):
    """Estimate task-coordinate translation from high-resolution descriptors.

    The estimator deliberately operates before the DINO backbone. It learns a
    shared stride-2 descriptor, constructs a local correlation surface for
    every spatial location, selects reliable locations, and robustly aggregates
    them into one global offset distribution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        descriptor_channels: int = 32,
        search_radius: int = 4,
        top_fraction: float = 0.25,
        initial_local_logit_scale: float = 10.0,
        initial_global_logit_scale: float = 5.0,
        surface_decoder_channels: int = 16,
        change_exclusion_radius: int = 2,
        decoupled_shift_decision: bool = False,
    ) -> None:
        super().__init__()
        if descriptor_channels < 4:
            raise ValueError("descriptor_channels must be at least 4")
        if search_radius < 1:
            raise ValueError("search_radius must be positive")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if initial_local_logit_scale <= 0:
            raise ValueError("initial_local_logit_scale must be positive")
        if initial_global_logit_scale <= 0:
            raise ValueError("initial_global_logit_scale must be positive")
        if surface_decoder_channels < 0:
            raise ValueError(
                "surface_decoder_channels must be non-negative"
            )
        if change_exclusion_radius < 0:
            raise ValueError("change_exclusion_radius must be non-negative")

        hidden_channels = max(16, descriptor_channels // 2)
        self.descriptor = nn.Sequential(
            _DescriptorBlock(in_channels, hidden_channels, stride=2),
            _DescriptorBlock(
                hidden_channels,
                descriptor_channels,
            ),
            _DescriptorBlock(
                descriptor_channels,
                descriptor_channels,
            ),
        )
        self.reliability = nn.Sequential(
            _DescriptorBlock(
                descriptor_channels * 3,
                hidden_channels,
            ),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.search_radius = int(search_radius)
        self.top_fraction = float(top_fraction)
        self.change_exclusion_radius = int(change_exclusion_radius)
        self.decoupled_shift_decision = bool(
            decoupled_shift_decision
        )
        self.log_local_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(initial_local_logit_scale)))
        )
        self.log_global_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(initial_global_logit_scale)))
        )
        offsets = torch.tensor(
            [
                [dx, dy]
                for dy in range(-self.search_radius, self.search_radius + 1)
                for dx in range(-self.search_radius, self.search_radius + 1)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("offsets", offsets, persistent=False)
        side = 2 * self.search_radius + 1
        decoder_channels = int(surface_decoder_channels)
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
            self.zero_anchor_head: nn.Module | None = nn.Sequential(
                nn.Linear(side * side, decoder_channels),
                nn.GELU(),
                nn.Linear(decoder_channels, 1),
            )
            nn.init.zeros_(self.zero_anchor_head[-1].weight)
            nn.init.zeros_(self.zero_anchor_head[-1].bias)
        else:
            self.surface_refiner = None
            self.zero_anchor_head = None
        shiftedness_input_channels = 2 * side * side + 4
        shiftedness_hidden = max(16, decoder_channels)
        self.shiftedness_head: nn.Module | None
        if self.decoupled_shift_decision:
            self.shiftedness_head = nn.Sequential(
                nn.Linear(
                    shiftedness_input_channels,
                    shiftedness_hidden,
                ),
                nn.GELU(),
                nn.Linear(shiftedness_hidden, 1),
            )
            nn.init.zeros_(self.shiftedness_head[-1].weight)
            nn.init.zeros_(self.shiftedness_head[-1].bias)
            if self.zero_anchor_head is not None:
                self.zero_anchor_head.requires_grad_(False)
        else:
            self.shiftedness_head = None

    def _correlation_volume(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if reference.shape != target.shape or reference.ndim != 4:
            raise ValueError(
                "reference and target descriptors must have matching "
                "BxCxHxW shapes"
            )
        radius = self.search_radius
        height, width = reference.shape[-2:]
        padded = F.pad(
            target,
            (radius, radius, radius, radius),
            mode="reflect",
        )
        correlations = []
        for dx_float, dy_float in self.offsets.tolist():
            dx, dy = int(dx_float), int(dy_float)
            shifted_target = padded[
                :,
                :,
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
            correlations.append(
                (reference * shifted_target).sum(dim=1)
            )
        return torch.stack(correlations, dim=1)

    def forward_descriptors(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        change_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reference = F.normalize(reference.float(), dim=1, eps=1e-6)
        target = F.normalize(target.float(), dim=1, eps=1e-6)
        correlation = self._correlation_volume(reference, target)
        reliability_logits = self.reliability(
            torch.cat(
                (reference, target, torch.abs(reference - target)),
                dim=1,
            )
        ).float()

        local_top_two = torch.topk(correlation, k=2, dim=1).values
        local_margin = local_top_two[:, 0] - local_top_two[:, 1]
        local_scale = self.log_local_logit_scale.exp().clamp(1.0, 100.0)
        selection_logits = (
            reliability_logits[:, 0] + local_scale * local_margin
        )
        if change_mask is not None:
            excluded = F.interpolate(
                change_mask.float(),
                size=selection_logits.shape[-2:],
                mode="nearest",
            )
            if self.change_exclusion_radius > 0:
                radius = self.change_exclusion_radius
                excluded = F.max_pool2d(
                    excluded,
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                )
            selection_logits = selection_logits.masked_fill(
                excluded[:, 0] > 0.5,
                -1e4,
            )

        flat_selection = selection_logits.flatten(1)
        location_count = int(flat_selection.shape[1])
        keep_count = max(
            1,
            min(
                location_count,
                int(round(location_count * self.top_fraction)),
            ),
        )
        selected_logits, selected_indices = torch.topk(
            flat_selection,
            k=keep_count,
            dim=1,
        )
        selected_weights = torch.softmax(selected_logits, dim=1)
        flat_correlation = correlation.flatten(2)
        selected_correlation = torch.gather(
            flat_correlation,
            dim=2,
            index=selected_indices[:, None, :].expand(
                -1,
                flat_correlation.shape[1],
                -1,
            ),
        )
        score_surface = (
            selected_correlation * selected_weights[:, None, :]
        ).sum(dim=2)
        selected_local_winners = selected_correlation.argmax(dim=1)
        vote_histogram = score_surface.new_zeros(score_surface.shape)
        vote_histogram.scatter_add_(
            dim=1,
            index=selected_local_winners,
            src=selected_weights,
        )
        normalized_surface = score_surface - score_surface.mean(
            dim=1,
            keepdim=True,
        )
        normalized_surface = normalized_surface / normalized_surface.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-4)
        global_scale = self.log_global_logit_scale.exp().clamp(1.0, 100.0)
        logits = normalized_surface * global_scale

        side = 2 * self.search_radius + 1
        if self.surface_refiner is not None:
            logits = logits + self.surface_refiner(
                normalized_surface.view(-1, 1, side, side)
            ).flatten(1)
            if (
                not self.decoupled_shift_decision
                and self.zero_anchor_head is None
            ):
                raise RuntimeError("zero_anchor_head was not initialized")
            center = (
                self.search_radius * side + self.search_radius
            )
            center_mask = F.one_hot(
                torch.tensor(center, device=logits.device),
                num_classes=logits.shape[1],
            ).to(logits)
            if not self.decoupled_shift_decision:
                if self.zero_anchor_head is None:
                    raise RuntimeError(
                        "zero_anchor_head was not initialized"
                    )
                logits = logits + self.zero_anchor_head(
                    normalized_surface
                ) * center_mask[None]

        center_index = self.search_radius * side + self.search_radius
        if self.decoupled_shift_decision:
            conditional_logits = logits.clone()
            conditional_logits[:, center_index] = -1e4
            probabilities = torch.softmax(
                conditional_logits,
                dim=1,
            )
            if self.shiftedness_head is None:
                raise RuntimeError(
                    "shiftedness_head was not initialized"
                )
            best_nonzero_surface = normalized_surface.clone()
            best_nonzero_surface[:, center_index] = -1e4
            best_nonzero_score = best_nonzero_surface.max(
                dim=1
            ).values
            center_score = normalized_surface[:, center_index]
            best_nonzero_vote = vote_histogram.clone()
            best_nonzero_vote[:, center_index] = -1.0
            best_nonzero_vote = best_nonzero_vote.max(dim=1).values
            center_vote = vote_histogram[:, center_index]
            shiftedness_features = torch.cat(
                (
                    normalized_surface,
                    vote_histogram,
                    (best_nonzero_score - center_score)[:, None],
                    center_vote[:, None],
                    best_nonzero_vote[:, None],
                    torch.sigmoid(reliability_logits[:, 0])
                    .mean(dim=(-2, -1))[:, None],
                ),
                dim=1,
            )
            shiftedness_logits = self.shiftedness_head(
                shiftedness_features
            )[:, 0]
            shift_probability = torch.sigmoid(shiftedness_logits)
            zero_probability = 1.0 - shift_probability
            output_logits = conditional_logits
        else:
            probabilities = torch.softmax(logits, dim=1)
            zero_probability = probabilities[:, center_index]
            shiftedness_logits = torch.logit(
                (1.0 - zero_probability).clamp(
                    min=1e-6,
                    max=1.0 - 1e-6,
                )
            )
            shift_probability = 1.0 - zero_probability
            output_logits = logits
        offsets = self.offsets.to(probabilities)
        expected_offset = probabilities @ offsets
        hard_indices = probabilities.argmax(dim=1)
        hard_offset = offsets[hard_indices]
        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(float(probabilities.shape[1]))
        global_top_two = torch.topk(
            score_surface,
            k=min(2, score_surface.shape[1]),
            dim=1,
        ).values
        global_margin = (
            global_top_two[:, 0] - global_top_two[:, 1]
            if global_top_two.shape[1] > 1
            else torch.zeros_like(entropy)
        )
        return {
            "offset_feature": expected_offset,
            "hard_offset_feature": hard_offset,
            "offset_probabilities": probabilities,
            "offset_logits": output_logits,
            "zero_probability": zero_probability,
            "shift_probability": shift_probability,
            "shiftedness_logits": shiftedness_logits,
            "confidence": 1.0 - entropy,
            "score_margin": global_margin,
            "score_surface": score_surface,
            "vote_histogram": vote_histogram,
            "mean_local_margin": local_margin.mean(dim=(-2, -1)),
            "mean_reliability": torch.sigmoid(
                reliability_logits[:, 0]
            ).mean(dim=(-2, -1)),
            "selected_fraction": probabilities.new_full(
                (probabilities.shape[0],),
                keep_count / float(location_count),
            ),
            "local_logit_scale": local_scale,
            "global_logit_scale": global_scale,
        }

    def forward(
        self,
        reference_image: torch.Tensor,
        target_image: torch.Tensor,
        change_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if reference_image.shape != target_image.shape:
            raise ValueError("reference and target image shapes must match")
        reference = self.descriptor(reference_image)
        target = self.descriptor(target_image)
        return self.forward_descriptors(
            reference,
            target,
            change_mask=change_mask,
        )


class DinoV3TaskAwareInputRegistrationChangeDetector(
    DinoV3ConvNeXtChangeDetector
):
    """Frozen P3b preceded by task-aware, selectively applied input alignment."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        decoder_channels: int = 128,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        descriptor_channels: int = 32,
        search_radius: int = 4,
        top_fraction: float = 0.25,
        initial_local_logit_scale: float = 10.0,
        initial_global_logit_scale: float = 5.0,
        surface_decoder_channels: int = 16,
        change_exclusion_radius: int = 2,
        offset_decoding: str = "hard",
        alignment_gate: str = "threshold_nonzero",
        zero_probability_threshold: float = 0.5,
        registration_only: bool = True,
        freeze_registration: bool = False,
        target_distribution_sigma: float = 0.5,
        offset_distribution_weight: float = 1.0,
        offset_regression_weight: float = 1.0,
        shiftedness_weight: float = 0.25,
        clean_anchor_weight: float = 2.0,
        decoupled_shift_decision: bool = False,
        freeze_localizer: bool = False,
        use_change_mask_for_matching: bool = True,
        skip_change_detector_during_registration: bool = False,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            trainable_stages=0,
            decoder_channels=decoder_channels,
            out_indices=out_indices,
        )
        self.registration = TaskAwareInputRegistrationEstimator(
            in_channels=3,
            descriptor_channels=int(descriptor_channels),
            search_radius=int(search_radius),
            top_fraction=float(top_fraction),
            initial_local_logit_scale=float(initial_local_logit_scale),
            initial_global_logit_scale=float(initial_global_logit_scale),
            surface_decoder_channels=int(surface_decoder_channels),
            change_exclusion_radius=int(change_exclusion_radius),
            decoupled_shift_decision=bool(
                decoupled_shift_decision
            ),
        )
        self.offset_decoding = str(offset_decoding)
        if self.offset_decoding not in {"soft", "hard"}:
            raise ValueError("offset_decoding must be soft or hard")
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
        self.registration_only = bool(registration_only)
        self.freeze_registration = bool(freeze_registration)
        self.target_distribution_sigma = float(target_distribution_sigma)
        self.offset_distribution_weight = float(
            offset_distribution_weight
        )
        self.offset_regression_weight = float(offset_regression_weight)
        self.shiftedness_weight = float(shiftedness_weight)
        self.clean_anchor_weight = float(clean_anchor_weight)
        self.decoupled_shift_decision = bool(
            decoupled_shift_decision
        )
        self.freeze_localizer = bool(freeze_localizer)
        self.use_change_mask_for_matching = bool(
            use_change_mask_for_matching
        )
        self.skip_change_detector_during_registration = bool(
            skip_change_detector_during_registration
        )
        if self.freeze_localizer and not self.decoupled_shift_decision:
            raise ValueError(
                "freeze_localizer requires decoupled_shift_decision"
            )
        if self.target_distribution_sigma <= 0:
            raise ValueError("target_distribution_sigma must be positive")
        if self.clean_anchor_weight <= 0:
            raise ValueError("clean_anchor_weight must be positive")

        self.backbone.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.freeze_backbone = True
        if self.freeze_registration:
            self.registration.requires_grad_(False)
        elif self.freeze_localizer:
            self.registration.requires_grad_(False)
            if self.registration.shiftedness_head is None:
                raise RuntimeError(
                    "Decoupled shiftedness head was not initialized"
                )
            self.registration.shiftedness_head.requires_grad_(True)
        self.backbone.eval()
        self.decoder.eval()

    def train(
        self,
        mode: bool = True,
    ) -> DinoV3TaskAwareInputRegistrationChangeDetector:
        super().train(mode)
        self.backbone.eval()
        self.decoder.eval()
        if self.freeze_registration:
            self.registration.eval()
        elif self.freeze_localizer:
            self.registration.descriptor.eval()
            self.registration.reliability.eval()
            if self.registration.surface_refiner is not None:
                self.registration.surface_refiner.eval()
            if self.registration.zero_anchor_head is not None:
                self.registration.zero_anchor_head.eval()
            if self.registration.shiftedness_head is not None:
                self.registration.shiftedness_head.train(mode)
        return self

    @staticmethod
    def _target_offset(
        batch: dict[str, Any] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if batch is None:
            return None
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
        outputs: dict[str, torch.Tensor],
        target_pixels: torch.Tensor | None,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        predicted_feature = outputs["offset_feature"]
        zero = predicted_feature.new_zeros(())
        if target_pixels is None:
            return zero, {}
        target_feature = target_pixels / stride
        offsets = self.registration.offsets.to(predicted_feature)
        squared_distance = (
            offsets[None, :, 0] - target_feature[:, None, 0]
        ).square() + (
            offsets[None, :, 1] - target_feature[:, None, 1]
        ).square()
        sigma = self.target_distribution_sigma
        target_distribution = torch.softmax(
            -0.5 * squared_distance / (sigma * sigma),
            dim=1,
        )
        probabilities = outputs["offset_probabilities"].clamp(
            min=1e-8,
            max=1.0 - 1e-8,
        )
        distribution_per_sample = -(
            target_distribution * probabilities.log()
        ).sum(dim=1)
        regression_per_sample = F.smooth_l1_loss(
            predicted_feature,
            target_feature,
            reduction="none",
        ).mean(dim=1)
        shifted_bool = (
            torch.linalg.vector_norm(target_pixels, dim=1) > 0
        )
        shifted = shifted_bool.to(predicted_feature)
        clean = 1.0 - shifted
        shifted_probability = outputs["shift_probability"]
        if self.decoupled_shift_decision:
            side = 2 * self.registration.search_radius + 1
            center_index = (
                self.registration.search_radius * side
                + self.registration.search_radius
            )
            conditional_target = target_distribution.clone()
            conditional_target[:, center_index] = 0.0
            conditional_target = conditional_target / conditional_target.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            conditional_distribution_per_sample = -(
                conditional_target * probabilities.log()
            ).sum(dim=1)
            if bool(shifted_bool.any()):
                distribution_loss = (
                    conditional_distribution_per_sample[shifted_bool].mean()
                )
                regression_loss = regression_per_sample[
                    shifted_bool
                ].mean()
            else:
                distribution_loss = zero
                regression_loss = zero
            shiftedness_loss = F.binary_cross_entropy_with_logits(
                outputs["shiftedness_logits"],
                shifted,
            )
            hard_gate = (
                shifted_probability
                >= 1.0 - self.zero_probability_threshold
            ).to(predicted_feature)
            predicted_pixels = (
                predicted_feature * stride * shifted_probability[:, None]
            )
            hard_predicted_pixels = (
                outputs["hard_offset_feature"]
                * stride
                * hard_gate[:, None]
            )
        else:
            sample_weights = (
                1.0 + clean * (self.clean_anchor_weight - 1.0)
            )
            distribution_loss = (
                distribution_per_sample * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)
            regression_loss = (
                regression_per_sample * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)
            shiftedness_loss = -(
                shifted * shifted_probability.clamp_min(1e-8).log()
                + clean
                * (1.0 - shifted_probability).clamp_min(1e-8).log()
            ).mean()
            predicted_pixels = predicted_feature * stride
            hard_predicted_pixels = (
                outputs["hard_offset_feature"] * stride
            )
        auxiliary_loss = (
            self.offset_distribution_weight * distribution_loss
            + self.offset_regression_weight * regression_loss
            + self.shiftedness_weight * shiftedness_loss
        )

        epe = torch.linalg.vector_norm(
            predicted_pixels - target_pixels,
            dim=1,
        )
        hard_epe = torch.linalg.vector_norm(
            hard_predicted_pixels - target_pixels,
            dim=1,
        )
        clean_count = clean.sum().clamp_min(1.0)
        shifted_count = shifted.sum().clamp_min(1.0)
        metrics = {
            "aux": float(auxiliary_loss.detach()),
            "offset_distribution": float(distribution_loss.detach()),
            "offset_regression": float(regression_loss.detach()),
            "shiftedness_bce": float(shiftedness_loss.detach()),
            "offset_epe_pixels": float(epe.mean().detach()),
            "hard_offset_epe_pixels": float(hard_epe.mean().detach()),
            "hard_clean_within_1px": float(
                (((hard_epe <= 1.0).to(clean) * clean).sum() / clean_count)
                .detach()
            ),
            "hard_shift_within_1px": float(
                (
                    (hard_epe <= 1.0).to(shifted) * shifted
                ).sum().div(shifted_count).detach()
            ),
            "hard_shift_within_2px": float(
                (
                    (hard_epe <= 2.0).to(shifted) * shifted
                ).sum().div(shifted_count).detach()
            ),
            "predicted_shift_probability_clean": float(
                (
                    shifted_probability * clean
                ).sum().div(clean_count).detach()
            ),
            "predicted_shift_probability_shifted": float(
                (
                    shifted_probability * shifted
                ).sum().div(shifted_count).detach()
            ),
            "registration_confidence": float(
                outputs["confidence"].mean().detach()
            ),
            "registration_margin": float(
                outputs["score_margin"].mean().detach()
            ),
            "registration_local_margin": float(
                outputs["mean_local_margin"].mean().detach()
            ),
            "registration_reliability": float(
                outputs["mean_reliability"].mean().detach()
            ),
            "local_logit_scale": float(
                outputs["local_logit_scale"].detach()
            ),
            "global_logit_scale": float(
                outputs["global_logit_scale"].detach()
            ),
        }
        return auxiliary_loss, metrics

    def _applied_offset(
        self,
        outputs: dict[str, torch.Tensor],
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        soft_pixels = outputs["offset_feature"] * stride
        hard_pixels = outputs["hard_offset_feature"] * stride
        chosen = (
            hard_pixels if self.offset_decoding == "hard" else soft_pixels
        )
        zero_probability = outputs["zero_probability"]
        if self.alignment_gate == "soft_nonzero":
            gate = 1.0 - zero_probability
        elif self.alignment_gate == "hard_nonzero":
            if self.decoupled_shift_decision:
                gate = (
                    outputs["shift_probability"] >= 0.5
                ).to(chosen)
            else:
                gate = (
                    torch.linalg.vector_norm(hard_pixels, dim=1) > 0
                ).to(chosen)
        elif self.alignment_gate == "threshold_nonzero":
            gate = (
                zero_probability < self.zero_probability_threshold
            ).to(chosen)
        else:
            gate = torch.ones_like(zero_probability)
        return chosen * gate[:, None], gate

    def _estimate_registration(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        change_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        outputs = self.registration(
            image_t1,
            image_t2,
            change_mask=change_mask,
        )
        descriptor_height = (image_t1.shape[-2] + 1) // 2
        descriptor_width = (image_t1.shape[-1] + 1) // 2
        stride = outputs["offset_feature"].new_tensor(
            [
                image_t1.shape[-1] / descriptor_width,
                image_t1.shape[-2] / descriptor_height,
            ]
        )
        return outputs, stride

    def estimate_registration(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return offsets without executing the expensive frozen P3b path."""

        outputs, stride = self._estimate_registration(
            image_t1,
            image_t2,
        )
        applied_offset, sample_gate = self._applied_offset(
            outputs,
            stride,
        )
        if self.decoupled_shift_decision:
            predicted_soft_pixels = (
                outputs["offset_feature"]
                * stride
                * outputs["shift_probability"][:, None]
            )
            predicted_hard_pixels = (
                outputs["hard_offset_feature"]
                * stride
                * (
                    outputs["shift_probability"]
                    >= 1.0 - self.zero_probability_threshold
                ).to(stride)[:, None]
            )
        else:
            predicted_soft_pixels = outputs["offset_feature"] * stride
            predicted_hard_pixels = (
                outputs["hard_offset_feature"] * stride
            )
        return {
            "predicted_offset_feature": outputs["offset_feature"],
            "predicted_offset_pixels": predicted_soft_pixels,
            "predicted_hard_offset_feature": outputs[
                "hard_offset_feature"
            ],
            "predicted_hard_offset_pixels": predicted_hard_pixels,
            "conditional_offset_pixels": (
                outputs["offset_feature"] * stride
            ),
            "conditional_hard_offset_pixels": (
                outputs["hard_offset_feature"] * stride
            ),
            "registration_confidence": outputs["confidence"],
            "registration_margin": outputs["score_margin"],
            "registration_local_margin": outputs["mean_local_margin"],
            "registration_reliability": outputs["mean_reliability"],
            "registration_zero_probability": outputs[
                "zero_probability"
            ],
            "registration_shift_probability": outputs[
                "shift_probability"
            ],
            "alignment_sample_gate": sample_gate,
            "applied_offset_pixels": applied_offset,
        }

    def forward_detector_only(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Run one frozen P3b detector path without registration.

        This public entry point makes deployment and efficiency controls
        explicit. Calling ``forward`` on this class would also execute the
        learned registration module and would invalidate single-path timing.
        """

        outputs = super().forward(image_t1, image_t2)
        logits = outputs["change_logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Frozen detector did not return Tensor logits")
        return logits

    @torch.no_grad()
    def forward_with_forced_offset(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        offset_pixels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        corrected_t2 = warp_reference_from_target(
            image_t2,
            offset_pixels.to(image_t2),
        )
        raw = super().forward(image_t1, image_t2)["change_logits"]
        aligned = super().forward(image_t1, corrected_t2)[
            "change_logits"
        ]
        return {
            "raw_change_logits": raw,
            "aligned_change_logits": aligned,
            "oracle_offset_pixels": offset_pixels,
        }

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float] | None]:
        if image_t1.shape != image_t2.shape:
            raise ValueError("T1/T2 shapes must match")
        mask_for_matching = (
            targets
            if (
                self.training
                and self.use_change_mask_for_matching
                and targets is not None
            )
            else None
        )
        registration_outputs, stride = self._estimate_registration(
            image_t1,
            image_t2,
            change_mask=mask_for_matching,
        )
        applied_offset, sample_gate = self._applied_offset(
            registration_outputs,
            stride,
        )
        if self.decoupled_shift_decision:
            predicted_pixels = (
                registration_outputs["offset_feature"]
                * stride
                * registration_outputs["shift_probability"][:, None]
            )
            hard_predicted_pixels = (
                registration_outputs["hard_offset_feature"]
                * stride
                * (
                    registration_outputs["shift_probability"]
                    >= 1.0 - self.zero_probability_threshold
                ).to(stride)[:, None]
            )
        else:
            predicted_pixels = (
                registration_outputs["offset_feature"] * stride
            )
            hard_predicted_pixels = (
                registration_outputs["hard_offset_feature"] * stride
            )

        if (
            self.registration_only
            and self.skip_change_detector_during_registration
        ):
            change_logits = image_t1.new_zeros(
                (
                    image_t1.shape[0],
                    1,
                    image_t1.shape[-2],
                    image_t1.shape[-1],
                )
            )
        else:
            if self.registration_only:
                corrected_t2 = image_t2
            else:
                corrected_t2 = warp_reference_from_target(
                    image_t2,
                    applied_offset.to(image_t2),
                )
            with torch.no_grad():
                change_logits = super().forward(
                    image_t1,
                    corrected_t2,
                )["change_logits"]

        target_pixels = self._target_offset(
            batch,
            device=predicted_pixels.device,
            dtype=predicted_pixels.dtype,
        )
        auxiliary_loss, auxiliary_metrics = self._registration_loss(
            registration_outputs,
            target_pixels=target_pixels,
            stride=stride,
        )
        return {
            "change_logits": change_logits,
            "predicted_offset_feature": registration_outputs[
                "offset_feature"
            ],
            "predicted_offset_pixels": predicted_pixels,
            "predicted_hard_offset_feature": registration_outputs[
                "hard_offset_feature"
            ],
            "predicted_hard_offset_pixels": hard_predicted_pixels,
            "registration_confidence": registration_outputs["confidence"],
            "registration_margin": registration_outputs["score_margin"],
            "registration_zero_probability": registration_outputs[
                "zero_probability"
            ],
            "registration_shift_probability": registration_outputs[
                "shift_probability"
            ],
            "alignment_sample_gate": sample_gate,
            "applied_offset_pixels": applied_offset,
            "aux_loss": auxiliary_loss,
            "aux_metrics": auxiliary_metrics,
        }
