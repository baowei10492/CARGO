from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dinov3_convnext_cd import ConvNormAct, DifferenceFPNDecoder


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _shift_feature(
    tensor: torch.Tensor,
    dx: int,
    dy: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output[y, x] = input[y + dy, x + dx] with zero padding."""
    batch, channels, height, width = tensor.shape
    output = torch.zeros_like(tensor)
    valid = torch.zeros(
        batch,
        1,
        height,
        width,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    source_x0 = max(0, dx)
    source_x1 = min(width, width + dx)
    source_y0 = max(0, dy)
    source_y1 = min(height, height + dy)
    target_x0 = max(0, -dx)
    target_x1 = target_x0 + (source_x1 - source_x0)
    target_y0 = max(0, -dy)
    target_y1 = target_y0 + (source_y1 - source_y0)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return output, valid
    output[
        :,
        :,
        target_y0:target_y1,
        target_x0:target_x1,
    ] = tensor[
        :,
        :,
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    valid[:, :, target_y0:target_y1, target_x0:target_x1] = 1.0
    return output, valid


def _binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _binary_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(
        1.0 - mask, kernel_size=kernel, stride=1, padding=radius
    )


class SelectiveCorrespondenceHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        projection_channels: int = 32,
        radius: int = 3,
    ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be positive")
        self.radius = int(radius)
        self.projection_channels = int(projection_channels)
        # Share the descriptor projection across time so swapping T1/T2 does not
        # change the feature space used for matching.
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, projection_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(projection_channels), projection_channels),
        )
        self.null_head = nn.Sequential(
            ConvNormAct(projection_channels * 3, projection_channels, kernel_size=1),
            nn.Conv2d(projection_channels, 1, kernel_size=1),
        )
        self.change_head = nn.Sequential(
            ConvNormAct(projection_channels, projection_channels),
            nn.Conv2d(projection_channels, 1, kernel_size=1),
        )
        correction_channels = projection_channels * 3 + 3
        self.correction_head = nn.Sequential(
            ConvNormAct(correction_channels, projection_channels),
            ConvNormAct(projection_channels, projection_channels),
            nn.Conv2d(projection_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        # The zero-initialized correction layer preserves the pretrained baseline
        # exactly, while alpha=1 still lets the correction layer receive gradients.
        self.alpha = nn.Parameter(torch.ones(1))
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

    def _match(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized_reference = F.normalize(reference.float(), dim=1, eps=1e-6)
        normalized_target = F.normalize(target.float(), dim=1, eps=1e-6)
        similarity_scores = []
        scale = self.logit_scale.clamp(min=1.0, max=40.0)
        for dx_float, dy_float in self.offsets.tolist():
            dx = int(dx_float)
            dy = int(dy_float)
            shifted_target, valid = _shift_feature(normalized_target, dx=dx, dy=dy)
            similarity = (normalized_reference * shifted_target).sum(
                dim=1, keepdim=True
            )
            similarity_scores.append(
                torch.where(valid > 0.5, similarity * scale, similarity.new_full(similarity.shape, -50.0))
            )
        score_volume = torch.cat(similarity_scores, dim=1)
        raw_difference = torch.abs(reference - target)
        null_logit = self.null_head(
            torch.cat((reference, target, raw_difference), dim=1)
        )
        logits = torch.cat((score_volume, null_logit), dim=1)
        probabilities = torch.softmax(logits, dim=1)
        shift_probabilities = probabilities[:, :-1]
        null_probability = probabilities[:, -1:]
        non_null_mass = shift_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
        conditional_probabilities = shift_probabilities / non_null_mass
        entropy = -(
            conditional_probabilities
            * conditional_probabilities.clamp_min(1e-6).log()
        ).sum(dim=1, keepdim=True)
        normalized_entropy = entropy / torch.log(
            entropy.new_tensor(float(self.offsets.shape[0]))
        )
        confidence = (1.0 - null_probability) * (1.0 - normalized_entropy)
        # Accumulate the aligned, unnormalised descriptor one displacement at a
        # time. This avoids materialising BxKxCxHxW (several GiB at batch 128).
        aligned_target = torch.zeros_like(target)
        valid_expectation = torch.zeros_like(null_probability)
        for offset_index, (dx_float, dy_float) in enumerate(
            self.offsets.tolist()
        ):
            shifted_target, valid = _shift_feature(
                target,
                dx=int(dx_float),
                dy=int(dy_float),
            )
            probability = conditional_probabilities[:, offset_index : offset_index + 1]
            aligned_target = aligned_target + probability * shifted_target
            valid_expectation = valid_expectation + probability * valid
        expected_offset = torch.einsum(
            "bkhw,kd->bdhw",
            shift_probabilities,
            self.offsets.to(shift_probabilities),
        ) / non_null_mass.clamp_min(1e-6)
        top_two = torch.topk(score_volume, k=min(2, score_volume.shape[1]), dim=1).values
        score_margin = top_two[:, :1] - top_two[:, 1:2] if top_two.shape[1] > 1 else torch.zeros_like(null_probability)
        return {
            "aligned_target": aligned_target,
            "confidence": confidence,
            "null_logit": null_logit,
            "null_probability": null_probability,
            "expected_offset": expected_offset,
            "valid_expectation": valid_expectation,
            "shift_probabilities": shift_probabilities,
            "score_margin": score_margin,
            "raw_difference": raw_difference,
        }

    def forward(
        self,
        feature_t1: torch.Tensor,
        feature_t2: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        projected_t1 = self.projection(feature_t1)
        projected_t2 = self.projection(feature_t2)
        forward = self._match(projected_t1, projected_t2)
        backward = self._match(projected_t2, projected_t1)
        raw_difference = torch.abs(projected_t1 - projected_t2)
        aligned_difference = 0.5 * (
            torch.abs(projected_t1 - forward["aligned_target"])
            + torch.abs(projected_t2 - backward["aligned_target"])
        )
        coarse_change_logits = self.change_head(raw_difference)
        change_probability = torch.sigmoid(coarse_change_logits)
        confidence = 0.5 * (forward["confidence"] + backward["confidence"])
        null_logits = 0.5 * (forward["null_logit"] + backward["null_logit"])
        null_probability = torch.sigmoid(null_logits)
        align_weight = confidence * (1.0 - change_probability)
        correction_input = torch.cat(
            (
                raw_difference,
                aligned_difference,
                raw_difference - aligned_difference,
                align_weight,
                change_probability,
                null_probability,
            ),
            dim=1,
        )
        correction_logits = self.correction_head(correction_input)
        correction_logits = correction_logits * align_weight
        image_correction_logits = F.interpolate(
            correction_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return {
            "correction_logits": image_correction_logits,
            "feature_correction_logits": correction_logits,
            "align_weight": align_weight,
            "coarse_change_logits": coarse_change_logits,
            "change_probability": change_probability,
            "null_logits": null_logits,
            "null_probability": null_probability,
            "forward_offset": forward["expected_offset"],
            "backward_offset": backward["expected_offset"],
            "forward_shift_probabilities": forward["shift_probabilities"],
            "backward_shift_probabilities": backward["shift_probabilities"],
            "forward_score_margin": forward["score_margin"],
            "backward_score_margin": backward["score_margin"],
        }


class DinoV3SelectiveCorrespondenceChangeDetector(nn.Module):
    """DINOv3 baseline plus a selective local-correspondence residual branch."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny.dinov3_lvd1689m",
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = True,
        trainable_stages: int | None = None,
        decoder_channels: int = 128,
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        correspondence_channels: int = 32,
        correspondence_radius: int = 3,
        freeze_base_model: bool = True,
        change_margin_radius: int = 1,
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
        self.freeze_base_model = bool(freeze_base_model)
        self.change_margin_radius = int(change_margin_radius)
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
        self.correspondence = SelectiveCorrespondenceHead(
            in_channels=feature_channels[0],
            projection_channels=int(correspondence_channels),
            radius=int(correspondence_radius),
        )
        self.backbone.requires_grad_(False)
        for stage_index in range(4 - self.trainable_stages, 4):
            self.backbone.get_submodule(
                f"stages_{stage_index}"
            ).requires_grad_(True)
        if self.freeze_backbone:
            self.backbone.eval()
        if self.freeze_base_model:
            self.backbone.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.backbone.eval()
            self.decoder.eval()

    def train(self, mode: bool = True) -> DinoV3SelectiveCorrespondenceChangeDetector:
        super().train(mode)
        if self.freeze_backbone or self.freeze_base_model:
            self.backbone.eval()
        if self.freeze_base_model:
            self.decoder.eval()
        return self

    def _encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        context = (
            torch.no_grad()
            if (self.freeze_backbone or self.freeze_base_model)
            else nullcontext()
        )
        with context:
            features = self.backbone(image)
        return list(features)

    def _base_logits(
        self,
        features_t1: list[torch.Tensor],
        features_t2: list[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        differences = [
            torch.abs(left - right)
            for left, right in zip(features_t1, features_t2, strict=True)
        ]
        context = torch.no_grad() if self.freeze_base_model else nullcontext()
        with context:
            return self.decoder(differences, output_size=output_size)

    def _auxiliary_losses(
        self,
        batch: dict[str, Any] | None,
        outputs: dict[str, torch.Tensor],
        raw_logits: torch.Tensor,
        final_logits: torch.Tensor,
        targets: torch.Tensor | None,
        feature_size: tuple[int, int],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = raw_logits.new_zeros(())
        if batch is None or targets is None:
            return zero, {}
        if "misregistration_target" not in batch:
            return zero, {}
        batch_size = raw_logits.shape[0]
        device = raw_logits.device
        target_names = list(batch["misregistration_target"])
        shift_dx = torch.as_tensor(batch["misregistration_dx"], device=device).float()
        shift_dy = torch.as_tensor(batch["misregistration_dy"], device=device).float()
        shifted_samples = (
            torch.tensor(
                [
                    1.0
                    if name == "t2" and (abs(float(dx)) > 0.0 or abs(float(dy)) > 0.0)
                    else 0.0
                    for name, dx, dy in zip(target_names, shift_dx.tolist(), shift_dy.tolist(), strict=True)
                ],
                dtype=raw_logits.dtype,
                device=device,
            )
            .view(batch_size, 1, 1, 1)
        )
        if float(shifted_samples.sum()) < 0.5:
            return zero, {}
        feature_height, feature_width = feature_size
        image_height, image_width = image_size
        stride_x = image_width / feature_width
        stride_y = image_height / feature_height
        dilated_targets = _binary_dilate(targets.float(), self.change_margin_radius)
        unchanged_mask = 1.0 - F.adaptive_max_pool2d(
            dilated_targets, output_size=feature_size
        )
        unchanged_mask = unchanged_mask * shifted_samples
        changed_mask = _binary_erode(
            F.interpolate(
                targets.float(),
                size=feature_size,
                mode="nearest",
            ),
            self.change_margin_radius,
        ) * shifted_samples
        valid_mask = batch.get("valid_mask")
        if isinstance(valid_mask, torch.Tensor):
            valid_feature = F.interpolate(
                valid_mask.to(device, non_blocking=True),
                size=feature_size,
                mode="nearest",
            )
            unchanged_mask = unchanged_mask * valid_feature
            changed_mask = changed_mask * valid_feature

        offsets = self.correspondence.offsets.to(device)
        target_dx = shift_dx.view(batch_size, 1, 1, 1) / stride_x
        target_dy = shift_dy.view(batch_size, 1, 1, 1) / stride_y
        offset_dx = offsets[:, 0].view(1, -1, 1, 1)
        offset_dy = offsets[:, 1].view(1, -1, 1, 1)
        squared_distance = (
            (offset_dx - target_dx) ** 2 + (offset_dy - target_dy) ** 2
        )
        target_distribution = torch.softmax(-0.5 * squared_distance, dim=1)
        predicted_distribution = outputs["forward_shift_probabilities"].clamp_min(1e-8)
        correlation_kl = (
            target_distribution
            * (
                target_distribution.clamp_min(1e-8).log()
                - predicted_distribution.log()
            )
        ).sum(dim=1, keepdim=True)
        offset_loss = (
            (correlation_kl * unchanged_mask).sum()
            / unchanged_mask.sum().clamp_min(1.0)
        )

        offset_prediction = outputs["forward_offset"]
        target_offset = torch.cat((target_dx, target_dy), dim=1)
        epe_loss = (
            F.smooth_l1_loss(
                offset_prediction,
                target_offset.expand_as(offset_prediction),
                reduction="none",
            ).sum(dim=1, keepdim=True)
            * unchanged_mask
        ).sum() / unchanged_mask.sum().clamp_min(1.0)

        change_target = F.interpolate(
            targets.float(),
            size=feature_size,
            mode="nearest",
        )
        change_loss = F.binary_cross_entropy_with_logits(
            outputs["coarse_change_logits"],
            change_target,
        )

        null_logits = outputs["null_logits"]
        null_loss = (
            F.binary_cross_entropy_with_logits(
                -null_logits,
                torch.ones_like(null_logits),
                reduction="none",
            )
            * unchanged_mask
        ).sum()
        null_loss += (
            F.binary_cross_entropy_with_logits(
                null_logits,
                torch.ones_like(null_logits),
                reduction="none",
            )
            * changed_mask
        ).sum()
        null_denominator = (unchanged_mask + changed_mask).sum().clamp_min(1.0)
        null_loss = null_loss / null_denominator

        raw_probability = torch.sigmoid(raw_logits).detach()
        final_probability = torch.sigmoid(final_logits)
        preserve_loss = (
            torch.relu(raw_probability - final_probability) * targets.float()
        ).sum() / targets.float().sum().clamp_min(1.0)

        auxiliary_loss = (
            0.10 * offset_loss
            + 0.05 * epe_loss
            + 0.05 * change_loss
            + 0.05 * null_loss
            + 0.10 * preserve_loss
        )
        metrics = {
            "aux": float(auxiliary_loss.detach()),
            "offset_kl": float(offset_loss.detach()),
            "offset_l1": float(epe_loss.detach()),
            "change_aux_bce": float(change_loss.detach()),
            "null_aux_bce": float(null_loss.detach()),
            "preserve": float(preserve_loss.detach()),
            "align_weight_mean": float(outputs["align_weight"].detach().mean()),
        }
        return auxiliary_loss, metrics

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float] | None]:
        if image_t1.shape != image_t2.shape:
            raise ValueError(
                f"T1/T2 shapes must match, got {image_t1.shape} and {image_t2.shape}"
            )
        features_t1 = self._encode(image_t1)
        features_t2 = self._encode(image_t2)
        raw_logits = self._base_logits(
            features_t1,
            features_t2,
            output_size=image_t1.shape[-2:],
        )
        correspondence_outputs = self.correspondence(
            features_t1[0],
            features_t2[0],
            output_size=image_t1.shape[-2:],
        )
        correction_logits = correspondence_outputs["correction_logits"]
        if not isinstance(correction_logits, torch.Tensor):
            raise TypeError("correspondence branch did not return correction logits")
        final_logits = raw_logits + self.correspondence.alpha * correction_logits
        auxiliary_loss, auxiliary_metrics = self._auxiliary_losses(
            batch=batch,
            outputs=correspondence_outputs,
            raw_logits=raw_logits,
            final_logits=final_logits,
            targets=targets,
            feature_size=features_t1[0].shape[-2:],
            image_size=image_t1.shape[-2:],
        )
        return {
            "change_logits": final_logits,
            "raw_change_logits": raw_logits,
            "correction_logits": correction_logits,
            "align_weight": correspondence_outputs["align_weight"],
            "change_probability": correspondence_outputs["change_probability"],
            "null_probability": correspondence_outputs["null_probability"],
            "aux_loss": auxiliary_loss,
            "aux_metrics": auxiliary_metrics,
        }
