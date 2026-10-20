"""Run the frozen localizer in front of an arbitrary change detector.

The selective-alignment layer is a claim about change detection as a task,
not about our detector, so it has to be testable with detectors we did not
build. The P7 model bundles both roles in one object: it estimates the
offset and it detects change, and the dual-path tools reach into it for
both. This adapter separates them.

    localizer  (frozen P7C)   : (I1, I2)            -> offset
    detector   (any)          : (I1, I2)            -> raw logits
    detector   (same weights) : (I1, warp(I2))      -> aligned logits

The detector only has to accept a bi-temporal pair and return
``change_logits``; nothing else about its architecture is assumed. The
adapter exposes ``estimate_registration`` and ``forward_with_forced_offset``
with exactly the signatures ``tools/train_evaluate_task_benefit_router.py``
already calls, so the router training, threshold calibration, and
evaluation tooling run unchanged.

Two properties are deliberate and are checked by
``tests/test_dual_path_adapter.py``:

* the warp is ``rs_cd.geometry.warp_reference_from_target`` with the
  project's padding and interpolation, called exactly as the P7 model
  calls it, so an adapter wrapping our own detector must reproduce the
  frozen P7C numbers rather than merely approximate them;
* the localizer is held in eval mode with gradients disabled for its whole
  lifetime, so no downstream training loop can silently fine-tune it and
  make the cross-detector comparison unfair.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rs_cd.geometry import warp_reference_from_target


class DualPathAdapter(nn.Module):
    """Compose a frozen offset estimator with an arbitrary detector."""

    def __init__(
        self,
        localizer: nn.Module,
        detector: nn.Module,
        *,
        freeze_localizer: bool = True,
    ) -> None:
        super().__init__()
        estimate = getattr(localizer, "estimate_registration", None)
        if not callable(estimate):
            raise TypeError(
                "localizer must expose estimate_registration(t1, t2); got "
                f"{type(localizer).__name__}"
            )
        self.localizer = localizer
        self.detector = detector
        self.freeze_localizer = bool(freeze_localizer)
        if self.freeze_localizer:
            self.localizer.eval()
            for parameter in self.localizer.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "DualPathAdapter":
        """Keep the localizer in eval mode even when the adapter trains."""

        super().train(mode)
        if self.freeze_localizer:
            self.localizer.eval()
        return self

    def _detect(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.detector(image_t1, image_t2)
        if isinstance(outputs, dict):
            if "change_logits" not in outputs:
                raise KeyError(
                    "detector returned a dict without 'change_logits'; keys="
                    f"{sorted(outputs)}"
                )
            logits = outputs["change_logits"]
        else:
            logits = outputs
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"detector must return a tensor, got {type(logits).__name__}"
            )
        return logits

    @torch.no_grad()
    def estimate_registration(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        was_training = self.localizer.training
        self.localizer.eval()
        try:
            return self.localizer.estimate_registration(image_t1, image_t2)
        finally:
            if was_training and not self.freeze_localizer:
                self.localizer.train()

    @torch.no_grad()
    def forward_with_forced_offset(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        offset_pixels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Raw and aligned logits from one detector and one warp.

        Mirrors DinoV3TaskAwareInputRegistrationChangeDetector so that an
        adapter around our own detector is numerically identical to it.
        """

        if image_t1.shape != image_t2.shape:
            raise ValueError("T1/T2 shapes must match")
        if offset_pixels.shape != (image_t1.shape[0], 2):
            raise ValueError(
                f"offset_pixels must have shape {(image_t1.shape[0], 2)}"
            )
        corrected_t2 = warp_reference_from_target(
            image_t2,
            offset_pixels.to(image_t2),
        )
        raw = self._detect(image_t1, image_t2)
        aligned = self._detect(image_t1, corrected_t2)
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
    ) -> dict[str, Any]:
        """Plain detection on the raw path, for training the detector."""

        outputs = self.detector(image_t1, image_t2, targets, batch) if _accepts_targets(
            self.detector
        ) else self.detector(image_t1, image_t2)
        if isinstance(outputs, dict):
            return outputs
        return {"change_logits": outputs, "aux": None}


def _accepts_targets(module: nn.Module) -> bool:
    import inspect

    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return False
    return len(signature.parameters) >= 4
