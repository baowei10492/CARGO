"""The adapter must be a refactor, not a reimplementation.

P21 compares detectors against each other, so every cell of that matrix has
to be produced by the same dual-path arithmetic that produced the frozen
LEVIR numbers. If the adapter's warp or its call order differed from
DinoV3TaskAwareInputRegistrationChangeDetector even slightly, the new rows
would not be comparable with the existing one and the comparison would be
worthless. These tests pin the equivalence on synthetic modules so they run
without weights or a GPU.
"""
from __future__ import annotations

import torch
from torch import nn

from rs_cd.geometry import warp_reference_from_target
from rs_cd.models.dual_path_adapter import DualPathAdapter


class _SpyDetector(nn.Module):
    """Deterministic, input-sensitive stand-in that records its calls."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.calls.append((image_t1.clone(), image_t2.clone()))
        logits = (image_t1 - image_t2).mean(dim=1, keepdim=True) * self.weight
        return {"change_logits": logits}


class _BareTensorDetector(nn.Module):
    """A detector that returns a tensor rather than a dict."""

    def forward(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> torch.Tensor:
        return (image_t1 - image_t2).mean(dim=1, keepdim=True)


class _StubLocalizer(nn.Module):
    def __init__(self, offset: torch.Tensor) -> None:
        super().__init__()
        self.offset = offset
        self.parameter = nn.Parameter(torch.zeros(1))

    def estimate_registration(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch = image_t1.shape[0]
        return {
            "conditional_hard_offset_pixels": self.offset[:batch],
            "shift_probability": torch.ones(batch),
        }


def _inputs(batch: int = 3, size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3407)
    t1 = torch.rand(batch, 3, size, size, generator=generator)
    t2 = torch.rand(batch, 3, size, size, generator=generator)
    return t1, t2


def test_aligned_path_uses_the_project_warp_exactly() -> None:
    """Reproduces the P7 model's arithmetic: warp T2, then detect twice."""

    t1, t2 = _inputs()
    offset = torch.tensor([[2.0, -3.0], [0.0, 0.0], [-1.5, 4.0]])
    detector = _SpyDetector()
    adapter = DualPathAdapter(_StubLocalizer(offset), detector)

    outputs = adapter.forward_with_forced_offset(t1, t2, offset)

    expected_t2 = warp_reference_from_target(t2, offset)
    expected_raw = (t1 - t2).mean(dim=1, keepdim=True)
    expected_aligned = (t1 - expected_t2).mean(dim=1, keepdim=True)
    torch.testing.assert_close(outputs["raw_change_logits"], expected_raw)
    torch.testing.assert_close(outputs["aligned_change_logits"], expected_aligned)

    # Same detector object on both paths: weights are shared by construction,
    # which is what makes the raw/aligned contrast attributable to the warp.
    assert len(detector.calls) == 2
    torch.testing.assert_close(detector.calls[0][1], t2)
    torch.testing.assert_close(detector.calls[1][1], expected_t2)


def test_zero_offset_leaves_the_aligned_path_bit_identical() -> None:
    """Clean pairs must not pay grid-sampling round-off."""

    t1, t2 = _inputs()
    zero = torch.zeros(t1.shape[0], 2)
    adapter = DualPathAdapter(_StubLocalizer(zero), _SpyDetector())
    outputs = adapter.forward_with_forced_offset(t1, t2, zero)
    torch.testing.assert_close(
        outputs["raw_change_logits"],
        outputs["aligned_change_logits"],
        rtol=0.0,
        atol=0.0,
    )


def test_detector_returning_a_bare_tensor_is_accepted() -> None:
    """Ported detectors should not all need a dict wrapper."""

    t1, t2 = _inputs()
    offset = torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, -2.0]])
    adapter = DualPathAdapter(_StubLocalizer(offset), _BareTensorDetector())
    outputs = adapter.forward_with_forced_offset(t1, t2, offset)
    assert outputs["raw_change_logits"].shape == (3, 1, 16, 16)


def test_localizer_stays_frozen_and_in_eval() -> None:
    """A training loop must not be able to fine-tune the shared localizer."""

    offset = torch.zeros(3, 2)
    localizer = _StubLocalizer(offset)
    adapter = DualPathAdapter(localizer, _SpyDetector())

    assert not any(p.requires_grad for p in localizer.parameters())
    adapter.train()
    assert not localizer.training, "localizer must ignore adapter.train()"
    assert adapter.detector.training, "detector must still follow train()"


def test_offset_shape_is_validated() -> None:
    t1, t2 = _inputs()
    adapter = DualPathAdapter(_StubLocalizer(torch.zeros(3, 2)), _SpyDetector())
    try:
        adapter.forward_with_forced_offset(t1, t2, torch.zeros(3, 3))
    except ValueError as error:
        assert "offset_pixels must have shape" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError for a malformed offset")


def test_localizer_without_estimate_registration_is_rejected() -> None:
    try:
        DualPathAdapter(nn.Identity(), _SpyDetector())
    except TypeError as error:
        assert "estimate_registration" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a TypeError for an invalid localizer")


def test_detector_dict_without_change_logits_is_rejected() -> None:
    class _WrongKey(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"logits": (a - b).mean(dim=1, keepdim=True)}

    t1, t2 = _inputs()
    adapter = DualPathAdapter(_StubLocalizer(torch.zeros(3, 2)), _WrongKey())
    try:
        adapter.forward_with_forced_offset(t1, t2, torch.zeros(3, 2))
    except KeyError as error:
        assert "change_logits" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a KeyError for a missing logits key")
