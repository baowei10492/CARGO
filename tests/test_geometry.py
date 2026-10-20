import torch

from rs_cd.geometry import (
    interior_valid_mask,
    scale_input_offset,
    warp_reference_from_target,
    warp_sampling_valid_mask,
)
from rs_cd.registration_controls import phase_correlation_offset
from rs_cd.robustness import _translate_reflect


def test_identity_warp_is_numerically_exact_in_fp32() -> None:
    generator = torch.Generator().manual_seed(3407)
    tensor = torch.randn(2, 3, 17, 19, generator=generator)
    offset = torch.zeros(2, 2)
    warped = warp_reference_from_target(tensor, offset)
    assert float((warped - tensor).abs().max()) <= 1e-6


def test_positive_offset_recovers_target_shifted_right_and_down() -> None:
    generator = torch.Generator().manual_seed(3407)
    reference = torch.randn(3, 31, 35, generator=generator)
    dx, dy = 4, 3
    target, _ = _translate_reflect(reference, dx=dx, dy=dy)
    recovered = warp_reference_from_target(
        target.unsqueeze(0),
        torch.tensor([[dx, dy]], dtype=torch.float32),
    )
    valid = interior_valid_mask(1, 31, 35, border=max(dx, dy)).bool()
    valid = valid.expand_as(recovered)
    torch.testing.assert_close(
        recovered[valid],
        reference.unsqueeze(0)[valid],
        atol=2e-5,
        rtol=0.0,
    )


def test_sampling_valid_mask_uses_reference_to_target_convention() -> None:
    offset = torch.tensor([[2.0, -1.0]])
    valid = warp_sampling_valid_mask(offset, height=7, width=9)
    assert valid.shape == (1, 1, 7, 9)
    assert int(valid.sum()) == (7 - 1) * (9 - 2)
    assert bool((valid[0, 0, 0, :] == 0).all())
    assert bool((valid[0, 0, :, -2:] == 0).all())


def test_input_pixel_offset_scales_per_feature_level() -> None:
    offset = torch.tensor([[8.0, -4.0]])
    level0 = scale_input_offset(offset, (256, 256), (64, 64))
    level3 = scale_input_offset(offset, (256, 256), (8, 8))
    torch.testing.assert_close(level0, torch.tensor([[2.0, -1.0]]))
    torch.testing.assert_close(level3, torch.tensor([[0.25, -0.125]]))


def test_phase_correlation_recovers_known_reflect_translation() -> None:
    generator = torch.Generator().manual_seed(3407)
    reference = torch.rand(3, 96, 96, generator=generator)
    for dx, dy in ((2, 0), (-4, 3), (8, -8)):
        target, _ = _translate_reflect(reference, dx=dx, dy=dy)
        result = phase_correlation_offset(
            reference,
            target,
            normalization="unit",
            max_abs_shift=12,
        )
        assert (result.dx, result.dy) == (float(dx), float(dy))
