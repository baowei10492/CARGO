import torch

from rs_cd.correspondence import global_cosine_offset
from rs_cd.robustness import _translate_reflect


def test_global_cosine_offset_recovers_known_translation() -> None:
    generator = torch.Generator().manual_seed(3407)
    reference = torch.randn(8, 24, 24, generator=generator)
    target, _ = _translate_reflect(reference, dx=2, dy=-1)
    valid = torch.ones(1, 24, 24)
    valid[:, :2, :] = 0.0
    valid[:, -2:, :] = 0.0
    valid[:, :, :2] = 0.0
    valid[:, :, -2:] = 0.0
    result = global_cosine_offset(
        reference,
        target,
        radius=3,
        valid_mask=valid,
    )
    assert (result.dx, result.dy) == (2, -1)
    assert result.margin > 0
