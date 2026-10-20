import pytest
import torch

from tools.evaluate_registration_controls import _detector_logits
from tools.summarize_p8_whu_headroom import headroom_decision


def _headroom_rows(valid_pixels: int):
    return [
        {
            "scenario": "clean:s0:common",
            "f1": "0.90",
            "valid_pixels": str(valid_pixels),
        },
        {
            "scenario": "shift:s3:t2",
            "f1": "0.70",
            "valid_pixels": str(valid_pixels),
        },
    ]


def test_whu_headroom_requires_identical_valid_pixel_support() -> None:
    raw = _headroom_rows(100)
    oracle = _headroom_rows(99)
    with pytest.raises(ValueError, match="different valid-pixel support"):
        headroom_decision(raw, oracle, shift8_headroom_min=0.05)


def test_whu_headroom_accepts_paired_support() -> None:
    raw = _headroom_rows(100)
    oracle = _headroom_rows(100)
    oracle[1]["f1"] = "0.80"
    result = headroom_decision(
        raw,
        oracle,
        shift8_headroom_min=0.05,
    )
    assert result["decisive_gate_passed"] is True
    assert result["paired_valid_pixels"]["shift:s3:t2"] == 100


class _ForcedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ordinary_forward_calls = 0

    def forward(self, t1, t2):
        self.ordinary_forward_calls += 1
        return {"change_logits": torch.full_like(t1[:, :1], -1.0)}

    def forward_with_forced_offset(self, t1, t2, offsets):
        return {
            "raw_change_logits": torch.full_like(t1[:, :1], 1.0),
            "aligned_change_logits": torch.full_like(t1[:, :1], 2.0),
        }


def test_registration_control_never_double_runs_registration_model() -> None:
    model = _ForcedModel()
    t1 = torch.zeros(2, 3, 8, 8)
    t2 = torch.zeros_like(t1)
    raw = _detector_logits(model, t1, t2, None)
    aligned = _detector_logits(model, t1, t2, torch.ones(2, 2))
    assert torch.all(raw == 1.0)
    assert torch.all(aligned == 2.0)
    assert model.ordinary_forward_calls == 0
