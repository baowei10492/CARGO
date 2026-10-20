"""P27 rotation feasibility check: coarse matching helpers and the decision rule."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rs_cd.coarse_matching import correspondences, mutual_nearest, token_centers

ROOT = Path(__file__).resolve().parents[1]


def test_token_centers_follow_the_stride() -> None:
    centers = token_centers(2, 3, 16)
    assert centers.shape == (6, 2)
    assert tuple(centers[0]) == (7.5, 7.5)
    assert tuple(centers[2]) == (39.5, 7.5)
    assert tuple(centers[3]) == (7.5, 23.5)


def test_mutual_nearest_recovers_a_shifted_token_grid() -> None:
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(16, 6, 6, generator=generator)
    target = torch.roll(reference, shifts=(1, 2), dims=(1, 2))  # content moves down 1, right 2
    ref_idx, tgt_idx, scores = mutual_nearest(reference, target)
    assert len(ref_idx) == 36 and np.allclose(scores, 1.0, atol=1e-5)
    centers = token_centers(6, 6, 8)
    source, moved = correspondences(ref_idx, tgt_idx, centers)
    offsets = moved - source
    # Interior tokens move by (+16, +8) px; wrapped ones jump the other way.
    assert (np.isclose(offsets[:, 0], 16) | np.isclose(offsets[:, 0], -32)).all()
    assert (np.isclose(offsets[:, 1], 8) | np.isclose(offsets[:, 1], -40)).all()


def test_mutual_nearest_drops_one_sided_matches() -> None:
    reference = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])     # two tokens: (1, 0), (0, 1)
    target = torch.tensor([[[1.0, 0.9]], [[0.0, 0.1]]])        # both closest to reference token 0
    ref_idx, tgt_idx, _ = mutual_nearest(reference, target)
    assert list(ref_idx) == [0] and list(tgt_idx) == [0]
    with pytest.raises(ValueError):
        mutual_nearest(reference, target[:, :, :1])


def test_rotation_override_keeps_the_sign() -> None:
    from tools.check_p27_rotation_feasibility import rotated

    assert rotated((-12.0, 1.1, 3.0, -4.0), 30) == (-30.0, 1.1, 3.0, -4.0)
    assert rotated((5.0, 0.9, 1.0, 2.0), 20) == (20.0, 0.9, 1.0, 2.0)
    assert rotated((5.0, 0.9, 1.0, 2.0), 0) == (0.0, 0.9, 1.0, 2.0)


def _summary(values: dict[float, float]) -> dict:
    summary = {}
    for name in ("A", "B"):
        for rotation, value in values.items():
            summary[f"{name}|real|s16|r{rotation:g}"] = {"success_8": value}
    return summary


def test_decision_rule() -> None:
    from tools.check_p27_rotation_feasibility import decide

    rule = {"stride": 16, "pair": "real", "success_px": 8, "retention_min": 0.8}
    kept = decide(_summary({0.0: 0.9, 30.0: 0.75}), ["A", "B"], [0.0, 30.0], rule)
    assert kept["rotation_ok"] and kept["retention"] == pytest.approx(0.75 / 0.9)
    dropped = decide(_summary({0.0: 0.9, 30.0: 0.6}), ["A", "B"], [0.0, 30.0], rule)
    assert not dropped["rotation_ok"] and "dropped" in dropped["version1"]
    empty = decide(_summary({0.0: 0.0, 30.0: 0.0}), ["A", "B"], [0.0, 30.0], rule)
    assert not empty["rotation_ok"]


def test_check_config_is_chained_to_p25() -> None:
    config = yaml.safe_load((ROOT / "configs/p27_rotation_check.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "preregistered"
    digest = hashlib.sha256((ROOT / config["p25_config"]).read_bytes()).hexdigest()
    assert digest == config["p25_config_sha256"]
    assert config["rotations_deg"][0] == 0 and config["rotations_deg"][-1] == 30
    assert config["decision"]["stride"] in (8, 16)
