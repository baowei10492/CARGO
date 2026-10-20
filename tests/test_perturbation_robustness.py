import pytest
import torch

from rs_cd.registration_controls import phase_correlation_offset
from rs_cd.robustness import (
    CorruptedPairDataset,
    CorruptionSpec,
    _rotate,
    _translate_reflect,
    _translate_subpixel,
    _unit_direction,
)
from tools.evaluate_perturbation_robustness import (
    perturbation_specs,
    safety_decision,
)


def _textured(size=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(3, size, size, generator=generator)
    grid = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    for cx, cy in ((-0.4, -0.3), (0.35, 0.1), (0.0, 0.45)):
        base += torch.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02)
        )[None]
    return base.clamp(0.0, 1.0)


def test_subpixel_translation_matches_integer_path() -> None:
    # A whole-pixel sub-pixel warp must reproduce the integer slicing
    # path in the interior; a mismatch means the grid normalization
    # convention is wrong and every sub-pixel number would carry a
    # hidden interpolation penalty.
    image = _textured(size=64, seed=1)
    for dx, dy in ((3, -2), (8, 8), (1, 0)):
        bilinear, _ = _translate_subpixel(image, float(dx), float(dy))
        integer, _ = _translate_reflect(image, dx, dy)
        interior = (
            bilinear[:, 12:-12, 12:-12] - integer[:, 12:-12, 12:-12]
        )
        assert float(interior.abs().max()) < 1e-4


def test_subpixel_direction_is_unit_length() -> None:
    # Severities are magnitudes; a diagonal direction of length sqrt(2)
    # would silently scale them by 41%.
    for seed in range(8):
        dx, dy = _unit_direction(seed)
        assert (dx * dx + dy * dy) ** 0.5 == pytest.approx(1.0)


def test_zero_rotation_is_identity() -> None:
    image = _textured(size=64, seed=2)
    rotated, valid = _rotate(image, 0.0)
    assert float((rotated - image).abs().max()) < 1e-4
    assert float(valid.mean()) == pytest.approx(1.0)


def test_rotation_moves_corners_more_than_centre() -> None:
    image = _textured(size=128, seed=3)
    rotated, _ = _rotate(image, 1.0)
    difference = (rotated - image).abs().mean(dim=0)
    centre = float(difference[56:72, 56:72].mean())
    corner = float(difference[4:20, 4:20].mean())
    assert corner > centre


def test_phase_subpixel_refinement_beats_integer_peak() -> None:
    base = _textured(size=128, seed=4)
    for true_dx in (0.5, 1.5):
        shifted, _ = _translate_subpixel(base, true_dx, 0.0)
        integer = phase_correlation_offset(
            base,
            shifted,
            normalization="unit",
            max_abs_shift=8,
        )
        refined = phase_correlation_offset(
            base,
            shifted,
            normalization="unit",
            max_abs_shift=8,
            subpixel=True,
        )
        assert abs(refined.dx - true_dx) < abs(integer.dx - true_dx)
        assert abs(refined.dx - true_dx) < 0.1


def test_rotation_records_no_translation_ground_truth() -> None:
    class _Fake:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "image_t1": _textured(size=64, seed=5),
                "image_t2": _textured(size=64, seed=6),
                "mask": torch.zeros(1, 64, 64),
                "id": "img",
            }

    rotated = CorruptedPairDataset(
        _Fake(),
        CorruptionSpec(name="rotate", severity=3, mode="t2"),
        normalization="unit",
    )[0]
    # A rotation has no correct translation; the offset stays zero so no
    # tool can silently score an estimate against it as if it did.
    assert rotated["shift_dx"] == 0
    assert rotated["shift_dy"] == 0
    assert abs(rotated["rotation_degrees"]) == pytest.approx(1.0)

    sub = CorruptedPairDataset(
        _Fake(),
        CorruptionSpec(name="subpixel_shift", severity=1, mode="t2"),
        normalization="unit",
    )[0]
    magnitude = (
        sub["shift_dx"] ** 2 + sub["shift_dy"] ** 2
    ) ** 0.5
    assert magnitude == pytest.approx(0.5)
    assert sub["rotation_degrees"] == 0.0


def test_perturbation_specs_cover_both_families() -> None:
    specs = perturbation_specs(
        {
            "seed": 3407,
            "perturbations": [
                {"name": "subpixel_shift", "mode": "t2", "severities": [1, 3]},
                {"name": "rotate", "mode": "t2", "severities": [2]},
            ],
        }
    )
    assert [spec.name for spec in specs] == [
        "clean",
        "subpixel_shift",
        "subpixel_shift",
        "rotate",
    ]
    assert specs[1].value == pytest.approx(0.5)
    assert specs[2].value == pytest.approx(3.0)
    assert specs[3].value == pytest.approx(0.5)


def _rows(primary_deltas, clean_delta=0.0):
    rows = [
        {"scenario": "clean:s0:common", "method": "raw", "f1": 0.90},
        {
            "scenario": "clean:s0:common",
            "method": "safe_veto_task",
            "f1": 0.90 + clean_delta,
        },
    ]
    for scenario, delta in primary_deltas.items():
        rows.append({"scenario": scenario, "method": "raw", "f1": 0.85})
        rows.append(
            {
                "scenario": scenario,
                "method": "safe_veto_task",
                "f1": 0.85 + delta,
            }
        )
    return rows


def test_graceful_degradation_passes() -> None:
    decision = safety_decision(
        _rows(
            {
                "rotate:s1:t2": -0.001,
                "rotate:s3:t2": 0.0,
                "subpixel_shift:s1:t2": 0.002,
            }
        ),
        primary_method="safe_veto_task",
        harm_tolerance=0.005,
    )
    assert decision["no_harm_gate_passed"] is True
    assert decision["verdict"] == "router_degrades_gracefully_outside_scope"
    assert set(decision["families"]) == {"rotate", "subpixel_shift"}


def test_net_harm_is_reported_as_a_limitation() -> None:
    decision = safety_decision(
        _rows({"rotate:s3:t2": -0.02, "subpixel_shift:s1:t2": 0.001}),
        primary_method="safe_veto_task",
        harm_tolerance=0.005,
    )
    assert decision["no_harm_gate_passed"] is False
    assert (
        decision["verdict"] == "router_does_net_harm_documented_limitation"
    )
    assert decision["families"]["rotate"]["all_no_harm"] is False
    assert decision["families"]["subpixel_shift"]["all_no_harm"] is True


def test_clean_regression_alone_fails_the_gate() -> None:
    decision = safety_decision(
        _rows({"rotate:s1:t2": 0.0}, clean_delta=-0.02),
        primary_method="safe_veto_task",
        harm_tolerance=0.005,
    )
    assert decision["no_harm_gate_passed"] is False
