import numpy as np
import pytest
import torch

from rs_cd.registration_controls import sift_translation_offset
from tools.evaluate_classical_baselines import baseline_decision


def _textured(size=192, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(3, size, size, generator=generator)
    # Add blobs so SIFT has repeatable structure to key on.
    grid = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    for cx, cy in ((-0.4, -0.3), (0.35, 0.1), (0.0, 0.45)):
        base += torch.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / 0.01)
        )[None]
    return base.clamp(0.0, 1.0)


def test_sift_follows_the_project_offset_convention() -> None:
    base = _textured()
    # reference(x, y) <-> target(x + dx, y + dy): rolling the target by
    # +dx in x means target(x) = reference(x - dx).
    shifted = torch.roll(base, shifts=(4, -6), dims=(2, 1))
    result = sift_translation_offset(base, shifted, normalization="unit")
    assert result.converged
    assert result.dx == pytest.approx(4.0, abs=0.5)
    assert result.dy == pytest.approx(-6.0, abs=0.5)
    assert 0.0 < result.score <= 1.0


def test_sift_reports_failure_instead_of_guessing() -> None:
    flat = torch.full((3, 64, 64), 0.5)
    result = sift_translation_offset(flat, flat, normalization="unit")
    assert result.converged is False
    assert result.dx == 0.0 and result.dy == 0.0


def _rows(entries):
    rows = [
        {
            "scenario": scenario,
            "method": "raw",
            "variant": "raw",
            "f1": value,
        }
        for scenario, value in (
            ("clean:s0:common", 0.9032),
            ("shift:s1:t2", 0.8672),
            ("shift:s2:t2", 0.8053),
            ("shift:s3:t2", 0.6793),
        )
    ]
    for (method, variant), scores in entries.items():
        for scenario, value in scores.items():
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "variant": variant,
                    "f1": value,
                }
            )
    return rows


_REFERENCE = {
    "reference_shift_mean_gain": 0.0821,
    "reference_clean_drop": 0.0,
    "margin": 0.005,
    "clean_drop_max": 0.001,
}


def test_classical_match_is_reported_not_hidden() -> None:
    # A classical method that reaches the reference operating point.
    rows = _rows(
        {
            ("phase", "gated"): {
                "clean:s0:common": 0.9032,
                "shift:s1:t2": 0.8950,
                "shift:s2:t2": 0.8880,
                "shift:s3:t2": 0.8750,
            }
        }
    )
    decision = baseline_decision(rows, **_REFERENCE)
    assert decision["methods_matching_reference"] == ["phase:gated"]
    assert (
        decision["verdict"] == "classical_sufficient_restate_contribution"
    )


def test_shift_recovery_bought_with_clean_damage_does_not_match() -> None:
    rows = _rows(
        {
            ("phase", "always"): {
                "clean:s0:common": 0.8800,  # 2.3 pp clean damage
                "shift:s1:t2": 0.8950,
                "shift:s2:t2": 0.8880,
                "shift:s3:t2": 0.8930,
            }
        }
    )
    decision = baseline_decision(rows, **_REFERENCE)
    summary = decision["method_summaries"]["phase:always"]
    assert summary["shift_mean_gain"] > 0.08
    assert summary["matches_reference"] is False
    assert decision["verdict"] == "learned_selective_router_justified"


def test_weak_classical_leaves_router_justified() -> None:
    rows = _rows(
        {
            ("sift", "gated"): {
                "clean:s0:common": 0.9032,
                "shift:s1:t2": 0.8700,
                "shift:s2:t2": 0.8100,
                "shift:s3:t2": 0.7000,
            }
        }
    )
    decision = baseline_decision(rows, **_REFERENCE)
    assert decision["methods_matching_reference"] == []
    assert decision["verdict"] == "learned_selective_router_justified"
    assert decision["best_classical_or_gated"] == "sift:gated"


def test_margin_is_applied_symmetrically() -> None:
    # 0.0771 = reference 0.0821 - margin 0.005 exactly: still a match.
    edge = {
        "clean:s0:common": 0.9032,
        "shift:s1:t2": 0.8672 + 0.0771,
        "shift:s2:t2": 0.8053 + 0.0771,
        "shift:s3:t2": 0.6793 + 0.0771,
    }
    decision = baseline_decision(
        _rows({("ecc", "gated"): edge}),
        **_REFERENCE,
    )
    assert decision["method_summaries"]["ecc:gated"][
        "shift_mean_gain"
    ] == pytest.approx(0.0771)
    assert decision["methods_matching_reference"] == ["ecc:gated"]


def test_missing_raw_rows_raise() -> None:
    with pytest.raises(ValueError):
        baseline_decision(
            [
                {
                    "scenario": "clean:s0:common",
                    "method": "phase",
                    "variant": "gated",
                    "f1": 0.9,
                }
            ],
            **_REFERENCE,
        )


def test_deterministic_consensus_has_no_random_seed() -> None:
    base = _textured(seed=1)
    shifted = torch.roll(base, shifts=(2, 3), dims=(2, 1))
    first = sift_translation_offset(base, shifted, normalization="unit")
    second = sift_translation_offset(base, shifted, normalization="unit")
    assert (first.dx, first.dy) == (second.dx, second.dy)
    assert np.isfinite(first.score)


def test_sift_search_window_rejects_far_consensus() -> None:
    # A large true displacement is inside the window when unconstrained
    # and rejected (converged=False, no guess) when the window is small.
    base = _textured(seed=3)
    shifted = torch.roll(base, shifts=(30, 0), dims=(2, 1))
    unconstrained = sift_translation_offset(
        base,
        shifted,
        normalization="unit",
    )
    assert unconstrained.converged
    assert unconstrained.dx == pytest.approx(30.0, abs=1.0)
    windowed = sift_translation_offset(
        base,
        shifted,
        normalization="unit",
        max_abs_shift=8.0,
    )
    assert windowed.converged is False
    assert (windowed.dx, windowed.dy) == (0.0, 0.0)


def test_ground_truth_key_is_the_injected_shift() -> None:
    # Guard for main record 85.1: CorruptedPairDataset records the
    # injected offset under shift_dx / shift_dy. misregistration_dx is
    # the base dataset's own augmentation and is zero at evaluation, so
    # reading it compares every estimate against zero.
    import inspect

    from rs_cd.robustness import CorruptedPairDataset

    source = inspect.getsource(CorruptedPairDataset)
    assert '"shift_dx"' in source and '"shift_dy"' in source

    from tools import evaluate_classical_baselines

    tool = inspect.getsource(evaluate_classical_baselines)
    assert 'batch["shift_dx"]' in tool
    assert 'batch["misregistration_dx"]' not in tool


def test_tool_never_calls_the_full_forward_for_logits() -> None:
    # Guard for main record 85.1: model(t1, t2) runs the P7 forward,
    # which applies the model's OWN registration. Every logits path in
    # this tool must go through forward_with_forced_offset instead.
    import inspect

    from tools import evaluate_classical_baselines

    source = inspect.getsource(evaluate_classical_baselines)
    assert "forward_with_forced_offset" in source
    assert 'model(t1, t2)["change_logits"]' not in source
    assert 'model(t1, corrected)["change_logits"]' not in source


def _offset_records(estimates, intrinsic=(0.0, 0.0), method="phase"):
    rows = []
    for scenario, (gx, gy) in (
        ("clean:s0:common", (0.0, 0.0)),
        ("shift:s1:t2", (2.0, 0.0)),
        ("shift:s3:t2", (8.0, 0.0)),
    ):
        for index in range(4):
            ex, ey = estimates(gx, gy, index)
            rows.append(
                {
                    "id": f"img_{index}",
                    "scenario": scenario,
                    "method": method,
                    "est_dx": ex + intrinsic[0],
                    "est_dy": ey + intrinsic[1],
                    "gt_dx": gx,
                    "gt_dy": gy,
                }
            )
    return rows


def test_decomposition_detects_a_fixed_photometric_residual() -> None:
    from tools.analyze_photometric_vs_task_offsets import decompose

    # A faithful photometric estimator on a pair carrying a constant
    # 3 px intrinsic displacement: raw error is ~3 at every severity,
    # but the delta against its own clean estimate is exact.
    rows = _offset_records(
        lambda gx, gy, index: (gx, gy),
        intrinsic=(3.0, 0.0),
    )
    summary = decompose(rows)["per_method"]["phase"]
    assert summary["clean_estimate_mean_magnitude"] == pytest.approx(3.0)
    for entry in summary["per_shift"].values():
        assert entry["raw_mean_epe"] == pytest.approx(3.0)
        assert entry["delta_mean_epe"] == pytest.approx(0.0)
        assert entry["delta_pck1"] == 1.0
    assert summary["raw_epe_spread"] == pytest.approx(0.0)
    assert summary["mean_delta_improvement"] == pytest.approx(3.0)


def test_decomposition_leaves_a_task_estimator_alone() -> None:
    from tools.analyze_photometric_vs_task_offsets import decompose

    # A task-trained estimator reports the injection only; its clean
    # output is forced-choice noise, so subtracting it makes things
    # WORSE -- delta improvement is negative.
    rows = _offset_records(
        lambda gx, gy, index: (
            (gx, gy) if gx or gy else (2.0 if index % 2 else -2.0, 0.0)
        ),
        method="ours",
    )
    summary = decompose(rows)["per_method"]["ours"]
    assert summary["clean_estimate_mean_magnitude"] == pytest.approx(2.0)
    for entry in summary["per_shift"].values():
        assert entry["raw_mean_epe"] == pytest.approx(0.0)
        assert entry["raw_pck1"] == 1.0
        assert entry["delta_mean_epe"] == pytest.approx(2.0)
    assert summary["mean_delta_improvement"] == pytest.approx(-2.0)


def test_decomposition_requires_clean_rows() -> None:
    from tools.analyze_photometric_vs_task_offsets import decompose

    with pytest.raises(ValueError):
        decompose(
            [
                {
                    "id": "a",
                    "scenario": "shift:s1:t2",
                    "method": "phase",
                    "est_dx": 1.0,
                    "est_dy": 0.0,
                    "gt_dx": 2.0,
                    "gt_dy": 0.0,
                }
            ]
        )


def test_clean_estimate_agreement_between_methods() -> None:
    from tools.analyze_photometric_vs_task_offsets import decompose

    rows = _offset_records(
        lambda gx, gy, index: (gx, gy),
        intrinsic=(3.0, 0.0),
        method="phase",
    )
    rows += _offset_records(
        lambda gx, gy, index: (gx, gy),
        intrinsic=(3.4, 0.0),
        method="ecc",
    )
    agreement = decompose(rows)["clean_estimate_agreement"]
    entry = agreement["ecc|phase"]
    assert entry["samples"] == 4
    assert entry["mean_distance"] == pytest.approx(0.4)
    assert entry["within_1px"] == 1.0


def test_photometric_verification_bands() -> None:
    from tools.verify_photometric_displacement import verify, zero_mean_ncc

    # Warping by the estimate helps on most images; the same-magnitude
    # random control does not.
    real = [
        {"identity": 0.50, "estimated": 0.56, "control": 0.49}
        for _ in range(9)
    ] + [{"identity": 0.50, "estimated": 0.49, "control": 0.48}]
    decision = verify(
        real,
        improvement_fraction_min=0.70,
        control_margin_min=0.002,
    )
    assert decision["fraction_improved_over_identity"] == pytest.approx(0.9)
    assert decision["self_objective_bands_passed"] is True
    assert decision["causal_real_displacement_established"] is False
    assert (
        decision["verdict"]
        == "photometric_objective_gain_observed_causal_geometry_unresolved"
    )

    # The estimate helps no more than a random offset of equal size:
    # consistent with an estimator bias, not a real displacement.
    biased = [
        {"identity": 0.50, "estimated": 0.505, "control": 0.505}
        for _ in range(10)
    ]
    decision = verify(
        biased,
        improvement_fraction_min=0.70,
        control_margin_min=0.002,
    )
    assert decision["gain_margin_over_control"] == pytest.approx(0.0)
    assert (
        decision["verdict"]
        == "photometric_objective_gain_not_established"
    )


def test_zero_mean_ncc_is_shift_sensitive() -> None:
    from tools.verify_photometric_displacement import zero_mean_ncc

    base = _textured(size=96, seed=5)[0]
    assert zero_mean_ncc(base, base, border=8) == pytest.approx(1.0, abs=1e-5)
    shifted = torch.roll(base, shifts=6, dims=1)
    assert zero_mean_ncc(base, shifted, border=8) < 0.95
    flat = torch.zeros_like(base)
    assert zero_mean_ncc(base, flat, border=8) == 0.0
