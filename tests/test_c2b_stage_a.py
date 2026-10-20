import pytest

from tools.diagnose_transfer_offset_consistency import (
    offset_consistency_summary,
)
from tools.summarize_c2b_stage_a import c2b_stage_a_decision


def _offset_summary(
    s4_pck1=0.80,
    s4_pck2=0.90,
    s8_pck1=0.80,
    s8_pck2=0.90,
    clean_q=0.05,
    boundary_fraction=0.10,
    gt_boundary_fraction=0.10,
):
    shift8 = {
        "raw_pck1": s8_pck1,
        "raw_pck2": s8_pck2,
    }
    if boundary_fraction is not None:
        shift8["boundary_fraction"] = boundary_fraction
        if gt_boundary_fraction is not None:
            shift8["gt_boundary_fraction"] = gt_boundary_fraction
            shift8["excess_boundary_fraction"] = (
                boundary_fraction - gt_boundary_fraction
            )
    return {
        "per_shift": {
            "shift:s1:t2": {"raw_pck1": 0.8, "raw_pck2": 0.9},
            "shift:s2:t2": {
                "raw_pck1": s4_pck1,
                "raw_pck2": s4_pck2,
            },
            "shift:s3:t2": shift8,
        },
        "q_trigger_rate": {"clean:s0:common": clean_q},
    }


def _registration_summary(clean_within1=0.95):
    return {
        "0": {"hard_within_1px_rate": clean_within1},
        "1": {"hard_within_1px_rate": 0.5},
        "2": {"hard_within_1px_rate": 0.5},
        "3": {"hard_within_1px_rate": 0.5},
    }


def test_decision_full_pass_routes_to_step3() -> None:
    decision = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(),
    )
    assert decision["estimator_conditional_passed"]
    assert decision["clean_gate_passed"]
    assert (
        decision["next_action"]
        == "localizer_stage_a_passed_continue_chain"
    )
    assert decision["shift8_boundary_fraction_reported"] == pytest.approx(
        0.10
    )


def test_decision_clean_failure_routes_to_calibration() -> None:
    decision = c2b_stage_a_decision(
        _registration_summary(clean_within1=0.60),
        _offset_summary(clean_q=0.40),
    )
    assert decision["estimator_conditional_passed"]
    assert not decision["clean_gate_passed"]
    assert (
        decision["next_action"]
        == "localizer_stage_a_fullres_q_calibration"
    )


def test_decision_conditional_failure_routes_to_stop_loss() -> None:
    decision = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(s8_pck1=0.40, s8_pck2=0.60),
    )
    assert not decision["estimator_conditional_passed"]
    assert (
        decision["next_action"]
        == "localizer_stage_a_stop_loss_scope_review"
    )
    # Provenance for decision files written before the rename.
    assert (
        decision["next_action_legacy_label"]
        == "s2looking_scope_review_stop_loss"
    )


def test_branch_labels_are_dataset_agnostic() -> None:
    # A WHU run must not be labelled with another dataset's name.
    for decision in (
        c2b_stage_a_decision(_registration_summary(), _offset_summary()),
        c2b_stage_a_decision(
            _registration_summary(clean_within1=0.60),
            _offset_summary(),
        ),
        c2b_stage_a_decision(
            _registration_summary(),
            _offset_summary(s8_pck1=0.10, s8_pck2=0.20),
        ),
    ):
        assert "s2looking" not in decision["next_action"]
        assert "whu" not in decision["next_action"]
        assert decision["next_action"].startswith("localizer_stage_a_")


def test_boundary_fraction_in_offset_summary() -> None:
    records = []
    for index, est in enumerate(
        [(8.0, 0.0), (6.0, 0.0), (-8.0, 8.0), (2.0, 2.0)]
    ):
        sample_id = f"img_{index}"
        records.append(
            {
                "id": sample_id,
                "scenario": "clean:s0:common",
                "est_dx": 2.0,
                "est_dy": 0.0,
                "gt_dx": 0.0,
                "gt_dy": 0.0,
                "q_gate": True,
            }
        )
        records.append(
            {
                "id": sample_id,
                "scenario": "shift:s3:t2",
                "est_dx": est[0],
                "est_dy": est[1],
                "gt_dx": 8.0,
                "gt_dy": 0.0,
                "q_gate": True,
            }
        )
    summary = offset_consistency_summary(
        records,
        clean_scenario="clean:s0:common",
        shift_scenarios=["shift:s3:t2"],
        boundary_threshold=7.9,
    )
    assert summary["per_shift"]["shift:s3:t2"][
        "boundary_fraction"
    ] == pytest.approx(0.5)


def test_boundary_contamination_is_annotation_never_a_gate() -> None:
    # Estimates pile onto a boundary the truth never reaches.
    saturated = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(
            boundary_fraction=0.706,
            gt_boundary_fraction=0.05,
        ),
    )
    matched = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(
            boundary_fraction=0.05,
            gt_boundary_fraction=0.05,
        ),
    )
    assert saturated["shift8_boundary_contamination"]["contaminated"]
    assert not matched["shift8_boundary_contamination"]["contaminated"]
    # The formal verdict must be identical in both cases.
    for key in (
        "estimator_conditional_passed",
        "clean_gate_passed",
        "conditional_gates",
        "clean_gates",
        "next_action",
    ):
        assert saturated[key] == matched[key]
    # A failing S8 with high excess boundary still fails.
    failing = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(
            s8_pck1=0.30,
            s8_pck2=0.50,
            boundary_fraction=0.706,
            gt_boundary_fraction=0.05,
        ),
    )
    assert not failing["estimator_conditional_passed"]
    assert failing["shift8_boundary_contamination"]["contaminated"]


def test_perfect_estimator_on_boundary_truth_is_not_contaminated() -> None:
    # The real LEVIR Shift-8 measurement: 98.4% of estimates reach the
    # search limit because 98.4% of the injected truths do. PCK@1 is
    # 0.922 and mean EPE is 0.58 -- the best measurement in the project.
    # The pre-correction rule flagged this as contaminated.
    decision = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(
            boundary_fraction=0.984375,
            gt_boundary_fraction=0.984375,
        ),
    )
    annotation = decision["shift8_boundary_contamination"]
    assert annotation["available"] is True
    assert annotation["contaminated"] is False
    assert annotation["undershooting"] is False
    assert (
        annotation["interpretation"]
        == "shift8_boundary_behaviour_matches_truth"
    )


def test_undershoot_is_distinguished_from_clamping() -> None:
    # The real WHU Shift-8 measurement: estimates reach the boundary far
    # LESS often than the truth does, which is the opposite failure.
    decision = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(
            s8_pck1=0.36,
            s8_pck2=0.44,
            boundary_fraction=0.4655,
            gt_boundary_fraction=0.984375,
        ),
    )
    annotation = decision["shift8_boundary_contamination"]
    assert annotation["contaminated"] is False
    assert annotation["undershooting"] is True
    assert (
        annotation["interpretation"]
        == "shift8_estimates_undershoot_the_true_magnitude"
    )


def test_boundary_contamination_reports_missing_measurement() -> None:
    decision = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(boundary_fraction=None),
    )
    annotation = decision["shift8_boundary_contamination"]
    assert annotation["available"] is False
    assert "contaminated" not in annotation
    assert (
        annotation["interpretation"] == "boundary_fraction_not_reported"
    )


def test_legacy_summary_without_gt_boundary_is_not_annotated() -> None:
    decision = c2b_stage_a_decision(
        _registration_summary(),
        _offset_summary(
            boundary_fraction=0.706,
            gt_boundary_fraction=None,
        ),
    )
    annotation = decision["shift8_boundary_contamination"]
    assert annotation["available"] is False
    assert "contaminated" not in annotation
    assert "gt_boundary_fraction_missing" in annotation["interpretation"]


def test_filter_state_by_prefixes_drops_radius_dependent_heads() -> None:
    import torch

    from tools.build_c2b_init_checkpoint import filter_state_by_prefixes

    state = {
        "backbone.stem.weight": torch.zeros(1),
        "registration.descriptor.0.weight": torch.zeros(1),
        "registration.zero_anchor_head.0.weight": torch.zeros(81, 16),
        "registration.shiftedness_head.0.weight": torch.zeros(16, 166),
        "registration.shiftedness_head.2.weight": torch.zeros(1, 16),
    }
    kept, dropped = filter_state_by_prefixes(
        state,
        (
            "registration.zero_anchor_head.",
            "registration.shiftedness_head.",
        ),
    )
    assert len(dropped) == 3
    assert set(kept) == {
        "backbone.stem.weight",
        "registration.descriptor.0.weight",
    }
    kept_all, dropped_none = filter_state_by_prefixes(state, ())
    assert dropped_none == []
    assert len(kept_all) == len(state)
