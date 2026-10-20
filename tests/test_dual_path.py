import torch

from rs_cd.dual_path import (
    REGISTRATION_ROUTER_FEATURE_NAMES,
    ROUTER_FEATURE_NAMES,
    SAFE_REGISTRATION_FEATURE_NAMES,
    SAFE_TASK_FEATURE_NAMES,
    SafeAlignmentClassifier,
    TaskBenefitRouter,
    dual_path_router_features,
    per_sample_binary_change_loss,
    select_aligned_by_task_loss,
)


def test_per_sample_loss_respects_valid_mask() -> None:
    targets = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 0.0]]],
        ]
    )
    logits = torch.tensor(
        [
            [[[8.0, -8.0], [-8.0, 8.0]]],
            [[[-8.0, 8.0], [-8.0, -8.0]]],
        ]
    )
    valid = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
    losses = per_sample_binary_change_loss(logits, targets, valid)
    assert losses.shape == (2,)
    assert torch.all(losses < 0.01)


def test_task_loss_oracle_selects_one_complete_branch_per_sample() -> None:
    targets = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )
    raw = torch.tensor(
        [
            [[[8.0, -8.0]]],
            [[[8.0, -8.0]]],
        ]
    )
    aligned = torch.tensor(
        [
            [[[-8.0, 8.0]]],
            [[[-8.0, 8.0]]],
        ]
    )
    valid = torch.ones_like(targets)
    selected, use_aligned, raw_loss, aligned_loss = (
        select_aligned_by_task_loss(raw, aligned, targets, valid)
    )
    assert use_aligned.tolist() == [False, True]
    torch.testing.assert_close(selected[0], raw[0])
    torch.testing.assert_close(selected[1], aligned[1])
    assert raw_loss[0] < aligned_loss[0]
    assert aligned_loss[1] < raw_loss[1]


def test_router_features_are_label_free_finite_and_fixed_width() -> None:
    raw = torch.tensor(
        [
            [[[8.0, -8.0], [-8.0, -8.0]]],
            [[[-8.0, 8.0], [-8.0, -8.0]]],
        ]
    )
    aligned = torch.tensor(
        [
            [[[-8.0, 8.0], [-8.0, -8.0]]],
            [[[-8.0, 8.0], [-8.0, -8.0]]],
        ]
    )
    valid = torch.ones_like(raw)
    registration = {
        "registration_shift_probability": torch.tensor([0.1, 0.9]),
        "registration_zero_probability": torch.tensor([0.9, 0.1]),
        "registration_confidence": torch.tensor([0.6, 0.8]),
        "registration_margin": torch.tensor([0.01, 0.02]),
        "registration_local_margin": torch.tensor([0.03, 0.04]),
        "registration_reliability": torch.tensor([0.7, 0.9]),
        "conditional_hard_offset_pixels": torch.tensor(
            [[2.0, 0.0], [8.0, -8.0]]
        ),
    }
    features = dual_path_router_features(
        raw,
        aligned,
        valid,
        registration,
        maximum_offset_pixels=8.0,
    )
    assert features.shape == (2, len(ROUTER_FEATURE_NAMES))
    assert torch.isfinite(features).all()
    boundary_index = ROUTER_FEATURE_NAMES.index("conditional_boundary")
    assert features[:, boundary_index].tolist() == [0.0, 1.0]


def test_task_benefit_router_returns_one_score_per_sample() -> None:
    router = TaskBenefitRouter(input_dim=5, hidden_dims=(8, 4), dropout=0.0)
    scores = router(torch.zeros(3, 5))
    assert scores.shape == (3,)
    scores.sum().backward()
    assert all(
        parameter.grad is not None for parameter in router.parameters()
    )


def test_safe_feature_sets_are_direction_invariant_subsets() -> None:
    signed = {"conditional_dx_normalized", "conditional_dy_normalized"}
    assert signed.isdisjoint(SAFE_REGISTRATION_FEATURE_NAMES)
    assert signed.isdisjoint(SAFE_TASK_FEATURE_NAMES)
    assert set(SAFE_REGISTRATION_FEATURE_NAMES) < set(
        REGISTRATION_ROUTER_FEATURE_NAMES
    )
    assert set(SAFE_TASK_FEATURE_NAMES) < set(ROUTER_FEATURE_NAMES)
    assert len(SAFE_REGISTRATION_FEATURE_NAMES) == 10
    assert len(SAFE_TASK_FEATURE_NAMES) == 23
    assert set(SAFE_REGISTRATION_FEATURE_NAMES) < set(
        SAFE_TASK_FEATURE_NAMES
    )


def test_safe_alignment_classifier_returns_one_logit_per_sample() -> None:
    classifier = SafeAlignmentClassifier(
        input_dim=7,
        hidden_dims=(8, 4),
        dropout=0.0,
    )
    logits = classifier(torch.zeros(3, 7))
    assert logits.shape == (3,)
    logits.sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in classifier.parameters()
    )
