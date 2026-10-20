import torch
from torch.nn import functional as F

from rs_cd.models import (
    DinoV3ConvNeXtChangeDetector,
    DinoV3GlobalRegistrationChangeDetector,
    DinoV3SelectiveCorrespondenceChangeDetector,
    DinoV3TaskAwareInputRegistrationChangeDetector,
    FCSiamDiff,
    GlobalTranslationEstimator,
    TaskAwareInputRegistrationEstimator,
    warp_feature,
)


def test_model_shape_symmetry_and_gradient() -> None:
    model = FCSiamDiff(in_channels=3, base_channels=8)
    t1 = torch.randn(2, 3, 65, 67, requires_grad=True)
    t2 = torch.randn(2, 3, 65, 67, requires_grad=True)
    output_forward = model(t1, t2)["change_logits"]
    output_swapped = model(t2, t1)["change_logits"]
    assert output_forward is not None
    assert output_forward.shape == (2, 1, 65, 67)
    torch.testing.assert_close(output_forward, output_swapped)
    output_forward.mean().backward()
    assert t1.grad is not None
    assert t2.grad is not None
    assert torch.isfinite(t1.grad).all()
    assert torch.isfinite(t2.grad).all()


def test_model_rejects_mismatched_temporal_shapes() -> None:
    model = FCSiamDiff(in_channels=3, base_channels=8)
    t1 = torch.randn(1, 3, 32, 32)
    t2 = torch.randn(1, 3, 31, 32)
    try:
        model(t1, t2)
    except ValueError as error:
        assert "shapes must match" in str(error)
    else:
        raise AssertionError("Expected ValueError for mismatched T1/T2")


def test_dinov3_convnext_shape_symmetry_and_decoder_gradient() -> None:
    model = DinoV3ConvNeXtChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        freeze_backbone=True,
        decoder_channels=16,
    ).eval()
    t1 = torch.randn(1, 3, 64, 64)
    t2 = torch.randn(1, 3, 64, 64)
    output_forward = model(t1, t2)["change_logits"]
    output_swapped = model(t2, t1)["change_logits"]
    assert output_forward is not None
    assert output_forward.shape == (1, 1, 64, 64)
    torch.testing.assert_close(output_forward, output_swapped)
    output_forward.mean().backward()
    decoder_gradients = [
        parameter.grad
        for parameter in model.decoder.parameters()
        if parameter.requires_grad
    ]
    assert decoder_gradients
    assert all(gradient is not None for gradient in decoder_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in decoder_gradients
        if gradient is not None
    )
    assert not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )


def test_dinov3_partial_unfreezing_targets_last_two_stages() -> None:
    model = DinoV3ConvNeXtChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        freeze_backbone=False,
        trainable_stages=2,
        decoder_channels=16,
    )
    trainable_backbone_names = {
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_backbone_names
    assert all(
        name.startswith(("stages_2.", "stages_3."))
        for name in trainable_backbone_names
    )
    assert any(
        name.startswith("stages_2.") for name in trainable_backbone_names
    )
    assert any(
        name.startswith("stages_3.") for name in trainable_backbone_names
    )


def test_dinov3_selective_correspondence_shape_symmetry_and_aux_gradient() -> None:
    model = DinoV3SelectiveCorrespondenceChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        freeze_backbone=True,
        correspondence_channels=8,
        correspondence_radius=1,
        decoder_channels=16,
        freeze_base_model=True,
    ).eval()
    t1 = torch.randn(1, 3, 64, 64)
    t2 = torch.randn(1, 3, 64, 64)
    mask = torch.zeros(1, 1, 64, 64)
    batch = {
        "misregistration_target": ["none"],
        "misregistration_dx": torch.zeros(1),
        "misregistration_dy": torch.zeros(1),
    }
    torch.nn.init.normal_(
        model.correspondence.correction_head[-1].weight,
        mean=0.0,
        std=0.01,
    )
    output_forward = model(t1, t2, targets=mask, batch=batch)["change_logits"]
    output_swapped = model(t2, t1, targets=mask, batch=batch)["change_logits"]
    assert output_forward.shape == (1, 1, 64, 64)
    torch.testing.assert_close(output_forward, output_swapped, rtol=1e-4, atol=1e-5)

    train_model = DinoV3SelectiveCorrespondenceChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        freeze_backbone=True,
        correspondence_channels=8,
        correspondence_radius=1,
        decoder_channels=16,
        freeze_base_model=True,
    )
    shifted_batch = {
        "misregistration_target": ["t2"],
        "misregistration_dx": torch.tensor([4]),
        "misregistration_dy": torch.tensor([0]),
    }
    train_output = train_model(t1, t2, targets=mask, batch=shifted_batch)
    loss = train_output["change_logits"].mean()
    if isinstance(train_output.get("aux_loss"), torch.Tensor):
        loss = loss + train_output["aux_loss"]
    loss.backward()
    aux_gradients = [
        parameter.grad
        for name, parameter in train_model.named_parameters()
        if parameter.requires_grad and name.startswith("correspondence.")
    ]
    assert aux_gradients
    assert all(gradient is not None for gradient in aux_gradients)
    assert any(
        bool(torch.count_nonzero(gradient).item())
        for gradient in aux_gradients
        if gradient is not None
    )


def test_global_translation_estimator_recovers_integer_shift() -> None:
    torch.manual_seed(7)
    reference = torch.randn(2, 8, 16, 16)
    expected = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    # Creating target with the negative sampling offset translates its content
    # by the expected positive displacement.
    target = warp_feature(reference, -expected)
    estimator = GlobalTranslationEstimator(
        channels=8,
        radius=2,
        initial_logit_scale=1000.0,
    )
    output = estimator(reference, target)
    torch.testing.assert_close(
        output["offset_feature"],
        expected,
        rtol=0.0,
        atol=0.15,
    )


def test_dinov3_global_registration_stage_contract_and_gradients() -> None:
    model = DinoV3GlobalRegistrationChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        freeze_backbone=True,
        decoder_channels=16,
        search_radius=1,
        registration_only=True,
    )
    t1 = torch.randn(2, 3, 64, 64)
    t2 = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    batch = {
        "misregistration_target": ["t2", "none"],
        "misregistration_dx": torch.tensor([4, 0]),
        "misregistration_dy": torch.tensor([0, 0]),
    }
    outputs = model(t1, t2, targets=mask, batch=batch)
    assert outputs["change_logits"].shape == (2, 1, 64, 64)
    torch.testing.assert_close(
        outputs["change_logits"],
        outputs["raw_change_logits"],
    )
    assert outputs["predicted_offset_pixels"].shape == (2, 2)
    assert outputs["predicted_hard_offset_pixels"].shape == (2, 2)
    loss = outputs["aux_loss"]
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    registration_gradients = [
        parameter.grad
        for parameter in model.registration.parameters()
        if parameter.requires_grad
    ]
    assert registration_gradients
    assert all(gradient is not None for gradient in registration_gradients)
    assert any(
        bool(torch.count_nonzero(gradient).item())
        for gradient in registration_gradients
        if gradient is not None
    )
    assert not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.decoder.parameters()
    )

    model.zero_grad(set_to_none=True)
    model.registration_only = False
    model.alignment_parameterization = "sigmoid"
    model.alignment_logit_bias = -3.0
    model.registration.requires_grad_(False)
    model.alignment_strengths.requires_grad_(True)
    torch.testing.assert_close(
        model._alignment_strength_values(),
        torch.full((4,), torch.sigmoid(torch.tensor(-3.0))),
    )
    fused_outputs = model(t1, t2)
    fused_outputs["change_logits"].mean().backward()
    assert model.alignment_strengths.grad is not None
    assert bool(torch.count_nonzero(model.alignment_strengths.grad).item())


def test_surface_decoder_receives_registration_gradients() -> None:
    torch.manual_seed(19)
    reference = torch.randn(4, 8, 16, 16)
    target_offset = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    target = warp_feature(reference, -target_offset)
    estimator = GlobalTranslationEstimator(
        channels=8,
        radius=2,
        initial_logit_scale=2.0,
        surface_decoder_channels=4,
        normalize_score_surface=True,
    )
    output = estimator(reference, target)
    loss = F.smooth_l1_loss(
        output["offset_feature"],
        target_offset,
    )
    loss.backward()
    decoder_gradients = [
        parameter.grad
        for name, parameter in estimator.named_parameters()
        if name.startswith(("surface_refiner.", "zero_offset_head."))
    ]
    assert decoder_gradients
    assert all(gradient is not None for gradient in decoder_gradients)
    assert any(
        bool(torch.count_nonzero(gradient).item())
        for gradient in decoder_gradients
        if gradient is not None
    )


def test_task_aware_input_estimator_recovers_descriptor_shift() -> None:
    torch.manual_seed(23)
    estimator = TaskAwareInputRegistrationEstimator(
        in_channels=3,
        descriptor_channels=8,
        search_radius=2,
        top_fraction=0.5,
        surface_decoder_channels=0,
    ).eval()
    reference = F.normalize(torch.randn(2, 8, 24, 24), dim=1)
    expected = torch.tensor([[1.0, -1.0], [-2.0, 1.0]])
    target = warp_feature(reference, -expected)
    outputs = estimator.forward_descriptors(reference, target)
    torch.testing.assert_close(
        outputs["hard_offset_feature"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_task_aware_input_registration_contract_and_gradient() -> None:
    model = DinoV3TaskAwareInputRegistrationChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        decoder_channels=16,
        descriptor_channels=8,
        search_radius=2,
        top_fraction=0.5,
        surface_decoder_channels=4,
        registration_only=True,
    )
    t1 = torch.randn(2, 3, 64, 64)
    t2 = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    batch = {
        "misregistration_target": ["t2", "none"],
        "misregistration_dx": torch.tensor([4, 0]),
        "misregistration_dy": torch.tensor([0, 0]),
    }
    outputs = model(t1, t2, targets=mask, batch=batch)
    assert outputs["change_logits"].shape == (2, 1, 64, 64)
    assert outputs["predicted_offset_pixels"].shape == (2, 2)
    assert outputs["predicted_hard_offset_pixels"].shape == (2, 2)
    loss = outputs["aux_loss"]
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    registration_gradients = [
        parameter.grad
        for parameter in model.registration.parameters()
        if parameter.requires_grad
    ]
    assert registration_gradients
    assert all(
        gradient is not None for gradient in registration_gradients
    )
    assert any(
        bool(torch.count_nonzero(gradient).item())
        for gradient in registration_gradients
        if gradient is not None
    )
    assert not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.decoder.parameters()
    )


def test_decoupled_shift_decision_excludes_zero_and_backpropagates() -> None:
    model = DinoV3TaskAwareInputRegistrationChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        decoder_channels=16,
        descriptor_channels=8,
        search_radius=2,
        top_fraction=0.5,
        surface_decoder_channels=4,
        registration_only=True,
        decoupled_shift_decision=True,
        shiftedness_weight=1.0,
    )
    assert model.registration.zero_anchor_head is not None
    assert not any(
        parameter.requires_grad
        for parameter in model.registration.zero_anchor_head.parameters()
    )
    t1 = torch.randn(2, 3, 64, 64)
    t2 = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    batch = {
        "misregistration_target": ["t2", "none"],
        "misregistration_dx": torch.tensor([4, 0]),
        "misregistration_dy": torch.tensor([0, 0]),
    }
    outputs = model(t1, t2, targets=mask, batch=batch)
    assert outputs["predicted_offset_pixels"].shape == (2, 2)
    assert outputs["registration_shift_probability"].shape == (2,)
    registration = model.registration(t1, t2)
    side = 2 * model.registration.search_radius + 1
    center = (
        model.registration.search_radius * side
        + model.registration.search_radius
    )
    assert bool(
        (
            registration["offset_probabilities"][:, center]
            < 1e-6
        ).all()
    )
    assert bool(
        (
            torch.linalg.vector_norm(
                registration["hard_offset_feature"],
                dim=1,
            )
            > 0
        ).all()
    )
    loss = outputs["aux_loss"]
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    shiftedness_gradients = [
        parameter.grad
        for parameter in model.registration.shiftedness_head.parameters()
        if parameter.requires_grad
    ]
    assert shiftedness_gradients
    assert all(
        gradient is not None for gradient in shiftedness_gradients
    )
    assert any(
        bool(torch.count_nonzero(gradient).item())
        for gradient in shiftedness_gradients
        if gradient is not None
    )


def test_full_resolution_decision_calibration_freezes_localizer() -> None:
    model = DinoV3TaskAwareInputRegistrationChangeDetector(
        backbone_name="convnext_tiny.dinov3_lvd1689m",
        pretrained=False,
        decoder_channels=16,
        descriptor_channels=8,
        search_radius=2,
        top_fraction=0.5,
        surface_decoder_channels=4,
        registration_only=True,
        decoupled_shift_decision=True,
        freeze_localizer=True,
        use_change_mask_for_matching=False,
        skip_change_detector_during_registration=True,
        offset_distribution_weight=0.0,
        offset_regression_weight=0.0,
        shiftedness_weight=1.0,
    )
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith("registration.shiftedness_head.")
        for name in trainable_names
    )
    t1 = torch.randn(2, 3, 64, 64)
    t2 = torch.randn(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    batch = {
        "misregistration_target": ["t2", "none"],
        "misregistration_dx": torch.tensor([4, 0]),
        "misregistration_dy": torch.tensor([0, 0]),
    }
    outputs = model(t1, t2, targets=mask, batch=batch)
    torch.testing.assert_close(
        outputs["change_logits"],
        torch.zeros_like(outputs["change_logits"]),
    )
    loss = outputs["aux_loss"]
    assert isinstance(loss, torch.Tensor)
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.registration.shiftedness_head.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.registration.named_parameters()
        if not name.startswith("shiftedness_head.")
    )
