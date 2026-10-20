"""P27 version 1 localizer: geometry, network wiring and gradients."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from rs_cd import affine_registration as ref_geometry
from rs_cd.models.change_aware_similarity_localizer import (
    ChangeAwareScaleTranslationLocalizer,
    _neighbour_table,
    build_localizer,
)
from rs_cd.similarity_geometry import (
    apply_affine,
    compose,
    fit_scale_translation,
    grid,
    identity,
    invert,
    mean_displacement,
    pixel_centres,
    robust_fit_scale_translation,
    scale_translation,
    transfer_distance,
    warp_image,
)

from p27_helpers import TinyBackbone


def _matrix(scale, tx, ty, size=256):
    return ref_geometry.perturbation_matrix(0.0, scale, tx, ty, width=size, height=size)


def test_affine_helpers_match_the_numpy_reference() -> None:
    a = _matrix(1.2, 5.0, -3.0)
    b = ref_geometry.perturbation_matrix(12.0, 0.9, -4.0, 2.0, width=256, height=256)
    points = ref_geometry.grid_points(256, 256, 5)
    ta, tb = torch.tensor(a)[None], torch.tensor(b)[None]
    tp = torch.tensor(points)
    assert apply_affine(ta, tp)[0].numpy() == pytest.approx(ref_geometry.apply(a, points))
    assert compose(ta, tb)[0].numpy() == pytest.approx(ref_geometry.compose(a, b))
    round_trip = compose(invert(tb), tb)[0].numpy()
    assert round_trip == pytest.approx(np.eye(2, 3), abs=1e-6)
    built = scale_translation(torch.tensor([1.2]), torch.tensor([a[0, 2]]), torch.tensor([a[1, 2]]))
    assert built[0].numpy() == pytest.approx(a)
    assert identity(2).shape == (2, 2, 3)


def test_scale_translation_fits_exactly_and_resists_outliers() -> None:
    truth = torch.tensor(_matrix(0.85, 17.0, -9.0))[None].float()
    source = torch.rand(1, 200, 2) * 255
    target = apply_affine(truth, source)
    weights = torch.ones(1, 200)
    assert fit_scale_translation(source, target, weights)[0] == pytest.approx(truth[0], abs=1e-3)
    corrupted = target.clone()
    corrupted[:, :40] += torch.rand(1, 40, 2) * 80 + 20
    robust = robust_fit_scale_translation(source, corrupted, weights, iterations=6, huber_px=2.0)
    plain = fit_scale_translation(source, corrupted, weights)
    points = grid(256, 256, 9)
    assert transfer_distance(robust, truth, points).mean() < 0.5 * transfer_distance(plain, truth, points).mean()
    zero_weight = fit_scale_translation(source, target, weights * 0)
    assert torch.isfinite(zero_weight).all()


def test_warp_image_matches_the_opencv_alignment() -> None:
    rng = np.random.default_rng(0)
    image = cv2.resize((rng.random((32, 32, 3)) * 255).astype(np.uint8), (128, 128), interpolation=cv2.INTER_CUBIC)
    matrix = _matrix(1.1, 6.5, -4.25, size=128)
    expected = ref_geometry.align(image, matrix).astype(np.float32)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    warped = warp_image(tensor, torch.tensor(matrix)[None].float())[0].permute(1, 2, 0).numpy()
    inner = slice(20, 108)
    assert np.abs(warped[inner, inner] - expected[inner, inner]).max() < 1.0
    assert warp_image(tensor, identity(1))[0].permute(1, 2, 0).numpy() == pytest.approx(image.astype(np.float32), abs=1e-3)


def test_pixel_centres_and_displacement() -> None:
    centres = pixel_centres(2, 2, 8)
    assert centres.tolist() == [[3.5, 3.5], [11.5, 3.5], [3.5, 11.5], [11.5, 11.5]]
    shifted = scale_translation(torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([4.0]))
    assert mean_displacement(shifted, grid(64, 64, 5)).item() == pytest.approx(5.0)


def test_neighbour_table_clamps_at_the_border() -> None:
    table = _neighbour_table(3, 4, 1)
    assert table.shape == (12, 9)
    assert table[0].tolist() == [0, 0, 1, 0, 0, 1, 4, 4, 5]
    assert table[5].tolist() == [0, 1, 2, 4, 5, 6, 8, 9, 10]


def _model(**overrides) -> ChangeAwareScaleTranslationLocalizer:
    config = dict(backbone_channels=[12, 24], coarse_dim=32, fine_dim=16, fine_iterations=2)
    config.update(overrides)
    return build_localizer(config, backbone=TinyBackbone())


def _texture(size=128, seed=1):
    rng = np.random.default_rng(seed)
    image = cv2.resize((rng.random((size // 4, size // 4, 3)) * 255).astype(np.uint8), (size, size),
                       interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(image).permute(2, 0, 1).float()[None] / 127.5 - 1.0


def test_forward_shapes_and_stage_count() -> None:
    model = _model()
    reference = _texture()
    output = model(reference.repeat(2, 1, 1, 1), reference.repeat(2, 1, 1, 1))
    assert output["matrix"].shape == (2, 2, 3)
    assert len(output["stage_matrices"]) == 3
    assert output["change_logits"].shape == (2, 16, 16)
    assert output["fine_confidence"].shape == (2, 32, 32)
    assert output["misaligned_logit"].shape == (2,)


def test_untrained_pipeline_recovers_a_translation_of_a_textured_image() -> None:
    model = _model().eval()
    reference = _texture()
    truth = torch.tensor(_matrix(1.0, 16.0, -8.0, size=128))[None].float()
    # target(q) = reference(M^-1 q): content moves by +16, -8
    target = warp_image(reference, invert(truth))
    with torch.no_grad():
        output = model(reference, target)
    points = grid(128, 128, 9)
    inner = points[(points[:, 0] > 30) & (points[:, 0] < 90) & (points[:, 1] > 30) & (points[:, 1] < 90)]
    assert transfer_distance(output["stage_matrices"][0], truth, inner).mean() < 6.0
    assert transfer_distance(output["matrix"], truth, inner).mean() < 3.0


def test_gradients_reach_every_trainable_part() -> None:
    model = _model()
    reference = _texture()
    truth = torch.tensor(_matrix(1.05, 6.0, 3.0, size=128))[None].float()
    target = warp_image(reference, invert(truth))
    output = model(reference, target)
    points = grid(128, 128, 5)
    loss = sum(transfer_distance(matrix, truth, points).mean() for matrix in output["stage_matrices"])
    loss = loss + output["change_logits"].mean() + output["misaligned_logit"].mean()
    loss.backward()
    for name in ("coarse_adapter", "fine_adapter", "change_head", "misalignment_head"):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
    assert model.log_coarse_temperature.grad is not None
    assert model.log_fine_temperature.grad is not None
    assert all(p.grad is None for p in model.backbone.parameters())
