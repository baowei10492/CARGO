from __future__ import annotations

import numpy as np
import pytest

from rs_cd.affine_registration import perturbation_matrix
from tools.diagnose_p26_flow import compare, truth_field


def test_truth_field_is_the_training_ground_truth() -> None:
    matrix = perturbation_matrix(0.0, 1.0, 12.75, -25.5, width=256, height=256)
    field = truth_field(matrix, 4, 4, 256)
    assert field.shape == (4, 4, 2)
    assert field[..., 0] == pytest.approx(np.full((4, 4), 0.1))
    assert field[..., 1] == pytest.approx(np.full((4, 4), -0.2))


def test_compare_separates_sign_and_magnitude() -> None:
    matrix = perturbation_matrix(10.0, 1.1, 5.0, 3.0, width=256, height=256)
    truth = truth_field(matrix, 8, 8, 256)
    exact = compare(truth, truth, 256)
    assert exact["epe_px"] == pytest.approx(0.0) and exact["mean_cosine"] == pytest.approx(1.0)
    assert exact["epe_negated_px"] == pytest.approx(2 * exact["true_mean_px"])
    flipped = compare(-truth, truth, 256)
    assert flipped["epe_negated_px"] == pytest.approx(0.0) and flipped["mean_cosine"] == pytest.approx(-1.0)
    zero = compare(np.zeros_like(truth), truth, 256)
    assert zero["pred_mean_px"] == 0.0 and zero["epe_px"] == pytest.approx(zero["true_mean_px"])


def test_compare_drops_overflowing_targets() -> None:
    truth = np.zeros((2, 2, 2))
    truth[0, 0] = (1.5, 0.0)
    result = compare(np.zeros_like(truth), truth, 256)
    assert result["points"] == 3 and result["true_mean_px"] == 0.0
