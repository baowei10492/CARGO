import pytest

from rs_cd.dual_path import SAFE_REGISTRATION_FEATURE_NAMES
from tools.diagnose_registration_hygiene import WHOLE_IMAGE_TILE_INDEX
from tools.screen_tile_consistency_features import (
    TILE_FEATURE_NAMES,
    directional_auc,
    roc_auc,
    screen_features,
    tile_features_by_key,
)


def test_roc_auc_perfect_and_inverted() -> None:
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert roc_auc(scores, [True, True, False, False]) == pytest.approx(
        0.0
    )
    value, direction = directional_auc(
        scores,
        [True, True, False, False],
    )
    assert value == pytest.approx(1.0)
    assert direction == -1


def test_roc_auc_ties_and_degenerate() -> None:
    assert roc_auc([1.0, 1.0, 1.0, 1.0], [True, False, True, False]) == (
        pytest.approx(0.5)
    )
    assert roc_auc([0.1, 0.2], [True, True]) is None
    with pytest.raises(ValueError):
        roc_auc([], [])


def test_tile_features_from_records() -> None:
    records = []
    for tile_index, offset in (
        (WHOLE_IMAGE_TILE_INDEX, (2.0, 0.0)),
        (0, (2.0, 0.0)),
        (1, (2.0, 0.0)),
        (2, (6.0, 0.0)),
        (3, (2.0, 0.0)),
    ):
        records.append(
            {
                "id": "a",
                "scenario": "shift:s1:t2",
                "tile_index": tile_index,
                "est_dx": offset[0],
                "est_dy": offset[1],
            }
        )
    features = tile_features_by_key(records)
    row = features[("a", "shift:s1:t2")]
    assert set(row) == set(TILE_FEATURE_NAMES)
    assert row["tile_max_pairwise_distance"] == pytest.approx(4.0)
    assert row["tile_agreement_within_2px"] == pytest.approx(0.75)
    assert row["tile_agreement_within_4px"] == pytest.approx(1.0)
    assert row["tile_magnitude_spread"] == pytest.approx(4.0)
    assert row["tile_mean_offset_magnitude"] == pytest.approx(3.0)


def test_tile_features_skip_single_tile_samples() -> None:
    records = [
        {
            "id": "a",
            "scenario": "clean:s0:common",
            "tile_index": 0,
            "est_dx": 0.0,
            "est_dy": 0.0,
        }
    ]
    assert tile_features_by_key(records) == {}


def _training_row(sample_id, scenario, advantage, incumbent_value):
    row = {
        "id": sample_id,
        "scenario": scenario,
        "aligned_loss_advantage": advantage,
    }
    for name in SAFE_REGISTRATION_FEATURE_NAMES:
        row[name] = incumbent_value
    return row


def test_screening_accepts_discriminative_candidate() -> None:
    tile_features = {}
    training_rows = []
    for index in range(40):
        safe = index % 2 == 0
        key = (f"img_{index}", "shift:s3:t2")
        # Dispersion cleanly separates safe from unsafe (inverted sign).
        tile_features[key] = {
            "tile_max_pairwise_distance": 0.0 if safe else 8.0,
            "tile_dispersion_radius": 0.0 if safe else 3.0,
            "tile_agreement_within_2px": 1.0 if safe else 0.25,
            "tile_agreement_within_4px": 1.0 if safe else 0.5,
            "tile_mean_offset_magnitude": 8.0,
            "tile_magnitude_spread": 0.0,
        }
        training_rows.append(
            _training_row(
                key[0],
                key[1],
                0.5 if safe else -0.5,
                # Incumbents carry no information here.
                0.0,
            )
        )
    decision, joined = screen_features(
        tile_features,
        training_rows,
        benefit_margin=0.0,
        minimum_auc=0.60,
    )
    assert decision["rows_joined"] == 40
    assert decision["rows_unmatched"] == 0
    assert decision["safe_rate"] == pytest.approx(0.5)
    assert decision["inclusion_bar"] == pytest.approx(0.60)
    assert "tile_max_pairwise_distance" in decision["accepted_candidates"]
    assert (
        decision["candidate_features"]["tile_max_pairwise_distance"][
            "direction"
        ]
        == -1
    )
    # Constant candidates must not be accepted.
    assert "tile_magnitude_spread" not in decision["accepted_candidates"]
    assert (
        decision["next_action"]
        == "p7_5_add_accepted_tile_features_new_protocol"
    )
    assert len(joined) == 40


def test_screening_rejects_uninformative_candidates() -> None:
    tile_features = {}
    training_rows = []
    for index in range(40):
        safe = index % 2 == 0
        key = (f"img_{index}", "shift:s3:t2")
        tile_features[key] = {
            name: 1.0 for name in TILE_FEATURE_NAMES
        }
        training_rows.append(
            _training_row(key[0], key[1], 0.5 if safe else -0.5, 0.0)
        )
    decision, _ = screen_features(
        tile_features,
        training_rows,
        benefit_margin=0.0,
        minimum_auc=0.60,
    )
    assert decision["accepted_candidates"] == []
    assert (
        decision["next_action"]
        == "p7_5_drop_tile_consistency_use_other_evidence"
    )


def test_screening_requires_a_join() -> None:
    with pytest.raises(RuntimeError):
        screen_features(
            {},
            [_training_row("a", "shift:s1:t2", 0.1, 0.0)],
            benefit_margin=0.0,
            minimum_auc=0.60,
        )
