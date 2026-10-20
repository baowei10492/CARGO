"""The mixing in tools/p32_mixed_distribution.py: one record per crop, and the same one every time.

The mixed set is built by picking, for each crop, the record of the level it was
assigned. Two mistakes would pass unnoticed: taking more than one level of a crop
(which inflates the set and double-counts pixels), and an assignment that is not
stable, which would silently re-draw the states when the two P32 detectors are added
to the same table.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.p32_mixed_distribution import assignment, scores, state_of

LEVELS = (0, 1, 3)
STATES = [{"name": "aligned", "level": 0}, {"name": "small", "level": 1}, {"name": "large", "level": 3}]


def cell(crops: int = 300) -> SimpleNamespace:
    names = [f"whu_test_r00c00|y{i:04d}x0064" for i in range(crops)]
    return SimpleNamespace(
        crop=np.array([name for name in names for _ in range(4)]),
        level=np.array([level for _ in names for level in (0, 1, 2, 3)]),
    )


def test_exactly_one_record_per_crop_and_it_is_the_assigned_level() -> None:
    fake = cell()
    keep = assignment(fake, seed=3407, levels=LEVELS)

    assert keep.sum() == len(set(fake.crop.tolist()))
    for crop in set(fake.crop.tolist()):
        chosen = fake.level[keep & (fake.crop == crop)]
        assert chosen.shape == (1,)
        assert int(chosen[0]) == LEVELS[state_of(crop, 3407, len(LEVELS))]


def test_the_assignment_does_not_move_between_calls_or_rows() -> None:
    """Adding SNUNet and BiFA later must not re-draw the states."""

    first = assignment(cell(), seed=3407, levels=LEVELS)
    second = assignment(cell(), seed=3407, levels=LEVELS)
    assert np.array_equal(first, second)

    # A different seed would be a different set; the config's seed is written once.
    assert not np.array_equal(first, assignment(cell(), seed=1, levels=LEVELS))


def test_the_three_states_partition_the_mixed_set() -> None:
    fake = cell()
    keep = assignment(fake, seed=3407, levels=LEVELS)
    counts = np.tile(np.array([4.0, 1.0, 1.0]), (len(fake.level), 1))

    result = scores(fake, keep, {"raw": counts}, STATES, seed=3407)

    assert sum(entry["crops"] for entry in result["by_state"].values()) == int(keep.sum())
    # Every count row is identical here, so pooling cannot change the F1: 2*4/(2*4+1+1).
    # (The metric's eps of 1e-12 keeps it just short of the exact value.)
    assert result["mixed"]["raw"] == pytest.approx(0.8, abs=1e-9)
