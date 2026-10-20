import pytest

from tools.benchmark_p10_efficiency import _percentile, _selected_count


def test_selected_count_is_deterministic_and_clamped() -> None:
    assert _selected_count(8, 0.0) == 0
    assert _selected_count(8, 0.25) == 2
    assert _selected_count(8, 1.0) == 8
    with pytest.raises(ValueError):
        _selected_count(8, 1.1)


def test_percentile_interpolates() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 0.5) == pytest.approx(2.5)
    assert _percentile(values, 1.0) == 4.0
