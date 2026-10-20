import json
from pathlib import Path

import pytest

from tools.summarize_native_window_probe import classify, read_pck1


def _classify(native: dict[int, float], downsampled: dict[int, float]) -> str:
    return classify(
        native,
        downsampled,
        baseline_scale=512,
        recovery_min=0.70,
        flat_max=0.10,
    )["verdict"]


# The downsampled reference is the measured original probe curve from
# main record 77, where coverage and GSD moved together.
REFERENCE = {512: 0.083, 1024: 0.450, 2048: 0.933, 3072: 1.000}


def test_recovery_at_native_resolution_means_coverage() -> None:
    native = {512: 0.083, 1024: 0.400, 2048: 0.880, 3072: 0.960}
    assert _classify(native, REFERENCE) == "coverage"


def test_flat_curve_at_native_resolution_means_gsd() -> None:
    native = {512: 0.083, 1024: 0.090, 2048: 0.110, 3072: 0.120}
    assert _classify(native, REFERENCE) == "gsd"


def test_partial_recovery_is_reported_as_both() -> None:
    native = {512: 0.083, 1024: 0.180, 2048: 0.300, 3072: 0.400}
    verdict = _classify(native, REFERENCE)
    assert verdict == "both"


def test_half_of_the_reference_gain_counts_as_coverage() -> None:
    # 0.083 -> 0.60 recovers 56% of the reference gain, below the
    # absolute recovery bar but above the relative one.
    native = {512: 0.083, 1024: 0.300, 2048: 0.500, 3072: 0.600}
    assert _classify(native, REFERENCE) == "coverage"


def test_report_records_gains_and_thresholds() -> None:
    native = {512: 0.083, 1024: 0.400, 2048: 0.880, 3072: 0.960}
    report = classify(
        native,
        REFERENCE,
        baseline_scale=512,
        recovery_min=0.70,
        flat_max=0.10,
    )
    assert report["largest_window_px"] == 3072
    assert report["native_gain"] == pytest.approx(0.877)
    assert report["downsampled_gain"] == pytest.approx(0.917)
    assert report["recovered_fraction_of_downsampled_gain"] == pytest.approx(
        0.877 / 0.917
    )
    assert report["thresholds"] == {"recovery_min": 0.70, "flat_max": 0.10}


def test_missing_baseline_is_rejected() -> None:
    with pytest.raises(KeyError):
        classify(
            {1024: 0.4, 2048: 0.9},
            REFERENCE,
            baseline_scale=512,
            recovery_min=0.70,
            flat_max=0.10,
        )


def test_read_pck1_uses_the_raw_estimate(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "per_shift": {
                    "shift:s2:t2": {"raw_pck1": 0.42, "delta_pck1": 0.99}
                }
            }
        ),
        encoding="utf-8",
    )
    assert read_pck1(path, "shift:s2:t2") == pytest.approx(0.42)
