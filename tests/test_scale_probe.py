import numpy as np
import pytest
from PIL import Image

from tools.build_whu_scale_probe import (
    blocks_for_scale,
    build_probe,
    downsample_block,
    enumerate_blocks,
    select_evenly,
)
from tools.summarize_scale_probe import scale_probe_decision


def test_enumerate_blocks_respects_region() -> None:
    origins = enumerate_blocks(64, 48, 16, 16)
    assert len(origins) == 4 * 3
    assert all(left + 16 <= 64 and top + 16 <= 48 for left, top in origins)
    assert enumerate_blocks(15, 48, 16, 16) == []
    with pytest.raises(ValueError):
        enumerate_blocks(64, 48, 0, 16)


def test_select_evenly_is_deterministic_and_spread() -> None:
    items = list(range(100))
    picked = select_evenly(items, 10)
    assert len(picked) == 10
    assert picked == select_evenly(items, 10)
    assert picked[0] == 0 and picked[-1] == 90
    assert select_evenly([1, 2], 10) == [1, 2]


def test_blocks_for_scale_densifies_only_when_sparse() -> None:
    # Plenty of non-overlapping blocks: stride stays = scale.
    blocks, stride = blocks_for_scale(640, 640, 16, max_blocks=60)
    assert stride == 16
    assert len(blocks) == 60
    # Sparse: falls back to half stride.
    blocks, stride = blocks_for_scale(96, 96, 64, max_blocks=60)
    assert stride == 32
    assert len(blocks) == 4


def test_downsample_block_is_exact_block_average() -> None:
    block = np.zeros((4, 4, 3), dtype=np.uint8)
    block[:2, :2] = 100
    block[:2, 2:] = 200
    block[2:, :2] = 40
    block[2:, 2:] = 80
    out = downsample_block(block, 2)
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == 100
    assert out[0, 1, 0] == 200
    assert out[1, 0, 0] == 40
    assert out[1, 1, 0] == 80


def _fake_raw(root, width=96, height=64):
    two = root / "1. The two-period image data"
    rng = np.random.default_rng(0)
    for year in ("2012", "2016"):
        path = (
            two / year / "whole_image" / "train" / "image" / "img.tif"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        ).save(path)
    return root


def test_build_probe_writes_scales_and_index(tmp_path) -> None:
    raw = _fake_raw(tmp_path / "raw")
    out = tmp_path / "probe"
    summary = build_probe(
        raw,
        out,
        scales=[16, 32],
        output_size=16,
        max_blocks=8,
        x_max=64,
        y_max=48,
        dry_run=False,
    )
    assert set(summary["scales"]) == {"scale0016", "scale0032"}
    assert summary["scales"]["scale0016"]["coverage_m"] == pytest.approx(
        16 * 0.075
    )
    for split in ("scale0016", "scale0032"):
        a_files = sorted((out / split / "A").glob("*.png"))
        assert a_files
        assert len(a_files) == summary["scales"][split]["blocks"]
        tile = Image.open(a_files[0])
        assert tile.size == (16, 16)
        assert (out / split / "label" / a_files[0].name).is_file()
    assert (out / "probe_index.csv").is_file()
    # Region limit: no block may cross x_max even downsampled.
    import csv

    with (out / "probe_index.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert int(row["left"]) + int(row["scale"]) <= 64
        assert int(row["top"]) + int(row["scale"]) <= 48


def test_build_probe_rejects_non_integer_ratio(tmp_path) -> None:
    raw = _fake_raw(tmp_path / "raw")
    with pytest.raises(ValueError):
        build_probe(
            raw,
            tmp_path / "probe",
            scales=[24],
            output_size=16,
            max_blocks=4,
            x_max=64,
            y_max=48,
            dry_run=True,
        )


def test_build_probe_dry_run_writes_nothing(tmp_path) -> None:
    raw = _fake_raw(tmp_path / "raw")
    out = tmp_path / "probe"
    build_probe(
        raw,
        out,
        scales=[16],
        output_size=16,
        max_blocks=4,
        x_max=64,
        y_max=48,
        dry_run=True,
    )
    assert not out.exists()


def _summary(s4_pck1, clean_q=0.4):
    return {
        "per_shift": {
            "shift:s2:t2": {
                "raw_pck1": s4_pck1,
                "raw_pck2": min(1.0, s4_pck1 + 0.1),
                "raw_mean_epe": 2.0,
                "samples": 60,
            },
            "shift:s3:t2": {
                "raw_pck1": s4_pck1,
                "raw_pck2": min(1.0, s4_pck1 + 0.1),
                "raw_mean_epe": 3.0,
                "samples": 60,
                "boundary_fraction": 0.5,
                "gt_boundary_fraction": 0.98,
                "excess_boundary_fraction": -0.48,
            },
        },
        "q_trigger_rate": {"clean:s0:common": clean_q},
    }


def _per_scale(values):
    scales = [512, 1024, 2048, 3072]
    return {
        f"scale{scale:04d}": {
            "scale_px": scale,
            "coverage_m": scale * 0.075,
            "summary": _summary(value),
        }
        for scale, value in zip(scales, values)
    }


def _decide(values, **kwargs):
    return scale_probe_decision(
        _per_scale(values),
        levir_reference=_summary(0.906),
        whu_native_reference=_summary(0.132),
        explain_min=kwargs.get("explain_min", 0.70),
        not_explain_max=kwargs.get("not_explain_max", 0.30),
        monotone_tolerance=kwargs.get("monotone_tolerance", 0.05),
    )


def test_monotone_recovery_reads_as_coverage_explanation() -> None:
    decision = _decide([0.13, 0.35, 0.60, 0.82])
    assert decision["verdict"] == "coverage_explains_the_failure"
    assert (
        decision["attribution"]
        == "applicability_condition_is_gsd_coverage_not_dataset"
    )
    assert decision["monotone_non_decreasing"]
    assert decision["levir_512_reference_s4_pck1"] == pytest.approx(0.906)


def test_flat_series_reads_as_dataset_property() -> None:
    decision = _decide([0.13, 0.15, 0.12, 0.14])
    assert decision["verdict"] == "coverage_does_not_explain_the_failure"
    assert (
        decision["attribution"] == "dataset_property_attribution_stands"
    )


def test_middle_band_is_partial() -> None:
    decision = _decide([0.13, 0.30, 0.42, 0.50])
    assert decision["verdict"] == "partial_coverage_effect"


def test_high_endpoint_without_monotonicity_is_partial() -> None:
    # Ends high but collapses in the middle: not a clean trend.
    decision = _decide([0.13, 0.60, 0.20, 0.75])
    assert not decision["monotone_non_decreasing"]
    assert decision["verdict"] == "partial_coverage_effect"


def test_probe_never_reopens_gates() -> None:
    for values in ([0.13, 0.35, 0.60, 0.82], [0.13, 0.15, 0.12, 0.14]):
        decision = _decide(values)
        assert decision["gates_reopened"] is False


def test_two_scales_minimum() -> None:
    with pytest.raises(ValueError):
        scale_probe_decision(
            {
                "scale0512": {
                    "scale_px": 512,
                    "coverage_m": 38.4,
                    "summary": _summary(0.13),
                }
            },
            levir_reference=None,
            whu_native_reference=None,
            explain_min=0.70,
            not_explain_max=0.30,
            monotone_tolerance=0.05,
        )
