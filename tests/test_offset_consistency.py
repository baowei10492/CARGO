import pytest

from tools.diagnose_transfer_offset_consistency import (
    offset_consistency_summary,
)


def _record(sample_id, scenario, est, gt=(0.0, 0.0), q=True):
    return {
        "id": sample_id,
        "scenario": scenario,
        "est_dx": est[0],
        "est_dy": est[1],
        "gt_dx": gt[0],
        "gt_dy": gt[1],
        "q_gate": q,
    }


def _build_records(intrinsic_by_id, noise=0.0):
    """Localizer that predicts injected + per-image intrinsic offset."""

    shifts = {
        "shift:s1:t2": (2.0, 0.0),
        "shift:s2:t2": (0.0, 4.0),
        "shift:s3:t2": (8.0, 0.0),
    }
    records = []
    for sample_id, intrinsic in intrinsic_by_id.items():
        records.append(
            _record(
                sample_id,
                "clean:s0:common",
                (intrinsic[0] + noise, intrinsic[1]),
            )
        )
        for scenario, shift in shifts.items():
            records.append(
                _record(
                    sample_id,
                    scenario,
                    (
                        shift[0] + intrinsic[0],
                        shift[1] + intrinsic[1],
                    ),
                    gt=shift,
                )
            )
    return records


def test_summary_detects_stable_intrinsic_offset() -> None:
    intrinsic = {
        f"img_{index}": (2.0, 2.0) if index % 2 else (-2.0, 0.0)
        for index in range(10)
    }
    summary = offset_consistency_summary(
        _build_records(intrinsic),
        clean_scenario="clean:s0:common",
        shift_scenarios=[
            "shift:s1:t2",
            "shift:s2:t2",
            "shift:s3:t2",
        ],
    )
    shift8 = summary["per_shift"]["shift:s3:t2"]
    # Raw PCK fails (intrinsic offset ~2.8 px shifts every estimate)...
    assert shift8["raw_pck1"] == pytest.approx(0.0)
    # ...but delta-PCK and residual consistency are perfect.
    assert shift8["delta_pck1"] == pytest.approx(1.0)
    assert shift8["residual_matches_clean_pck1"] == pytest.approx(1.0)
    pairwise = summary["residual_pairwise_consistency"]
    assert pairwise["shift:s1:t2|shift:s3:t2"]["pck1"] == pytest.approx(
        1.0
    )
    assert summary["clean_conditional_offset"][
        "mean_magnitude"
    ] == pytest.approx((5 * (8.0**0.5) + 5 * 2.0) / 10.0)


def test_summary_detects_collapsed_descriptor() -> None:
    # Estimates unrelated to the injected shift: constant prediction.
    records = []
    for index in range(10):
        sample_id = f"img_{index}"
        records.append(
            _record(sample_id, "clean:s0:common", (2.0, 0.0))
        )
        for scenario, gt in (
            ("shift:s1:t2", (2.0, 0.0)),
            ("shift:s2:t2", (0.0, 4.0)),
            ("shift:s3:t2", (8.0, 0.0)),
        ):
            records.append(
                _record(sample_id, scenario, (2.0, 0.0), gt=gt)
            )
    summary = offset_consistency_summary(
        records,
        clean_scenario="clean:s0:common",
        shift_scenarios=[
            "shift:s1:t2",
            "shift:s2:t2",
            "shift:s3:t2",
        ],
    )
    shift8 = summary["per_shift"]["shift:s3:t2"]
    assert shift8["delta_pck1"] == pytest.approx(0.0)
    # Residuals differ across severities because the injected shifts
    # differ while the estimate stays constant.
    pairwise = summary["residual_pairwise_consistency"]
    assert pairwise["shift:s2:t2|shift:s3:t2"]["pck1"] == pytest.approx(
        0.0
    )


def test_summary_requires_clean_records() -> None:
    with pytest.raises(RuntimeError):
        offset_consistency_summary(
            [_record("a", "shift:s1:t2", (0.0, 0.0))],
            clean_scenario="clean:s0:common",
            shift_scenarios=["shift:s1:t2"],
        )
