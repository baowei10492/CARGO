import pytest

from tools.analyze_threshold_sensitivity import (
    clean_safe_ceiling,
    render_markdown,
    sensitivity_table,
    summarise_tau,
)


def _scenarios(
    clean_accepted: float,
    clean_harmful: float = 0.0,
    shift_accepted: float = 30.0,
    shift_harmful: float = 2.0,
) -> dict:
    return {
        "clean:s0:common": {
            "q_candidates": 4.0,
            "q_benefit": -1.0,
            "accepted": clean_accepted,
            "accepted_harmful": clean_harmful,
            "accepted_benefit": -0.5 if clean_accepted else 0.0,
        },
        **{
            key: {
                "q_candidates": 20.0,
                "q_benefit": 10.0,
                "accepted": shift_accepted / 3.0,
                "accepted_harmful": shift_harmful / 3.0,
                "accepted_benefit": 8.0,
            }
            for key in ("shift:s1:t2", "shift:s2:t2", "shift:s3:t2")
        },
    }


def test_summary_aggregates_shift_scenarios_only() -> None:
    summary = summarise_tau(_scenarios(clean_accepted=1.0))
    assert summary["shift_candidates"] == pytest.approx(60.0)
    assert summary["shift_accepted"] == pytest.approx(30.0)
    assert summary["shift_adoption_rate"] == pytest.approx(0.5)
    # captured 24 of 30 available benefit across the three shift rows
    assert summary["shift_captured_fraction"] == pytest.approx(0.8)
    assert summary["clean_accepted"] == pytest.approx(1.0)


def test_harmful_count_includes_clean_scenario() -> None:
    summary = summarise_tau(
        _scenarios(clean_accepted=1.0, clean_harmful=1.0)
    )
    assert summary["total_harmful_adoptions"] == pytest.approx(3.0)


def test_clean_ceiling_is_the_smallest_tau_without_clean_adoption() -> None:
    sweep = {
        0.80: _scenarios(clean_accepted=2.0),
        0.90: _scenarios(clean_accepted=1.0),
        0.95: _scenarios(clean_accepted=0.0),
        0.99: _scenarios(clean_accepted=0.0),
    }
    assert clean_safe_ceiling(sweep) == pytest.approx(0.95)


def test_clean_ceiling_is_none_when_clean_is_never_safe() -> None:
    sweep = {0.5: _scenarios(clean_accepted=1.0)}
    assert clean_safe_ceiling(sweep) is None


def test_table_marks_the_frozen_operating_point_once() -> None:
    sweep = {
        round(0.05 * index, 2): _scenarios(
            clean_accepted=1.0 if index < 18 else 0.0
        )
        for index in range(21)
    }
    rows = sensitivity_table(
        sweep, frozen_tau=0.9082, reference_taus=(0.0, 0.5, 0.95)
    )
    frozen = [row for row in rows if row["is_frozen_operating_point"]]
    assert len(frozen) == 1
    assert frozen[0]["tau"] == pytest.approx(0.90)
    assert rows == sorted(rows, key=lambda row: row["tau"])


def test_markdown_flags_the_frozen_row() -> None:
    sweep = {
        0.50: _scenarios(clean_accepted=1.0),
        0.90: _scenarios(clean_accepted=0.0),
    }
    rows = sensitivity_table(
        sweep, frozen_tau=0.90, reference_taus=(0.50,)
    )
    text = render_markdown(
        {"task_router": {"rows": rows, "frozen_tau": 0.90}}
    )
    assert "(frozen)" in text
    assert text.count("(frozen)") == 1
