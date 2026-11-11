import pytest

from tools.summarize_p8_whu_headroom import (
    CLEAN_KEY,
    headroom_decision,
)


def _rows(clean, s1, s2, s8):
    return [
        {"scenario": CLEAN_KEY, "f1": str(clean), "valid_pixels": "1000"},
        {"scenario": "shift:s1:t2", "f1": str(s1), "valid_pixels": "1000"},
        {"scenario": "shift:s2:t2", "f1": str(s2), "valid_pixels": "1000"},
        {"scenario": "shift:s3:t2", "f1": str(s8), "valid_pixels": "1000"},
    ]


def test_headroom_gate_passes_like_levir() -> None:
    # LEVIR-like: large recoverable headroom at severity 3.
    raw = _rows(0.9032, 0.8672, 0.8053, 0.6793)
    oracle = _rows(0.9032, 0.9032, 0.9031, 0.9033)
    decision = headroom_decision(raw, oracle, shift8_headroom_min=0.05)
    assert decision["decisive_gate_passed"] is True
    assert decision["input_oracle_headroom"][
        "shift:s3:t2"
    ] == pytest.approx(0.2240, abs=1e-4)
    assert (
        decision["next_action"]
        == "p8_continue_chain_dual_path_oracle_then_p7_1b"
    )


def test_headroom_gate_fails_like_s2looking() -> None:
    # S2Looking-like: severity-3 headroom is only ~0.033.
    raw = _rows(0.7107, 0.7071, 0.6983, 0.6782)
    oracle = _rows(0.7107, 0.7108, 0.7108, 0.7112)
    decision = headroom_decision(raw, oracle, shift8_headroom_min=0.05)
    assert decision["decisive_gate_passed"] is False
    assert decision["input_oracle_headroom"][
        "shift:s3:t2"
    ] == pytest.approx(0.0330, abs=1e-4)
    assert (
        decision["next_action"]
        == "p8_stop_whu_joins_domain_boundary_cases"
    )


def test_clean_mismatch_is_rejected() -> None:
    raw = _rows(0.90, 0.86, 0.80, 0.67)
    oracle = _rows(0.88, 0.90, 0.90, 0.90)
    with pytest.raises(ValueError):
        headroom_decision(raw, oracle, shift8_headroom_min=0.05)


def test_missing_scenarios_are_rejected() -> None:
    raw = _rows(0.90, 0.86, 0.80, 0.67)
    with pytest.raises(ValueError):
        headroom_decision(
            raw,
            [
                {"scenario": CLEAN_KEY, "f1": "0.90"},
                {"scenario": "shift:s1:t2", "f1": "0.90"},
            ],
            shift8_headroom_min=0.05,
        )
    with pytest.raises(ValueError):
        headroom_decision(
            [{"scenario": "shift:s1:t2", "f1": "0.9"}],
            raw,
            shift8_headroom_min=0.05,
        )


def test_duplicate_scenario_is_rejected() -> None:
    rows = _rows(0.9, 0.86, 0.8, 0.67)
    rows.append({"scenario": "shift:s1:t2", "f1": "0.5"})
    with pytest.raises(ValueError):
        headroom_decision(rows, rows, shift8_headroom_min=0.05)
