"""P24: RoMa enters Gate 1 by protocol after the P23 screen stopped."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tools.run_p23_whu_strong_localizer import (
    load_p24_config,
    row_group,
    row_group_summary,
    selected_for_p24,
    selected_from_screen,
)

ROOT = Path(__file__).resolve().parents[1]
BANDS = [("rows 00-10", 0, 10), ("rows 11-21", 11, 21), ("rows 22-28", 22, 28)]


def _screen(tmp_path: Path, name: str, verdict: str, selected=None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "screen_decision.json").write_text(
        json.dumps({"verdict": verdict, "selected": selected}), encoding="utf-8"
    )
    return directory


P24 = {
    "protocol_doc": "docs/P24_WHU_ROMA_GATE1_PROTOCOL.md",
    "p23_screen_run": "20260916T133941Z",
    "p23_screen_verdict": "stop_no_candidate_tracks",
    "selected": "roma_outdoor",
}


def test_p24_names_roma_only_after_the_stopped_screen(tmp_path) -> None:
    screen = _screen(tmp_path, "20260916T133941Z", "stop_no_candidate_tracks")
    selected, authorisation = selected_for_p24(P24, screen)
    assert selected == "roma_outdoor"
    assert authorisation["by"] == "p24_protocol"
    assert authorisation["p23_screen_verdict"] == "stop_no_candidate_tracks"
    other_run = _screen(tmp_path, "20260917T000000Z", "stop_no_candidate_tracks")
    with pytest.raises(SystemExit, match="screen run"):
        selected_for_p24(P24, other_run)
    passed = _screen(tmp_path, "20260918T000000Z", "proceed_to_gate1", "roma_outdoor")
    with pytest.raises(SystemExit, match="verdict"):
        selected_for_p24({**P24, "p23_screen_run": passed.name}, passed)


def test_p23_gate1_still_requires_a_passing_screen(tmp_path) -> None:
    stopped = _screen(tmp_path, "a", "stop_no_candidate_tracks")
    with pytest.raises(SystemExit):
        selected_from_screen(stopped)
    passed = _screen(tmp_path, "b", "proceed_to_gate1", "roma_outdoor")
    assert selected_from_screen(passed)[0] == "roma_outdoor"


def test_row_bands_cover_the_strip_without_overlap() -> None:
    assert [row_group(row, BANDS) for row in (0, 10, 11, 21, 22, 28)] == [
        "rows 00-10", "rows 00-10", "rows 11-21", "rows 11-21", "rows 22-28", "rows 22-28"
    ]
    assert row_group(29, BANDS) is None
    assert row_group(3, None) is None


class _Metric:
    def __init__(self, f1: float) -> None:
        self.value = f1

    def compute(self) -> dict[str, float]:
        return {"f1": self.value}


def test_row_band_summary_reports_both_recoveries() -> None:
    values = {"raw": 0.40, "oracle": 0.90, "p7c_estimated": 0.50, "p7c_gated": 0.50,
              "estimated": 0.80, "estimated_gated": 0.80}
    clean = {path: 0.90 for path in values}
    metrics = {
        "rows 11-21": {
            "clean:s0:common": {path: _Metric(v) for path, v in clean.items()},
            "physical_shift:s3:t2": {path: _Metric(v) for path, v in values.items()},
        }
    }
    summary = row_group_summary(metrics, {"rows 11-21": 66})["rows 11-21"]
    assert summary["tiles"] == 66
    assert summary["recovery"] == {"physical_shift:s3:t2": pytest.approx(0.8)}
    assert summary["p7c_recovery"] == {"physical_shift:s3:t2": pytest.approx(0.2)}
    assert "clean:s0:common" not in summary["recovery"]


def test_p24_config_is_chained_to_the_p23_and_p22_configs() -> None:
    config = yaml.safe_load((ROOT / "configs/p24_whu_roma_gate1.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "preregistered"
    assert hashlib.sha256((ROOT / config["p23_config"]).read_bytes()).hexdigest() == config["p23_config_sha256"]
    assert (ROOT / config["protocol_doc"]).is_file()
    assert config["selected"] == "roma_outdoor"
    bands = [(label, first, last) for label, first, last in config["gate1"]["row_groups"]]
    rows = [row for _, first, last in bands for row in range(first, last + 1)]
    assert rows == list(range(29))


def test_load_p24_config_refuses_a_changed_p23_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    loaded = load_p24_config(ROOT / "configs/p24_whu_roma_gate1.yaml")
    assert loaded[0]["selected"] == "roma_outdoor"
    assert loaded[1]["status"] == "preregistered"
    config = yaml.safe_load((ROOT / "configs/p24_whu_roma_gate1.yaml").read_text(encoding="utf-8"))
    config["p23_config_sha256"] = "0" * 64
    path = tmp_path / "p24.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="P24 expects"):
        load_p24_config(path)
    config = yaml.safe_load((ROOT / "configs/p24_whu_roma_gate1.yaml").read_text(encoding="utf-8"))
    config["selected"] = "p7c"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a P23 candidate"):
        load_p24_config(path)
