"""P27 ablation and efficiency tables are built from the ablation JSON files only."""
from __future__ import annotations

import json

import pytest

from tools.p27_ablation_tables import build_ablation_tables, selector_cells, selector_summary


def _variant(clean_drop: float, gain: float) -> dict:
    return {"tau": 0.5, "clean_drop": clean_drop, "gain_1": gain / 3, "gain_2": gain / 2, "gain_3": gain}


def _selector_run(root, row, datasets, drops) -> None:
    run = root / "selector" / row / "20260918T000000Z"
    run.mkdir(parents=True)
    entry = {"reference": {"estimated": _variant(0.03, 0.30), "gated": _variant(0.01, 0.29)},
             "variants": {"full": _variant(drops, 0.31), "registration_only": _variant(0.001, 0.25),
                          "task_only": _variant(0.008, 0.30), "without_forward_backward": _variant(0.0, 0.30)},
             "full_reproduces_main": {"tau_equal": True, "max_probability_difference": 0.0}}
    (run / "selector_ablation.json").write_text(json.dumps({"datasets": {d: entry for d in datasets}}),
                                                encoding="utf-8")


def test_selector_summary_and_tables(tmp_path) -> None:
    _selector_run(tmp_path, "dinov3", ("LEVIR-CD", "DSIFN-CD", "WHU-CD"), 0.002)
    _selector_run(tmp_path, "bit_whu", ("WHU-CD",), -0.004)
    cells = selector_cells(tmp_path / "selector")
    assert [(c["detector"], c["dataset"]) for c in cells] == [("DINOv3", "LEVIR-CD"), ("DINOv3", "DSIFN-CD"),
                                                              ("DINOv3", "WHU-CD"), ("BIT", "WHU-CD")]
    summary = {s["variant"]: s for s in selector_summary(cells)}
    assert summary["always"]["unsafe_cells"] == 4 and summary["full"]["unsafe_cells"] == 0
    assert summary["full"]["mean_clean_change"] == pytest.approx(-(3 * 0.002 - 0.004) / 4)
    assert summary["full"]["worst_clean_change"] == pytest.approx(-0.002)
    assert summary["task_only"]["unsafe_cells"] == 4 and summary["full"]["mean_gain"][2] == pytest.approx(0.31)

    localizer = tmp_path / "localizer" / "test"
    localizer.mkdir(parents=True)
    entry = {"localization_success": {"0": None, "1": 0.7, "2": 0.6, "3": 0.5},
             "clean_drop": {"estimated": 0.02, "estimated_gated": 0.01},
             "gain": {k: {"estimated": 0.1 * int(k)} for k in ("1", "2", "3")}}
    (localizer / "p27_full.json").write_text(json.dumps({"datasets": {"LEVIR-CD": entry}}), encoding="utf-8")
    efficiency = tmp_path / "efficiency"
    efficiency.mkdir()
    timing = {k: {"median_ms": 1.0} for k in ("localizer_both_directions", "warp", "selector", "detector_two_passes")}
    (efficiency / "20260917T000000Z.json").write_text(json.dumps({
        "crop_size": 256, "device": "GPU", "localizer_parameters": {"total": 1_000_000, "trainable": 200_000},
        "roma_similarity_ms": {"median_ms": 290.0},
        "rows": {"DINOv3 (ours)": {"detector_parameters": {"total": 29_000_000}, "selector_parameters": 1249,
                                   "timing": {**timing, "total_median_ms": 4.0}}}}), encoding="utf-8")
    out = tmp_path / "tables"
    assert build_ablation_tables(tmp_path, out) == ["localizer", "selector (4 cells)", "efficiency"]
    assert "| LEVIR-CD | P27 localizer (full) | 70.00 | 60.00 | 50.00 | -2.00 | +10.00 | +20.00 | +30.00 |" in \
        (out / "table_p27_localizer_ablation.md").read_text(encoding="utf-8")
    assert "| selector (19 features) |" in (out / "table_p27_selector_ablation.md").read_text(encoding="utf-8")
    assert "| DINOv3 (ours) | 29.0 | 1249 |" in (out / "table_p27_efficiency.md").read_text(encoding="utf-8")


def test_a_selector_that_does_not_reproduce_is_refused(tmp_path) -> None:
    _selector_run(tmp_path, "bit", ("LEVIR-CD", "DSIFN-CD"), 0.0)
    path = tmp_path / "selector" / "bit" / "20260918T000000Z" / "selector_ablation.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["datasets"]["DSIFN-CD"]["full_reproduces_main"]["tau_equal"] = False
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(SystemExit, match="reproduce"):
        selector_cells(tmp_path / "selector")
