"""Overnight session (record 192.71): per-crop registration errors, dose-response, input swap, selector seeds."""
from __future__ import annotations

import numpy as np
import torch

from test_p27_f1 import _tiny_localizer
from test_p30_evaluation import _setup


def test_registration_errors_keep_every_crop(tmp_path, monkeypatch) -> None:
    from tools.p31_registration_errors import anchor, crop_errors, success_by_level
    from tools.run_p27_selector import harness, open_dataset

    tool, p30, evaluation, _, _, _ = _setup(tmp_path, monkeypatch)
    h = harness(evaluation)
    rows_by_dataset = {}
    for spec in evaluation["datasets"]:
        rows = crop_errors(spec, evaluation, _tiny_localizer(), margin=64)
        crops = tool.margin_crops(open_dataset(spec, spec["split"]), spec, h["size"], 64)
        assert len(rows) == len(crops) * (1 + len(h["levels"]))
        assert all(r["ate_raw"] is not None and 0.0 <= r["change_fraction"] <= 1.0 for r in rows)
        rows_by_dataset[spec["name"]] = rows
    success = {name: success_by_level(rows) for name, rows in rows_by_dataset.items()}
    reference = {"datasets": {name: {"localization_success": dict(levels), "crops": 1000}
                              for name, levels in success.items()}}
    assert anchor("v2_full", success, reference)["passed"]
    name = next(iter(reference["datasets"]))
    reference["datasets"][name]["localization_success"]["3"] += 0.005         # 5 of 1000 crops: allowed
    check = anchor("v2_full", success, reference)
    assert check["passed"] and check["max_flipped_crops"] == 5
    reference["datasets"][name]["localization_success"]["3"] += 0.01          # 15 of 1000: an instrument error
    check = anchor("v2_full", success, reference)
    assert not check["passed"] and check["over_limit"] == [f"{name}/3"]
    assert anchor("roma_similarity", success, reference)["passed"]            # RoMa: up to 20 of 1000


def test_dose_response_pastes_only_inside_the_window(tmp_path, monkeypatch) -> None:
    from tools.p31_dose_response import dataset_rows, pasted_tile, rectangles, summarize
    from tools.run_p27_selector import harness

    for fraction in (0.1, 0.3, 0.5):
        covered = rectangles("crop|y0064x0064", fraction, 256, 7)
        assert fraction <= covered.mean() < fraction + 0.1
        assert np.array_equal(covered, rectangles("crop|y0064x0064", fraction, 256, 7))
    assert not rectangles("crop", 0.0, 256, 7).any()
    tile = np.zeros((512, 512, 3), np.uint8)
    source = np.full((512, 512, 3), 200, np.uint8)
    covered = rectangles("crop", 0.3, 256, 7)
    out = pasted_tile(tile, source, 64, 128, covered)
    changed = out.max(axis=2) > 0
    assert changed.sum() == covered.sum() and changed[64:320, 128:384].sum() == covered.sum()

    _, _, evaluation, _, _, _ = _setup(tmp_path, monkeypatch)
    spec = evaluation["datasets"][0]
    rows = dataset_rows("v2_full", spec, evaluation, _tiny_localizer(), margin=64, crops_per_dataset=3,
                        fractions=(0.0, 0.3))
    assert len(rows) == 3 * 2 * 2
    assert {int(r["level"]) for r in rows} == {0, harness(evaluation)["levels"][-1]}
    assert all(r["coverage"] == 0.0 for r in rows if r["fraction"] == 0.0)
    assert all(r["coverage"] >= 0.3 for r in rows if r["fraction"] == 0.3)
    table = summarize(rows)
    assert {e["dataset"] for e in table} == {spec["name"], "all"}
    assert all(0.0 <= e["success"] <= 1.0 for e in table)


def test_input_swap_normal_view_is_the_raw_clean_score(tmp_path, monkeypatch) -> None:
    from tools.p31_input_swap import swap_scores
    from tools.run_p27_selector import harness, open_dataset

    tool, _, evaluation, _, groups, _ = _setup(tmp_path, monkeypatch)
    h = harness(evaluation)
    for spec in evaluation["datasets"]:
        detector = groups["original"][spec["name"]]
        scores = swap_scores(spec, evaluation, detector, torch.device("cpu"), margin=64)
        crops = tool.margin_crops(open_dataset(spec, spec["split"]), spec, h["size"], 64)
        raw = tool.raw_scores(open_dataset(spec, spec["split"]), spec, crops, detector, h, torch.device("cpu"))
        assert abs(scores["f1"]["normal"] - raw["0"]) < 1e-9
        assert scores["f1"]["normal"] > 0.0
        assert scores["f1"]["same_date"] == 0.0        # the toy detector differences the two images


def test_selector_seed_spread() -> None:
    from tools.p31_selector_seeds import spread

    seeds = (3407, 1, 2)
    cells = {f"row/{k}": {"seeds": {str(s): {"clean_drop": 0.001 * k + 0.001 * i, "gain_1": 0.01, "gain_2": 0.02,
                                             "gain_3": 0.1 + 0.01 * i} for i, s in enumerate(seeds)}}
             for k in range(3)}
    result = spread(cells, seeds)
    assert set(result["per_seed"]) == {"3407", "1", "2"}
    assert abs(result["over_seeds"]["gain_3_mean"]["mean"] - 0.11) < 1e-9
    assert result["over_seeds"]["gain_3_mean"]["std"] > 0
    assert abs(result["per_cell"]["row/0"]["gain_3"]["std"] - 0.01) < 1e-9
