"""Final measurements: same-crop border comparison, border-free qualitative panels, DiffRegCD timing loop."""
from __future__ import annotations

import numpy as np
import torch

from test_p27_f1 import _tiny_localizer
from test_p30_evaluation import _setup


def test_border_counts_reproduce_p30_rows(tmp_path, monkeypatch) -> None:
    from tools.p31_border_counts import GROUPS, VIEWS, anchor, run_dataset
    from tools.run_p27_selector import harness, open_dataset

    tool, p30, evaluation, _, groups, _ = _setup(tmp_path, monkeypatch)
    groups = {"original": groups["original"], "p29_augmented": groups["p29_augmented"],
              "p30_augmented": groups["p30_augmented"]}
    h = harness(evaluation)
    for spec in evaluation["datasets"]:
        name = spec["name"]
        detectors = {group: groups[group][name] for group in GROUPS}
        rows, f1 = run_dataset(spec, evaluation, detectors, margin=64, device=torch.device("cpu"))
        dataset = open_dataset(spec, spec["split"])
        crops = tool.margin_crops(dataset, spec, h["size"], 64)
        assert len(rows) == len(crops) * len(GROUPS) * (1 + 2 * len(h["levels"]))
        assert {row["view"] for row in rows} == set(VIEWS)
        reference = {group: tool.raw_scores(dataset, spec, crops, detectors[group], h, torch.device("cpu"))
                     for group in GROUPS}
        check = anchor(f1, reference, h["levels"])
        assert check["passed"] and check["max_abs_difference"] < 1e-9
        zero = [r for r in rows if r["view"] == "zero_border"]
        free = [r for r in rows if r["view"] == "border_free"]
        assert any((a["tp"], a["fp"], a["fn"]) != (b["tp"], b["fp"], b["fn"]) for a, b in zip(zero, free))
        broken = {group: dict(values) for group, values in reference.items()}
        broken["original"]["3"] += 0.01
        assert not anchor(f1, broken, h["levels"])["passed"]


def test_qualitative_views_are_border_free_and_match_the_record(tmp_path, monkeypatch) -> None:
    from tools.p27_qualitative import VIEWS, crop_f1
    from tools.p31_qualitative import example_views, parse_crop
    from tools.run_p27_selector import harness, open_dataset

    tool, p30, evaluation, _, groups, _ = _setup(tmp_path, monkeypatch)
    spec = evaluation["datasets"][0]
    h = harness(evaluation)
    dataset = open_dataset(spec, spec["split"])
    crop = tool.margin_crops(dataset, spec, h["size"], 64)[1]
    localizer = _tiny_localizer()
    detector = groups["original"][spec["name"]]
    records = tool.records_for(dataset, spec, [(crop, "val")], localizer, detector, h, torch.device("cpu"), "q")
    record = {**next(r for r in records if int(r["level"]) == 3), "selector_probability": 0.9, "tau": 0.5}
    assert parse_crop(record["crop_id"]) == (crop["sample_id"], crop["top"], crop["left"])
    images, scores = example_views(record, spec, evaluation, localizer, detector, torch.device("cpu"))
    assert set(images) == set(VIEWS)
    edges = np.concatenate([images["t2_input"][0], images["t2_input"][-1], images["t2_input"][:, 0], images["t2_input"][:, -1]])
    assert (edges.max(axis=1) > 0).all()                       # no zero border on the input shown
    assert abs(scores["without_correction"] - crop_f1(record, "raw")) < 1e-9
    assert abs(scores["localizer_only"] - crop_f1(record, "estimated")) < 1e-9
    assert np.array_equal(images["with_selector"], images["localizer_only"])  # probability 0.9 >= tau 0.5


def test_timing_loop() -> None:
    from tools.p31_diffregcd_timing import measure

    calls, syncs = [], []
    result = measure(lambda a, b: calls.append((a, b)), lambda i: (torch.zeros(1), torch.ones(1)),
                     warmup=3, repeats=20, synchronize=lambda: syncs.append(1))
    assert len(calls) == 23 and result["repeats"] == 20
    assert result["p10_ms"] <= result["median_ms"] <= result["p90_ms"]
    assert len(syncs) == 1 + 2 * 20
