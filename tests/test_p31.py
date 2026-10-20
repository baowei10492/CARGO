"""P31: config chain, the plug-in rows end to end, and the border-free localizer ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from test_p27_f1 import _tiny_localizer
from test_p30_evaluation import _setup

ROOT = Path(__file__).resolve().parents[1]


def test_config_chain() -> None:
    from tools.run_p31_plugins import load

    config, p30, evaluation, selector = load(ROOT / "configs/p31_border_free.yaml")
    assert sorted(config["rows"]) == ["bit", "cdlamba", "cdxlstm"]
    for row in config["rows"].values():
        assert sorted(row) == sorted(spec["name"] for spec in evaluation["datasets"])
        assert all(path in config["config_sha256"] for path in row.values())
    assert p30["crop_margin"] == 64 and config["ablation"]["split"] == "test"
    assert config["ablation"]["variants"][0] == "v2_full"


def test_plugin_row_end_to_end(tmp_path, monkeypatch) -> None:
    import tools.run_p31_plugins as p31

    tool, p30, evaluation, selector, groups, _ = _setup(tmp_path, monkeypatch)
    names = [spec["name"] for spec in evaluation["datasets"]]
    config = {"_sha256": "p31", "rows": {"toy": {name: "cfg" for name in names}}, "config_sha256": {"cfg": "abc"}}
    monkeypatch.setattr(p31, "row_detector", lambda config, row, name, device: groups["p30_augmented"][name])
    device = torch.device("cpu")
    args = argparse.Namespace(row="toy")
    root = tmp_path / "p31" / "toy"

    with pytest.raises(SystemExit, match="validation"):
        (root / "bundles").mkdir(parents=True)
        for name in names:                      # a stray bundle, so the digest can be computed
            (root / "bundles" / f"selector_{name}.pth").write_text("x")
        p31.step_evaluate(args, "test", config, p30, evaluation, device, root)
    for name in names:
        (root / "bundles" / f"selector_{name}.pth").unlink()
    (root / "bundles").rmdir()
    (root / "markers").rename(tmp_path / "markers_before")

    p31.step_features(args, config, p30, evaluation, selector, device, root)
    summary = json.loads((root / "features" / "features_summary.json").read_text())
    assert summary["row"] == "toy" and summary["crop_margin"] == 64
    p31.step_train(args, config, p30, evaluation, selector, root)
    p31.step_evaluate(args, "val", config, p30, evaluation, device, root)
    p31.step_evaluate(args, "test", config, p30, evaluation, device, root)
    test = json.loads((root / "test" / "p31_test.json").read_text())
    assert test["test_split_touched"] is True and test["row"] == "toy"
    for name in names:
        entry = test["datasets"][name]
        assert set(entry["without_correction"]) == {"0", "1", "2", "3"}
        assert entry["without_correction"] == {k: v["raw"] for k, v in entry["f1"].items()}
        assert "localizer + selector" in p31.render("toy", name, entry)
    p31.step_evaluate(args, "test", config, p30, evaluation, device, root)     # finished: skipped, not rerun
    with pytest.raises(SystemExit, match="used already"):
        p31.run_p30.claim("test", "other", {}, marker_dir=root / "markers")


def test_ablation_raw_path_reproduces_p30(tmp_path, monkeypatch) -> None:
    from tools.p31_localizer_ablation import evaluate, raw_anchor
    from tools.run_p27_selector import harness, open_dataset

    tool, p30, evaluation, _, groups, _ = _setup(tmp_path, monkeypatch)
    h = harness(evaluation)
    results = {}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        detector = groups["original"][name]
        result = evaluate(spec, evaluation, _tiny_localizer(), detector, torch.device("cpu"), margin=64)
        dataset = open_dataset(spec, spec["split"])
        crops = tool.margin_crops(dataset, spec, h["size"], 64)
        raw = tool.raw_scores(dataset, spec, crops, detector, h, torch.device("cpu"))
        assert all(abs(result["f1"][level]["raw"] - raw[level]) < 1e-9 for level in raw)
        assert result["crops"] == len(crops) and set(result["localization_success"]) == {"1", "2", "3"}
        assert result["gain"]["3"]["raw"] == 0.0
        results[name] = result
    anchor = {"datasets": {name: {"original": {k: v["raw"] for k, v in r["f1"].items()}} for name, r in results.items()}}
    assert raw_anchor({"datasets": results}, anchor, 0.001)["passed"]
    anchor["datasets"][name]["original"]["3"] += 0.01
    assert not raw_anchor({"datasets": results}, anchor, 0.001)["passed"]
