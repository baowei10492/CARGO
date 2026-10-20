"""P27 ablations: offline selector variants and threshold sweep."""
from __future__ import annotations

import json
import math

import torch

from test_p27_f1 import _tiny_localizer
from test_p27_selector import _toy


def test_selector_ablation_reproduces_the_frozen_selector(tmp_path) -> None:
    from tools.p27_selector_ablation import VARIANTS, render, run
    from tools.run_p27_f1 import run as run_f1
    from tools.run_p27_selector import read_records, run_evaluate, run_features, train_dataset

    config, f1_config, evaluation, localizer_config, detectors = _toy(tmp_path)
    localizer, device = _tiny_localizer(), torch.device("cpu")
    train_root = tmp_path / "train"
    (train_root / "features").mkdir(parents=True)
    (train_root / "bundles").mkdir()
    run_features(config, evaluation, localizer, detectors, localizer_config, device, train_root / "features")
    bundles = {}
    for name in ("big", "small"):
        bundles[name], _ = train_dataset(name, read_records(train_root / "features" / f"features_{name}.csv"), config)
        torch.save(bundles[name], train_root / "bundles" / f"selector_{name}.pth")
    f1_dir, eval_dir = tmp_path / "f1", tmp_path / "eval"
    f1_dir.mkdir()
    eval_dir.mkdir()
    reference = run_f1(f1_config, evaluation, localizer,
                       {name: (entry["model"], 0.5, entry["sha256"]) for name, entry in detectors.items()},
                       device, f1_dir)
    run_evaluate(config, f1_config, evaluation, localizer, detectors, bundles, reference, device, eval_dir)

    result = run(config, train_root, eval_dir / "validation_records.csv", sweep_step=0.25)
    for name, entry in result["datasets"].items():
        assert entry["full_reproduces_main"]["tau_equal"]
        assert entry["full_reproduces_main"]["max_probability_difference"] < 1e-6
        assert set(entry["variants"]) == set(VARIANTS)
        assert entry["variants"]["registration_only"]["features"] == 6
        assert entry["variants"]["task_only"]["features"] == 13
        taus = [row["tau"] for row in entry["sweep"]]
        assert taus == [0.0, 0.25, 0.5, 0.75, 1.0, None]
        assert entry["sweep"][-1]["clean_drop"] == 0.0 and entry["sweep"][-1]["gain_3"] == 0.0
        assert math.isclose(entry["main"]["gain_3"], entry["variants"]["full"]["gain_3"])
    text = render(result)
    assert "retrained registration_only (6)" in text and "localizer gate" in text
    json.dumps(result)


def test_efficiency_measurement_covers_every_part(tmp_path) -> None:
    from p27_helpers import tile
    from test_p27_f1 import DifferenceDetector
    from tools.p27_efficiency import count, measure
    from tools.run_p27_selector import read_records, run_features, train_dataset

    config, _, evaluation, localizer_config, detectors = _toy(tmp_path)
    localizer = _tiny_localizer()
    features = tmp_path / "features"
    features.mkdir()
    run_features(config, evaluation, localizer, detectors, localizer_config, torch.device("cpu"), features)
    bundle, _ = train_dataset("small", read_records(features / "features_small.csv"), config)
    t1, t2 = tile(256, 1), tile(256, 2)
    timing = measure(localizer, DifferenceDetector(), bundle, t1, t2, torch.device("cpu"), repeats=2, warmup=1)
    assert set(timing) == {"localizer_both_directions", "warp", "detector_two_passes", "selector", "total_median_ms"}
    assert timing["total_median_ms"] > 0
    assert count(localizer.model)["total"] > count(localizer.model)["trainable"] > 0
