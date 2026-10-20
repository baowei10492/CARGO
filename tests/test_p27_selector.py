"""P27 selector step: P7 parity, calibration, F1-step parity and the three commands."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from rs_cd.dual_path import ROUTER_FEATURE_NAMES, dual_path_router_features
from rs_cd.p27_selector import (
    COUNT_KEYS,
    FEATURE_NAMES,
    calibrate_tau,
    choose,
    f1_table,
    sample_weights,
    selector_split,
    task_features,
)

from test_p27_f1 import DifferenceDetector, _tiny_localizer
from p27_helpers import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs/p27_selector.yaml").read_text(encoding="utf-8"))


def test_task_features_are_p7_columns() -> None:
    generator = torch.Generator().manual_seed(1)
    raw = torch.randn(3, 1, 16, 16, generator=generator)
    aligned = torch.randn(3, 1, 16, 16, generator=generator)
    valid = (torch.rand(3, 1, 16, 16, generator=generator) > 0.2).float()
    registration = {key: torch.rand(3, generator=generator) for key in (
        "registration_shift_probability", "registration_zero_probability", "registration_confidence",
        "registration_margin", "registration_local_margin", "registration_reliability")}
    registration["conditional_hard_offset_pixels"] = torch.randn(3, 2, generator=generator)
    reference = dual_path_router_features(raw, aligned, valid, registration, maximum_offset_pixels=8.0)
    assert torch.allclose(task_features(raw, aligned, valid, 0.5), reference[:, 12:])
    assert FEATURE_NAMES[6:] == tuple(ROUTER_FEATURE_NAMES[12:]) and len(FEATURE_NAMES) == 19


def test_sample_weights_follow_p7_3() -> None:
    from tools.train_evaluate_safe_alignment_classifier import _prepare_training_tensors

    generator = torch.Generator().manual_seed(2)
    advantage = (torch.rand(400, generator=generator) - 0.5) * 0.4
    rows = []
    for index, value in enumerate(advantage.tolist()):
        rows.append({"id": f"img{index // 4}", "scenario": ("clean", "s2", "s4", "s8")[index % 4],
                     "aligned_loss_advantage": value, "existing_q_selected_aligned": True,
                     **{name: 0.0 for name in ROUTER_FEATURE_NAMES}})
    p7 = yaml.safe_load((ROOT / "configs/p7_safe_alignment_classifier.yaml").read_text(encoding="utf-8"))
    prepared = _prepare_training_tensors(rows, protocol=p7)
    clean = torch.tensor([row["scenario"] == "clean" for row in rows])
    safe, weight = sample_weights(torch.tensor([row["aligned_loss_advantage"] for row in rows]), clean, _config())
    assert torch.equal(safe, prepared["safe"])
    assert torch.allclose(weight, prepared["weight"])


def test_split_is_stable_and_hold_out_goes_to_calibration() -> None:
    assert selector_split("a", holdout=True, seed=1, early_stop_fraction=0.15) == "risk_calibration"
    labels = [selector_split(f"t{i}", holdout=False, seed=3407, early_stop_fraction=0.15) for i in range(2000)]
    assert set(labels) == {"fit", "early_stop"}
    assert 0.12 < labels.count("early_stop") / 2000 < 0.18
    assert labels == [selector_split(f"t{i}", holdout=False, seed=3407, early_stop_fraction=0.15) for i in range(2000)]


def _record(level, trigger, raw, estimated, raw_loss=1.0, estimated_loss=0.5):
    record = {"level": level, "trigger": trigger, "raw_loss": raw_loss, "estimated_loss": estimated_loss}
    for path, counts in (("raw", raw), ("oracle", raw), ("estimated", estimated)):
        record.update({f"{path}_{key}": value for key, value in zip(COUNT_KEYS, counts)})
    return record


def test_rules_and_calibration() -> None:
    good, bad = (90, 10, 10, 890), (40, 60, 60, 840)
    clean = [_record(0, True, good, bad), _record(0, False, good, good)]
    probabilities = [0.7, 0.9]
    assert choose("selector", clean[0], 0.7, 0.6) == "estimated"
    assert choose("selector_veto", clean[1], 0.9, 0.6) == "raw"
    assert choose("gated", clean[0], 0.0, 0.6) == "estimated"
    assert choose("loss_oracle", clean[0], 0.0, 0.6) == "estimated"
    # The harmful clean crop has P = 0.7, so every tau <= 0.70 drops clean F1.
    tau, table = calibrate_tau(clean, probabilities, target=0.0025, start=0.5, stop=1.0, step=0.01)
    assert tau == pytest.approx(0.71)
    assert table[0]["clean_drop"] > 0.0025 and len(table) == 51
    never, _ = calibrate_tau([_record(0, True, good, bad)], [1.0], target=0.0, start=0.5, stop=1.0, step=0.1)
    assert math.isinf(never)
    shifted = [_record(3, True, bad, good), _record(3, True, bad, good)]
    result = f1_table(clean + shifted, probabilities + [0.8, 0.2], 0.71)
    assert result["3"]["estimated"] > result["3"]["selector"] > result["3"]["raw"]
    assert result["0"]["selector"] == result["0"]["raw"]
    assert math.isinf(tau) is False and choose("selector", shifted[0], 0.5, math.inf) == "raw"


def _toy(tmp_path):
    from tools.run_p27_selector import load_config

    config, f1_config, evaluation = load_config(ROOT / "configs/p27_selector.yaml")
    # Tile counts chosen so the hashed splits give every training split at least one tile.
    big = make_dataset(tmp_path, "big", 10, 1024, splits=["train"] * 7 + ["val"] * 2 + ["test"])
    small = make_dataset(tmp_path, "small", 13, 512, splits=["train"] * 10 + ["val"] * 2 + ["test"])
    evaluation = copy.deepcopy(evaluation)
    evaluation["datasets"] = [
        {"name": "big", "dataset_root": big.dataset_root, "manifest": big.manifest, "split": "val",
         "tile_size": 1024, "mask_threshold": 127},
        {"name": "small", "dataset_root": small.dataset_root, "manifest": small.manifest, "split": "val",
         "tile_size": 512, "mask_threshold": 127},
    ]
    f1_config = copy.deepcopy(f1_config)
    f1_config["detectors"] = {"big": {"checkpoint": "x"}, "small": {"checkpoint": "y"}}
    config = copy.deepcopy(config)
    config["training_crops"].update(max_fit_crops_per_dataset=12, early_stop_fraction=0.5)
    config["classifier"].update(max_epochs=4, early_stop_patience=2)
    localizer_config = {"seed": 3407, "holdout": {"fraction": 0.4},
                        "sources": [{key: spec[key] for key in ("name", "dataset_root", "manifest", "tile_size",
                                                                "mask_threshold")}
                                    for spec in evaluation["datasets"]]}
    detectors = {name: {"model": DifferenceDetector(), "threshold": 0.5,
                        "loss": {"bce_weight": 1.0, "dice_weight": 1.0, "pos_weight": 2.0}, "sha256": name * 3}
                 for name in ("big", "small")}
    return config, f1_config, evaluation, localizer_config, detectors


def test_features_train_evaluate(tmp_path) -> None:
    from tools.run_p27_f1 import run as run_f1
    from tools.run_p27_selector import (
        read_records,
        render_calibration,
        render_evaluation,
        run_evaluate,
        run_features,
        train_dataset,
        write_csv,
    )

    config, f1_config, evaluation, localizer_config, detectors = _toy(tmp_path)
    localizer = _tiny_localizer()
    device = torch.device("cpu")
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    summary = run_features(config, evaluation, localizer, detectors, localizer_config, device, features_dir)
    for name, entry in summary.items():
        assert entry["records"] == 4 * sum(entry["crops"].values())
        assert all(entry["crops"][split] > 0 for split in ("fit", "early_stop", "risk_calibration")), entry

    bundles, summaries = {}, {}
    for name in ("big", "small"):
        records = read_records(features_dir / f"features_{name}.csv")
        assert {record["split"] for record in records} == {"fit", "early_stop", "risk_calibration"}
        assert all(isinstance(record["trigger"], bool) for record in records)
        bundles[name], summaries[name] = train_dataset(name, records, config)
        assert summaries[name]["tau_table"][0]["tau"] == 0.5
    assert "calibration F1" in render_calibration(summaries)
    json.dumps(summaries)

    # The F1 step on the same validation crops is the anchor.
    f1_dir = tmp_path / "f1"
    f1_dir.mkdir()
    reference = run_f1(f1_config, evaluation, localizer,
                       {name: (entry["model"], 0.5, entry["sha256"]) for name, entry in detectors.items()},
                       device, f1_dir)
    output = tmp_path / "evaluate"
    output.mkdir()
    decision = run_evaluate(config, f1_config, evaluation, localizer, detectors, bundles, reference, device, output)
    for name, entry in decision["datasets"].items():
        assert entry["anchor"]["passed"], entry["anchor"]
        assert entry["anchor"]["max_abs_difference"] < 1e-9
        assert entry["verdict"] in {"positive_gain", "no_positive_gain", "instrument_failed"}
        assert set(entry["f1"]["3"]) == {"raw", "oracle", "estimated", "gated", "selector", "selector_veto",
                                         "loss_oracle"}
        assert entry["f1"]["3"]["loss_oracle"] >= max(entry["f1"]["3"]["raw"], entry["f1"]["3"]["estimated"]) - 0.05
    assert (output / "validation_records.csv").is_file()
    assert "verdicts" in render_evaluation(decision)
    json.dumps(decision)

    broken = copy.deepcopy(reference)
    broken["datasets"]["big"]["f1"]["3"]["raw"] += 0.01
    second = tmp_path / "evaluate2"
    second.mkdir()
    failed = run_evaluate(config, f1_config, evaluation, localizer, detectors, bundles, broken, device, second)
    assert failed["verdicts"]["big"] == "anchor_failed" and "f1" not in failed["datasets"]["big"]
    assert not (second / "validation_records.csv").exists()
    write_csv(tmp_path / "x.csv", [{"a": 1}])


def test_train_command_and_evaluate_guards(tmp_path, monkeypatch) -> None:
    import tools.run_p27_f1 as f1_tool
    import tools.run_p27_selector as tool

    config, f1_config, evaluation, localizer_config, detectors = _toy(tmp_path)
    monkeypatch.setattr(tool, "load_config", lambda path: (config, f1_config, evaluation))
    config_file = tmp_path / "selector.yaml"
    config_file.write_text("status: preregistered\n", encoding="utf-8")
    config_sha = tool.sha256(config_file)
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    tool.run_features(config, evaluation, _tiny_localizer(), detectors, localizer_config, torch.device("cpu"),
                      features_dir)
    (features_dir / "features_summary.json").write_text(
        json.dumps({"config_sha256": config_sha, "localizer_sha256": "l" * 64}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run", "train", "--config", str(config_file), "--features-dir",
                                     str(features_dir), "--output-dir", str(tmp_path / "bundles")])
    tool.main()
    bundle = torch.load(tmp_path / "bundles" / "selector_big.pth", weights_only=False)
    assert bundle["config_sha256"] == config_sha and bundle["localizer_sha256"] == "l" * 64
    assert (tmp_path / "bundles" / "train_summary.json").is_file()

    # Evaluation refuses to start when the localizer never ran the F1 step.
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen")
    config["localizer_run"] = str(tmp_path / "run")
    reference = {"frozen": {"checkpoint_sha256": tool.sha256(checkpoint)},
                 "datasets": {name: {"detector": {"sha256": entry["sha256"]}} for name, entry in detectors.items()}}
    reference_file = tmp_path / "f1_decision.json"
    reference_file.write_text(json.dumps(reference), encoding="utf-8")
    config["f1_reference_decision"] = str(reference_file)
    monkeypatch.setattr(tool, "load_detectors", lambda f1, device: detectors)
    monkeypatch.setattr(tool, "load_localizer", lambda path, device: _tiny_localizer().model)
    monkeypatch.setattr(f1_tool, "MARKER_DIR", tmp_path / "markers")
    monkeypatch.setattr(tool, "SELECTOR_MARKER_DIR", tmp_path / "selector_markers")
    monkeypatch.setattr("sys.argv", ["run", "evaluate", "--config", str(config_file), "--device", "cpu",
                                     "--bundle-dir", str(tmp_path / "bundles"), "--output-dir", str(tmp_path / "e")])
    with pytest.raises(SystemExit, match="F1-step marker"):
        tool.main()
    (tmp_path / "markers").mkdir()
    (tmp_path / "markers" / f"{tool.sha256(checkpoint)}.json").write_text(json.dumps({"step": "p27_f1"}),
                                                                           encoding="utf-8")
    with pytest.raises(SystemExit, match="different config or localizer"):
        tool.main()
    reference["datasets"]["big"]["detector"]["sha256"] = "other"
    reference_file.write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(SystemExit, match="different detector"):
        tool.main()


def test_test_command_runs_once_after_validation(tmp_path, monkeypatch) -> None:
    import tools.run_p27_f1 as f1_tool
    import tools.run_p27_selector as tool

    config, f1_config, evaluation, localizer_config, detectors = _toy(tmp_path)
    monkeypatch.setattr(tool, "load_config", lambda path: (config, f1_config, evaluation))
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen")
    localizer_sha = tool.sha256(checkpoint)
    config["localizer_run"] = str(tmp_path / "run")
    config_file = tmp_path / "selector.yaml"
    config_file.write_text("status: preregistered\n", encoding="utf-8")
    config_sha = tool.sha256(config_file)
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    tool.run_features(config, evaluation, _tiny_localizer(), detectors, localizer_config, torch.device("cpu"),
                      features_dir)
    (features_dir / "features_summary.json").write_text(
        json.dumps({"config_sha256": config_sha, "localizer_sha256": localizer_sha}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run", "train", "--config", str(config_file), "--features-dir",
                                     str(features_dir), "--output-dir", str(tmp_path / "bundles")])
    tool.main()

    reference_file = tmp_path / "f1_decision.json"
    reference_file.write_text(json.dumps({
        "frozen": {"checkpoint_sha256": localizer_sha},
        "datasets": {name: {"detector": {"sha256": entry["sha256"]}} for name, entry in detectors.items()},
    }), encoding="utf-8")
    config["f1_reference_decision"] = str(reference_file)
    (tmp_path / "markers").mkdir()
    (tmp_path / "markers" / f"{localizer_sha}.json").write_text(json.dumps({"step": "p27_f1"}), encoding="utf-8")
    test_file = tmp_path / "test.yaml"
    test_file.write_text(yaml.safe_dump({"status": "preregistered", "selector_config_sha256": config_sha,
                                         "split": "test", "max_test_runs": 1}), encoding="utf-8")
    monkeypatch.setattr(tool, "load_detectors", lambda f1, device: detectors)
    monkeypatch.setattr(tool, "load_localizer", lambda path, device: _tiny_localizer().model)
    monkeypatch.setattr(f1_tool, "MARKER_DIR", tmp_path / "markers")
    monkeypatch.setattr(tool, "SELECTOR_MARKER_DIR", tmp_path / "selector_markers")
    monkeypatch.setattr(tool, "TEST_MARKER_DIR", tmp_path / "test_markers")

    def run_test(name):
        monkeypatch.setattr("sys.argv", ["run", "test", "--config", str(config_file), "--test-config",
                                         str(test_file), "--device", "cpu", "--bundle-dir",
                                         str(tmp_path / "bundles"), "--output-dir", str(tmp_path / name)])
        tool.main()

    with pytest.raises(SystemExit, match="not been evaluated on validation"):
        run_test("t0")
    _, digests = tool.load_bundles(tmp_path / "bundles", ["big", "small"])
    (tmp_path / "selector_markers").mkdir()
    (tmp_path / "selector_markers" / f"{tool._bundle_manifest_sha(digests, localizer_sha)}.json").write_text(
        "{}", encoding="utf-8")
    run_test("t1")
    decision = json.loads((tmp_path / "t1" / "test_decision.json").read_text(encoding="utf-8"))
    assert decision["split"] == "test" and decision["test_split_touched"] and decision["reporting_only"]
    assert decision["datasets"]["big"]["crops"] == 16 and decision["datasets"]["small"]["crops"] == 4
    assert decision["datasets"]["big"]["anchor"] is None
    assert (tmp_path / "t1" / "test_records.csv").is_file()
    with pytest.raises(SystemExit, match="already met"):
        run_test("t2")
