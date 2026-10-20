"""P27 F1 step: scoring pixels, counts, verdicts and the evaluation loop."""
from __future__ import annotations

import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from rs_cd.affine_f1 import confusion_counts, f1_decision, path_images, scoring_mask
from rs_cd.affine_registration import perturb, perturbation_matrix
from rs_cd.localizer_training import TrainedLocalizer
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.models.change_aware_similarity_localizer import build_localizer

from p27_helpers import TinyBackbone, make_dataset, tile

ROOT = Path(__file__).resolve().parents[1]


class DifferenceDetector(nn.Module):
    """Stand-in detector: large colour differences are change."""

    def forward(self, t1, t2):
        return {"change_logits": (t1 - t2).abs().mean(dim=1, keepdim=True) * 4.0 - 2.0}


def _tiny_localizer() -> TrainedLocalizer:
    model = build_localizer(dict(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1),
                            backbone=TinyBackbone())
    return TrainedLocalizer(model, torch.device("cpu"), amp=False)


def _toy(tmp_path) -> tuple[dict, dict]:
    from tools.run_p27_f1 import load_config

    config, evaluation = load_config(ROOT / "configs/p27_f1_gate.yaml")
    big = make_dataset(tmp_path, "big", 1, 1024, split="val")
    small = make_dataset(tmp_path, "small", 2, 512, split="val")
    evaluation = copy.deepcopy(evaluation)
    evaluation["datasets"] = [
        {"name": "big", "dataset_root": big.dataset_root, "manifest": big.manifest, "split": "val",
         "tile_size": 1024, "mask_threshold": 127},
        {"name": "small", "dataset_root": small.dataset_root, "manifest": small.manifest, "split": "val",
         "tile_size": 512, "mask_threshold": 127},
    ]
    config = copy.deepcopy(config)
    config["detectors"] = {"big": {"checkpoint": "x"}, "small": {"checkpoint": "y"}}
    return config, evaluation


def test_scoring_mask_matches_a_direct_check() -> None:
    size = 32
    assert scoring_mask([np.eye(2, 3)], size).all()
    shift = perturbation_matrix(0.0, 1.0, 10.0, -3.0, width=size, height=size)
    zoom = perturbation_matrix(0.0, 1.2, 0.0, 0.0, width=size, height=size)
    mask = scoring_mask([shift, zoom], size)
    for y in range(size):
        for x in range(size):
            inside = all(0 <= m[0, 0] * x + m[0, 1] * y + m[0, 2] <= size - 1
                         and 0 <= m[1, 0] * x + m[1, 1] * y + m[1, 2] <= size - 1 for m in (shift, zoom))
            assert mask[y, x] == inside
    assert 0 < mask.sum() < size * size


def test_confusion_counts_match_binary_change_metrics() -> None:
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 1, 20, 20, generator=generator)
    target = (torch.rand(1, 1, 20, 20, generator=generator) > 0.6).float()
    valid = (torch.rand(1, 1, 20, 20, generator=generator) > 0.3).float()
    counts = confusion_counts(logits, target, valid, 0.5)
    for index in range(3):
        reference = BinaryChangeMetrics(threshold=0.5)
        reference.update(logits[index:index + 1], target, from_logits=True, valid_mask=valid)
        result = reference.compute()
        assert counts[index] == [result["tp"], result["fp"], result["fn"], result["tn"]]


def test_paths_undo_the_perturbation() -> None:
    image = tile(64, 4)
    identity = np.eye(2, 3)
    clean = path_images(image, identity, identity)
    assert np.array_equal(clean["oracle"], image) and np.array_equal(clean["estimated"], image)
    matrix = perturbation_matrix(0.0, 1.1, 5.0, -4.0, width=64, height=64)
    paths = path_images(perturb(image, matrix), matrix, matrix)
    inner = scoring_mask([matrix], 64)
    inner[:4], inner[-4:], inner[:, :4], inner[:, -4:] = False, False, False, False
    error = np.abs(paths["oracle"].astype(float) - image.astype(float))[inner]
    assert error.mean() < 3.0
    assert np.array_equal(paths["oracle"], paths["estimated"])


def _table(clean_gated=0.80, gains=(0.02, 0.05, 0.10)):
    table = {"0": {"raw": 0.80, "oracle": 0.80, "estimated": 0.79, "estimated_gated": clean_gated}}
    for level, gain in zip((1, 2, 3), gains):
        raw = 0.8 - 0.1 * level
        table[str(level)] = {"raw": raw, "oracle": 0.80, "estimated": raw + gain, "estimated_gated": raw + gain}
    return table


def _decide(table, **overrides):
    settings = dict(levels=[1, 2, 3], primary="estimated_gated", binding_levels=[2, 3], min_gain=0.0,
                    max_clean_drop=0.005, min_clean_raw_f1=0.5)
    settings.update(overrides)
    return f1_decision(table, **settings)


def test_f1_decision_rules() -> None:
    passed = _decide(_table())
    assert passed["verdict"] == "positive_gain"
    assert passed["levels"]["3"]["harm"] == pytest.approx(0.3)
    assert passed["levels"]["3"]["recovery"] == pytest.approx(0.10 / 0.30)
    assert passed["clean_drop_estimated"] == pytest.approx(0.01)
    assert _decide(_table(clean_gated=0.79))["verdict"] == "no_positive_gain"
    assert _decide(_table(clean_gated=0.796))["verdict"] == "positive_gain"
    level_1_only = _decide(_table(gains=(0.02, 0.0, 0.10)))
    assert level_1_only["verdict"] == "no_positive_gain" and not level_1_only["checks"]["gain_level_2"]
    assert _decide(_table(gains=(-0.02, 0.01, 0.01)))["verdict"] == "positive_gain"   # level 1 is reported only
    broken = _table()
    broken["0"]["oracle"] = 0.81
    assert _decide(broken)["verdict"] == "instrument_failed"
    assert _decide(_table(), min_clean_raw_f1=0.9)["verdict"] == "instrument_failed"


def test_config_is_bound_to_the_evaluation_config(tmp_path, monkeypatch) -> None:
    from tools.run_p27_f1 import load_config

    config, evaluation = load_config(ROOT / "configs/p27_f1_gate.yaml")
    assert sorted(config["detectors"]) == sorted(spec["name"] for spec in evaluation["datasets"])
    assert config["decision"] == {"primary_path": "estimated_gated", "binding_levels": [2, 3],
                                  "min_gain": 0.0, "max_clean_drop": 0.005}
    tampered = yaml.safe_load((ROOT / "configs/p27_f1_gate.yaml").read_text(encoding="utf-8"))
    tampered["evaluation_config_sha256"] = "0" * 64
    path = tmp_path / "f1.yaml"
    path.write_text(yaml.safe_dump(tampered), encoding="utf-8")
    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit, match="sha256"):
        load_config(path)


def test_run_scores_every_path_on_shared_pixels(tmp_path) -> None:
    from tools.run_p27_f1 import render, run

    config, evaluation = _toy(tmp_path)
    detectors = {"big": (DifferenceDetector(), 0.5, "a" * 64), "small": (DifferenceDetector(), 0.5, "b" * 64)}
    output = tmp_path / "out"
    output.mkdir()
    decision = run(config, evaluation, _tiny_localizer(), detectors, torch.device("cpu"), output, max_crops=3)
    assert set(decision["verdicts"]) == {"big", "small"}
    for entry in decision["datasets"].values():
        assert entry["crops"] == 3
        assert set(entry["f1"]) == {"0", "1", "2", "3"}
        assert entry["instrument"]["clean_oracle_equals_raw"]
        assert len(entry["change_bins"]) == 4
    with (output / "f1_table.csv").open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    supports = defaultdict(set)
    for row in table:
        supports[row["dataset"]].add(sum(int(row[key]) for key in ("tp", "fp", "fn", "tn")))
    assert all(len(values) == 1 for values in supports.values())
    with (output / "crop_counts.csv").open(encoding="utf-8") as handle:
        counts = list(csv.DictReader(handle))
    assert len(counts) == 2 * 3 * 4 * 4
    grouped = defaultdict(dict)
    for row in counts:
        grouped[(row["dataset"], row["crop_id"], row["level"])][row["path"]] = row
    for paths in grouped.values():
        source = "estimated" if paths["estimated_gated"]["trigger"] == "True" else "raw"
        assert all(paths["estimated_gated"][key] == paths[source][key] for key in ("tp", "fp", "fn", "tn"))
    with (output / "localizer_rows.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2 * 3 * 4
    assert "verdicts" in render(decision)
    json.dumps(decision)


def test_main_claims_the_checkpoint_once(tmp_path, monkeypatch, capsys) -> None:
    import tools.run_p27_f1 as tool

    config, evaluation = _toy(tmp_path)
    monkeypatch.setattr(tool, "load_config", lambda path: (config, evaluation))
    monkeypatch.setattr(tool, "open_detectors", lambda config, device: {
        name: {"model": DifferenceDetector(), "threshold": 0.5, "loss": {}, "sha256": "d" * 64,
               "source": {"kind": "test"}} for name in config["detectors"]})
    monkeypatch.setattr(tool, "load_localizer", lambda path, device: _tiny_localizer().model)
    monkeypatch.setattr(tool, "MARKER_DIR", tmp_path / "markers")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"frozen")
    training_config = tmp_path / "config.yaml"
    training_config.write_text("seed: 1\n", encoding="utf-8")
    f1_config = tmp_path / "f1.yaml"
    f1_config.write_text("status: preregistered\n", encoding="utf-8")

    def invoke(name):
        monkeypatch.setattr("sys.argv", ["run_p27_f1", "--config", str(f1_config), "--checkpoint", str(checkpoint),
                                         "--training-config", str(training_config), "--device", "cpu",
                                         "--output-dir", str(tmp_path / name)])
        tool.main()

    invoke("first")
    decision = json.loads((tmp_path / "first" / "f1_decision.json").read_text(encoding="utf-8"))
    assert decision["frozen"]["step"] == "p27_f1"
    assert decision["datasets"]["big"]["crops"] == 16 and decision["datasets"]["small"]["crops"] == 8
    assert "[big]" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="already met"):
        invoke("second")
