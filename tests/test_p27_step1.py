"""P27 E1: evaluation config, one-use markers and the evaluation loop."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from rs_cd.localizer_training import TrainedLocalizer
from rs_cd.models.change_aware_similarity_localizer import build_localizer

from p27_helpers import TinyBackbone, make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_evaluation_config_is_p25_without_rotation() -> None:
    p25, p27 = _load("p25_affine_protocol.yaml"), _load("p27_eval_perturbation.yaml")
    assert p27["perturbation"]["max_rotation_deg"] == 0.0
    assert (ROOT / p27["protocol_doc"]).is_file()
    assert p27["p27"] == {"trigger_probability": 0.5, "max_frozen_versions": 2}
    stripped = copy.deepcopy(p27)
    del stripped["p27"]
    stripped["protocol_doc"] = p25["protocol_doc"]
    stripped["perturbation"]["max_rotation_deg"] = p25["perturbation"]["max_rotation_deg"]
    assert stripped == p25


def test_load_config_refuses_rotation_and_test_splits(tmp_path) -> None:
    from tools.run_p27_step1 import load_config

    assert load_config(ROOT / "configs/p27_eval_perturbation.yaml")["seed"] == 3407
    rotated = _load("p27_eval_perturbation.yaml")
    rotated["perturbation"]["max_rotation_deg"] = 30.0
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(rotated), encoding="utf-8")
    with pytest.raises(SystemExit, match="without rotation"):
        load_config(path)
    tested = _load("p27_eval_perturbation.yaml")
    tested["datasets"][1]["split"] = "test"
    path.write_text(yaml.safe_dump(tested), encoding="utf-8")
    with pytest.raises(SystemExit, match="validation"):
        load_config(path)


def test_markers_allow_one_use_per_checkpoint_and_two_versions(tmp_path) -> None:
    from tools.run_p27_step1 import claim

    config = {"p27": {"max_frozen_versions": 2}}
    first = claim("a" * 64, config, tmp_path, {"n": 1})
    assert json.loads(first.read_text()) == {"n": 1}
    with pytest.raises(SystemExit, match="already met"):
        claim("a" * 64, config, tmp_path, {"n": 2})
    claim("b" * 64, config, tmp_path, {"n": 3})
    with pytest.raises(SystemExit, match="frozen versions"):
        claim("c" * 64, config, tmp_path, {"n": 4})


def test_evaluation_loop_produces_p25_style_decisions(tmp_path) -> None:
    from tools.print_step1_summary import render
    from tools.run_p27_step1 import run

    config = _load("p27_eval_perturbation.yaml")
    big = make_dataset(tmp_path, "big", 2, 1024, split="val")
    small = make_dataset(tmp_path, "small", 3, 512, split="val")
    config["datasets"] = [
        {"name": "big", "dataset_root": big.dataset_root, "manifest": big.manifest, "split": "val",
         "tile_size": 1024, "mask_threshold": 127},
        {"name": "small", "dataset_root": small.dataset_root, "manifest": small.manifest, "split": "val",
         "tile_size": 512, "mask_threshold": 127},
    ]
    config["step1"]["max_crops_per_dataset"] = 4
    model = build_localizer(dict(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1),
                            backbone=TinyBackbone())
    output = tmp_path / "out"
    output.mkdir()
    decision = run(config, TrainedLocalizer(model, torch.device("cpu")), output)
    assert set(decision["verdicts"]) == {"big", "small"}
    for entry in decision["datasets"].values():
        assert set(entry["levels"]) == {"0", "1", "2", "3"}
        assert entry["crops"] == 4
    rows = (output / "step1_estimates.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * 4 * 4
    assert "misaligned_probability" in rows[0]
    assert "verdicts" in render(decision)
