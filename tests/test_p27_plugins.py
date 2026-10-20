"""P27 plug-in rows: detector sources, dataset subsets and per-detector markers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from rs_cd.p27_detectors import open_detector, resolve_detector, sha256
from rs_cd.runtime import build_model

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = {
    "model": {"name": "fc_siam_diff", "in_channels": 3, "base_channels": 4},
    "evaluation": {"threshold": 0.5},
    "loss": {"bce_weight": 1.0, "dice_weight": 1.0, "pos_weight": 2.0},
}


def _cell(root: Path, name: str, *, stage1: bool, dual_path: bool) -> Path:
    """A frozen P21 cell as run_p21_cell.sh leaves it (paths relative to root)."""

    torch.manual_seed(0)
    model = build_model(RUN_CONFIG)
    checkpoint = root / "outputs" / f"p21_{name}" / "run" / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"config": RUN_CONFIG, "model": model.state_dict()}, checkpoint)
    digest = sha256(checkpoint)
    report = root / "reports" / "p21" / "cells" / name / "20260914T000000Z"
    report.mkdir(parents=True)
    (report / "DETECTOR_CHECKPOINT").write_text(f"outputs/p21_{name}/run/checkpoints/best.pth\n", encoding="utf-8")
    if dual_path:
        config = {**RUN_CONFIG, "model": {"name": "dual_path_adapter", "detector": RUN_CONFIG["model"],
                                          "localizer": {"name": "p7"}}}
        state = {f"detector.{key}": value for key, value in model.state_dict().items()}
        state["localizer.weight"] = torch.zeros(1)
        torch.save({"config": config, "model": state, "provenance": {"detector_sha256": digest}},
                   report / "dual_path.pth")
    if not stage1:
        checkpoint.unlink()
    return report


def _manifest(root: Path, cells: dict[str, Path]) -> Path:
    path = root / "reports" / "p21" / "VALIDATION_FROZEN.json"
    path.write_text(json.dumps({"status": "frozen", "cells": {
        name: {"report_dir": str(report.relative_to(root)).replace("\\", "/")} for name, report in cells.items()
    }}), encoding="utf-8")
    return path


def test_cells_resolve_to_the_same_detector(tmp_path) -> None:
    both = _cell(tmp_path, "bit_levir", stage1=True, dual_path=True)
    dual_only = _cell(tmp_path, "cdxlstm_levir", stage1=False, dual_path=True)
    missing = _cell(tmp_path, "cdxlstm_dsifn", stage1=False, dual_path=False)
    manifest = _manifest(tmp_path, {"bit_levir": both, "cdxlstm_levir": dual_only, "cdxlstm_dsifn": missing})
    device = torch.device("cpu")

    assert resolve_detector({"cell": "bit_levir"}, manifest_path=manifest, project_root=tmp_path)["kind"] == "checkpoint"
    assert resolve_detector({"cell": "cdxlstm_levir"}, manifest_path=manifest,
                            project_root=tmp_path)["kind"] == "dual_path"
    first, config, digest, _ = open_detector({"cell": "bit_levir"}, device, manifest_path=manifest,
                                             project_root=tmp_path)
    second, _, dual_digest, source = open_detector({"cell": "cdxlstm_levir"}, device, manifest_path=manifest,
                                                   project_root=tmp_path)
    assert config["evaluation"]["threshold"] == 0.5 and source["kind"] == "dual_path"
    # Same seed, same weights: the dual-path half is the stage-1 detector.
    assert digest == dual_digest
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(first(image, image)["change_logits"], second(image, image)["change_logits"])
    assert not first.training
    with pytest.raises(SystemExit, match="neither"):
        resolve_detector({"cell": "cdxlstm_dsifn"}, manifest_path=manifest, project_root=tmp_path)
    with pytest.raises(SystemExit, match="not a frozen cell"):
        resolve_detector({"cell": "bit_gzcd"}, manifest_path=manifest, project_root=tmp_path)
    with pytest.raises(SystemExit, match="expected"):
        open_detector({"cell": "bit_levir", "sha256": "0" * 64}, device, manifest_path=manifest,
                      project_root=tmp_path)
    checkpoint = tmp_path / "outputs" / "p21_bit_levir" / "run" / "checkpoints" / "best.pth"
    _, _, direct, _ = open_detector({"checkpoint": str(checkpoint)}, device)
    assert direct == digest


def _plugin_configs():
    configs = {}
    for detector in ("bit", "cdlamba", "cdxlstm"):
        configs[detector] = {kind: ROOT / "configs" / "p27_plugins" / f"{detector}_{kind}.yaml"
                             for kind in ("f1", "selector", "test")}
    return configs


def test_plugin_configs_chain_and_cover_levir_and_dsifn(monkeypatch) -> None:
    from tools.run_p27_f1 import PLUGIN_MARKER_ROOT, load_config, marker_dir
    from tools.run_p27_selector import load_config as load_selector_config
    from tools.run_p27_selector import load_test_config, step_marker_dirs

    monkeypatch.chdir(ROOT)
    for detector, paths in _plugin_configs().items():
        f1_config, evaluation = load_config(paths["f1"])
        assert [spec["name"] for spec in evaluation["datasets"]] == ["LEVIR-CD", "DSIFN-CD"]
        assert f1_config["detectors"] == {"LEVIR-CD": {"cell": f"{detector}_levir"},
                                          "DSIFN-CD": {"cell": f"{detector}_dsifn"}}
        assert marker_dir(f1_config) == PLUGIN_MARKER_ROOT / detector / "frozen"
        dinov3 = yaml.safe_load((ROOT / "configs/p27_f1_gate.yaml").read_text(encoding="utf-8"))
        assert f1_config["decision"] == dinov3["decision"] and f1_config["instrument"] == dinov3["instrument"]
        config, chained_f1, chained_evaluation = load_selector_config(paths["selector"])
        assert chained_f1 == f1_config and chained_evaluation == evaluation
        assert config["f1_reference_decision"] == f"reports/p27_localizer/plugins/{detector}/f1/LATEST"
        dinov3_selector = yaml.safe_load((ROOT / "configs/p27_selector.yaml").read_text(encoding="utf-8"))
        for key in ("label", "weights", "classifier", "risk", "calibration", "evaluation", "training_crops",
                    "localizer_run", "localizer_checkpoint", "seed", "anchor_tolerance"):
            assert config[key] == dinov3_selector[key], key
        load_test_config(paths["test"], sha256(paths["selector"]))
        markers = step_marker_dirs(f1_config)
        assert markers == (PLUGIN_MARKER_ROOT / detector / "frozen",
                           PLUGIN_MARKER_ROOT / detector / "selector_frozen",
                           PLUGIN_MARKER_ROOT / detector / "test_frozen")
    dinov3_config, _ = load_config(ROOT / "configs/p27_f1_gate.yaml")
    from tools.run_p27_step1 import MARKER_DIR
    assert marker_dir(dinov3_config) == MARKER_DIR


def test_dataset_subset_is_checked(tmp_path, monkeypatch) -> None:
    from tools.run_p27_f1 import load_config

    monkeypatch.chdir(ROOT)
    config = yaml.safe_load(_plugin_configs()["bit"]["f1"].read_text(encoding="utf-8"))
    config["datasets"] = ["LEVIR-CD", "GZ-CD"]
    path = tmp_path / "f1.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="subset"):
        load_config(path)
    config["datasets"] = ["LEVIR-CD"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="do not match"):
        load_config(path)


def test_reference_pointer(tmp_path) -> None:
    from tools.run_p27_selector import reference_path

    run = tmp_path / "f1" / "20260917T000000Z"
    (tmp_path / "f1").mkdir()
    (tmp_path / "f1" / "LATEST").write_text(f"{run}\n", encoding="utf-8")
    assert reference_path(str(tmp_path / "f1" / "LATEST")) == run / "run" / "f1_decision.json"
    assert reference_path("reports/x/f1_decision.json") == Path("reports/x/f1_decision.json")
