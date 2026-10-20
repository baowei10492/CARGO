"""P27 hold-out diagnosis: per-crop rows, summaries and agreement with the trainer."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import yaml

from rs_cd.localizer_training import TrainedLocalizer
from rs_cd.models.change_aware_similarity_localizer import build_localizer

from p27_helpers import TinyBackbone, make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path) -> dict:
    config = yaml.safe_load((ROOT / "configs/p27_localizer_v1.yaml").read_text(encoding="utf-8"))
    sources = [make_dataset(tmp_path, "big", 12, 1024), make_dataset(tmp_path, "small", 12, 512)]
    config["sources"] = [dict(s.__dict__) for s in sources]
    config["holdout"].update(fraction=0.4, crops_per_source=3)
    return config


def _model():
    return build_localizer(dict(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1),
                           backbone=TinyBackbone())


def test_diagnosis_matches_the_trainer_and_summarizes_every_source(tmp_path) -> None:
    from tools.diagnose_p27_holdout import diagnose, render, trainer_difference, write_csv
    from tools.train_similarity_localizer import build_holdout, evaluate

    config = _config(tmp_path)
    holdout = build_holdout(config)
    model = _model()
    trainer = evaluate(model, holdout, config, torch.device("cpu"))
    rows, crops, summaries = diagnose(config, TrainedLocalizer(model, torch.device("cpu"), amp=False), holdout)

    assert set(summaries) == {"big", "small"}
    assert trainer_difference(summaries, trainer) < 1e-9
    assert len(crops) == sum(len(items) for items in holdout.values())
    assert len(rows) == 2 * 4 * len(crops)
    assert {row["pair"] for row in rows} == {"real", "synthetic"}
    for crop in crops:
        for key in ("identical_t1_px", "identical_t2_px", "forward_backward_px", "clean_px"):
            assert math.isfinite(crop[key]) and crop[key] >= 0.0
        assert {f"real_l{level}_error" for level in (1, 2, 3)} <= set(crop)
    for entry in summaries.values():
        pattern = entry["failure_pattern"]
        assert abs(sum(pattern.values()) - 1.0) < 1e-9
        assert sum(b["crops"] for b in entry["top_success_by_clean_px"]) == entry["crops"]
        assert sum(b["crops"] for b in entry["top_success_by_forward_backward_px"]) == entry["crops"]
        assert set(entry["success"]) == {"real", "synthetic"}

    summary = {"checkpoint": {"source": "x/last.pth", "sha256": "0" * 64, "step": 2}, "datasets": summaries,
               "trainer_max_difference": 0.0}
    json.dumps(summary)
    text = render(summary)
    assert "big (" in text and "forward-backward" in text
    write_csv(tmp_path / "crops.csv", crops)
    with (tmp_path / "crops.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(crops)


def test_failure_pattern_and_bins() -> None:
    from tools.diagnose_p27_holdout import CLEAN_BINS, binned, failure_pattern

    def crop(clean_px, flags):
        entry = {"clean_px": clean_px}
        for level, flag in zip((1, 2, 3), flags):
            entry[f"real_l{level}_success"] = flag
            entry[f"synthetic_l{level}_success"] = True
        return entry

    crops = [crop(0.5, (True, True, True)), crop(2.5, (False, False, False)),
             crop(2.6, (True, False, False)), crop(9.0, (False, False, False))]
    assert failure_pattern(crops, (1, 2, 3)) == {"fails_none": 0.25, "fails_some": 0.25, "fails_all": 0.5}
    bins = binned(crops, "clean_px", CLEAN_BINS, 3)
    assert [b["crops"] for b in bins] == [1, 0, 2, 0, 1]
    assert bins[0]["real_top_success"] == 1.0 and bins[2]["real_top_success"] == 0.0
    assert bins[1]["real_top_success"] is None and bins[-1]["range"] == [5.0, None]


def test_main_works_on_a_saved_run(tmp_path, monkeypatch, capsys) -> None:
    import tools.diagnose_p27_holdout as tool
    from tools.train_similarity_localizer import train

    config = _config(tmp_path)
    config["data"]["num_workers"] = 0
    config["model"].update(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1)
    config["loss"]["stage_weights"] = [0.5, 1.0]
    config["optim"].update(batch_size=2, warmup_steps=1, amp=False)
    config.update(log_every=1, eval_every=2)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    train(config, run, torch.device("cpu"), backbone=TinyBackbone(), max_steps=2)

    real_load = tool.load_localizer
    monkeypatch.setattr(tool, "load_localizer",
                        lambda path, device: real_load(path, device, backbone=TinyBackbone()))
    output = tmp_path / "diagnosis"
    monkeypatch.setattr("sys.argv", ["diagnose", "--run-dir", str(run), "--device", "cpu",
                                     "--output-dir", str(output)])
    tool.main()
    summary = json.loads((output / "diagnosis_summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint"]["step"] == 2
    assert summary["trainer_max_difference"] < 1e-9
    assert not (output / "checkpoint_snapshot.pth").exists()
    assert "step 2" in capsys.readouterr().out
