"""P27 comparison and ablation tools: baselines, localizer variants, DiffRegCD harness, panels, WHU rows."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from rs_cd.affine_registration import SimilarityEstimate, grid_points, perturb, perturbation_matrix, transfer_error
from rs_cd.p27_baseline_localizers import SiftSimilarity, SimilarityBaseline, p27_variant

from p27_helpers import TinyBackbone, make_dataset
from test_p27_f1 import DifferenceDetector, _tiny_localizer
from test_p27_selector import _toy

ROOT = Path(__file__).resolve().parents[1]
FIT = dict(reprojection_px=1.0, max_iters=2000, confidence=0.999, refine_iters=10, min_inliers=8)


def _textured(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 90, np.uint8)
    for _ in range(60):
        x, y = rng.integers(10, size - 30, 2)
        w, h = rng.integers(8, 28, 2)
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), colour, -1)
    return image


def test_sift_baseline_recovers_a_similarity() -> None:
    image = _textured()
    truth = perturbation_matrix(0.0, 1.08, 9.0, -6.0, width=256, height=256)
    baseline = SimilarityBaseline(SiftSimilarity(seed=1, **FIT), trigger_px=4.0)
    moved, same = baseline.estimate_many([image, image], [perturb(image, truth), image])
    points = grid_points(256, 256, 17)
    assert moved.converged and transfer_error(moved.matrix, truth, points, width=256, height=256) < 1.0
    assert moved.misaligned_probability == 1.0 and same.misaligned_probability == 0.0
    blank = np.zeros_like(image)
    empty = baseline.estimate_many([blank], [blank])[0]
    assert not empty.converged and empty.misaligned_probability == 0.0
    assert np.allclose(empty.matrix, np.eye(2, 3))


def test_baseline_trigger_needs_convergence() -> None:
    shifted = SimilarityEstimate(matrix=np.array([[1.0, 0, 10.0], [0, 1.0, 0]]), converged=False)
    baseline = SimilarityBaseline(lambda a, b, key: shifted, trigger_px=4.0, size=64)
    assert baseline.estimate_many([np.zeros((64, 64, 3), np.uint8)], [np.zeros((64, 64, 3), np.uint8)])[0] \
        .misaligned_probability == 0.0


def test_p27_variants_run_without_retraining() -> None:
    from rs_cd.models.change_aware_similarity_localizer import build_localizer

    image = _textured(128)
    for options in ({"fine_iterations": 0}, {"fine_iterations": 1}, {"reliability_weighting": False}):
        model = build_localizer(dict(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=2),
                                backbone=TinyBackbone())
        localizer = p27_variant(model, torch.device("cpu"), **options)
        estimate = localizer.estimate_many([image], [image])[0]
        assert np.isfinite(estimate.matrix).all() and 0.0 <= estimate.misaligned_probability <= 1.0
    with pytest.raises(ValueError, match="reduced"):
        p27_variant(model, torch.device("cpu"), fine_iterations=5)
    output = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))
    assert torch.equal(output["change_logits"], torch.zeros_like(output["change_logits"]))


def _test_split(evaluation: dict) -> dict:
    copied = copy.deepcopy(evaluation)
    for spec in copied["datasets"]:
        spec["split"] = "test"
    return copied


def test_localizer_ablation_reports_every_dataset(tmp_path) -> None:
    from tools.p27_localizer_ablation import render, run_variant

    _, f1_config, evaluation, _, detectors = _toy(tmp_path)
    evaluation = _test_split(evaluation)
    result = run_variant("p27_full", Path("unused"), f1_config, evaluation, detectors, torch.device("cpu"),
                         max_crops=2, localizer=_tiny_localizer())
    assert result["split"] == "test" and set(result["datasets"]) == {"big", "small"}
    entry = result["datasets"]["big"]
    assert set(entry["localization_success"]) == {"0", "1", "2", "3"}
    assert set(entry["gain"]["3"]) == {"raw", "oracle", "estimated", "estimated_gated"}
    assert entry["gain"]["3"]["raw"] == 0.0
    assert "p27_full" in render({"p27_full": result})
    json.dumps(result)


def test_diffregcd_harness_scores_both_paths(tmp_path) -> None:
    from tools.run_p27_diffregcd_f1 import change_logit, evaluate_dataset

    _, _, evaluation, _, _ = _toy(tmp_path)
    spec = {**_test_split(evaluation)["datasets"][1]}
    calls = []

    def infer(a, b):
        calls.append((a.min().item(), a.max().item(), torch.randint(0, 10**6, (1,)).item()))
        difference = (a - b).abs().mean(dim=1, keepdim=True) * 4.0 - 1.0
        return torch.cat([-difference, difference], dim=1)

    result = evaluate_dataset(spec, evaluation, infer, seed=3407, threshold=0.5, device=torch.device("cpu"),
                              max_crops=2)
    assert result["crops"] == 2 and len(calls) == 2 * 7
    assert all(-1.0 <= low and high <= 1.0 for low, high, _ in calls)
    # Seeded per crop: the random draw is the same for every call of a crop.
    assert len({draw for _, _, draw in calls[:7]}) == 1
    assert result["f1"]["0"]["diffregcd"] == result["f1"]["0"]["diffregcd_oracle"]
    assert set(result["f1"]) == {"0", "1", "2", "3"}
    logits = torch.tensor([[[[0.2]], [[1.0]]]])
    assert torch.sigmoid(change_logit(logits)).item() == pytest.approx(torch.softmax(logits, dim=1)[0, 1].item())


def test_diffregcd_config_names_the_released_heads(monkeypatch) -> None:
    from tools.run_p27_diffregcd_f1 import load_config, weight_paths

    monkeypatch.chdir(ROOT)
    config, evaluation = load_config(ROOT / "configs/p27_diffregcd.yaml")
    assert config["split"] == "test" and set(config["datasets"]) == {"LEVIR-CD", "DSIFN-CD", "WHU-CD"}
    assert weight_paths(config, "WHU-CD")["change_detection"].as_posix().endswith("checkpoints/whu/best_cd_model_gen.pth")
    assert all(len(entry[key]["sha256"]) == 64 for entry in config["datasets"].values()
               for key in ("registration", "change_detection"))


def test_qualitative_panels(tmp_path) -> None:
    from tools.p27_qualitative import crop_f1, pick, render_example
    from tools.run_p27_selector import read_records, run_evaluate, run_features, train_dataset

    config, f1_config, evaluation, localizer_config, detectors = _toy(tmp_path)
    localizer, device = _tiny_localizer(), torch.device("cpu")
    features = tmp_path / "features"
    features.mkdir()
    run_features(config, evaluation, localizer, detectors, localizer_config, device, features)
    bundles = {name: train_dataset(name, read_records(features / f"features_{name}.csv"), config)[0]
               for name in ("big", "small")}
    test_evaluation = _test_split(evaluation)
    out = tmp_path / "test"
    out.mkdir()
    run_evaluate(config, f1_config, test_evaluation, localizer, detectors, bundles, None, device, out, split="test")
    records = read_records(out / "test_records.csv")
    spec = test_evaluation["datasets"][0]
    members = [{**r, "tau": 0.0} for r in records if r["dataset"] == spec["name"]]
    for r in members:
        r["change_fraction"] = 0.5
    chosen = pick(members, 2)
    assert set(chosen) == {"corrected", "kept_clean", "failure"}
    assert len(chosen["corrected"]) == 2
    assert crop_f1(chosen["corrected"][0], "estimated") - crop_f1(chosen["corrected"][0], "raw") >= \
        crop_f1(chosen["corrected"][1], "estimated") - crop_f1(chosen["corrected"][1], "raw")
    detector = {"model": DifferenceDetector(), "threshold": 0.5}
    image = render_example(chosen["corrected"][0], spec, test_evaluation, localizer, detector, device)
    assert image.size == (7 * 256 + 6 * 6, 256 + 34)

    from tools.p27_qualitative import VIEWS, example_views, find_record

    record = chosen["corrected"][0]
    assert find_record(members, record["crop_id"], record["level"]) is record
    with pytest.raises(SystemExit):
        find_record(members, "missing|y0000x0000", 3)
    images, scores = example_views(record, spec, test_evaluation, localizer, detector, device)
    assert set(images) == set(VIEWS) and all(v.shape == (256, 256, 3) for v in images.values())
    # tau 0: the selector always corrects, so its output is the localizer's.
    assert np.array_equal(images["with_selector"], images["localizer_only"])
    assert scores["localizer_only"] == pytest.approx(crop_f1(record, "estimated"), abs=1e-6)
    assert scores["without_correction"] == pytest.approx(crop_f1(record, "raw"), abs=1e-6)


def test_qualitative_figure_assembles(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image

    from tools.p27_figure_qualitative import COLUMNS, assemble, entries

    (tmp_path / "tiles").mkdir()
    index = []
    for order, (dataset, level) in enumerate((("B", 0), ("A", 3)), start=1):
        tiles = {}
        for view, _ in COLUMNS:
            tiles[view] = f"tiles/{dataset}_{view}.png"
            Image.fromarray(np.full((32, 32, 3), 40 * order, np.uint8)).save(tmp_path / tiles[view])
        index.append({"dataset": dataset, "category": "selected", "order": 3 - order, "level": level, "tiles": tiles,
                      "rendered_f1": {"without_correction": 0.1, "localizer_only": 0.9, "with_selector": 0.9}})
    (tmp_path / "qualitative_index.json").write_text(json.dumps(index), encoding="utf-8")
    rows = entries(tmp_path)
    assert [r["dataset"] for r in rows] == ["A", "B"]
    written = assemble(tmp_path, rows, tmp_path / "figure" / "qualitative")
    assert [p.suffix for p in written] == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in written)


def test_whu_rows_chain_to_trained_detectors(tmp_path, monkeypatch) -> None:
    from rs_cd.p27_detectors import resolve_detector
    from tools.run_p27_f1 import load_config, marker_dir, PLUGIN_MARKER_ROOT
    from tools.run_p27_selector import load_config as load_selector_config
    from tools.run_p27_selector import load_test_config
    from tools.run_p27_step1 import sha256

    monkeypatch.chdir(ROOT)
    for detector in ("bit", "cdlamba", "cdxlstm"):
        training = yaml.safe_load((ROOT / "configs/p21" / f"{detector}_whu.yaml").read_text(encoding="utf-8"))
        dsifn = yaml.safe_load((ROOT / "configs/p21" / f"{detector}_dsifn.yaml").read_text(encoding="utf-8"))
        assert training["model"] == dsifn["model"] and training["training"] == dsifn["training"]
        assert training["data"]["manifest"] == "manifests/datasets/whu_cd.csv"
        assert training["data"]["mask_threshold"] == 127 and training["experiment"]["name"] == f"p21_{detector}_whu"
        row = f"{detector}_whu"
        f1_config, evaluation = load_config(ROOT / "configs/p27_plugins" / f"{row}_f1.yaml")
        assert [spec["name"] for spec in evaluation["datasets"]] == ["WHU-CD"]
        assert marker_dir(f1_config) == PLUGIN_MARKER_ROOT / row / "frozen"
        load_selector_config(ROOT / "configs/p27_plugins" / f"{row}_selector.yaml")
        load_test_config(ROOT / "configs/p27_plugins" / f"{row}_test.yaml",
                         sha256(ROOT / "configs/p27_plugins" / f"{row}_selector.yaml"))

    pointer = tmp_path / "outputs" / "p27_whu_detectors" / "bit" / "CHECKPOINT"
    spec = {"checkpoint_pointer": "outputs/p27_whu_detectors/bit/CHECKPOINT"}
    with pytest.raises(SystemExit, match="train the detector first"):
        resolve_detector(spec, project_root=tmp_path)
    pointer.parent.mkdir(parents=True)
    pointer.write_text("outputs/p21_bit_whu/run/checkpoints/best.pth\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not exist"):
        resolve_detector(spec, project_root=tmp_path)
    checkpoint = tmp_path / "outputs" / "p21_bit_whu" / "run" / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    assert resolve_detector(spec, project_root=tmp_path)["path"] == checkpoint


def test_summary_rejudges_with_a_tolerant_instrument(tmp_path) -> None:
    from tools.p27_summary import collect, render

    rules = yaml.safe_load((ROOT / "configs/p27_f1_gate.yaml").read_text(encoding="utf-8"))
    table = {"0": {"raw": 0.8958, "oracle": 0.89583, "estimated": 0.85, "estimated_gated": 0.874,
                   "selector": 0.8950},
             **{str(level): {"raw": 0.7 - 0.1 * level, "oracle": 0.86, "estimated": 0.78,
                             "estimated_gated": 0.77, "selector": 0.76} for level in (1, 2, 3)}}
    root = tmp_path / "reports" / "p27_localizer"
    for stage, filename, directory in (("f1", "f1_decision.json", root / "plugins" / "cdlamba" / "f1"),
                                       ("test", "test_decision.json", root / "plugins" / "cdlamba" / "test")):
        run = directory / "20260917T000000Z"
        (run / "run").mkdir(parents=True)
        datasets = {"LEVIR-CD": {"verdict": "instrument_failed", "f1": table},
                    "DSIFN-CD": {"verdict": "anchor_failed", "anchor": {}}}
        (run / "run" / filename).write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
        (directory / "LATEST").write_text(str(run.relative_to(tmp_path)).replace("\\", "/") + "\n", encoding="utf-8")
    rows = collect(Path("reports/p27_localizer"), rules, 0.001, project_root=tmp_path)
    strict = collect(Path("reports/p27_localizer"), rules, 1e-9, project_root=tmp_path)
    judged = {(r["stage"], r["dataset"]): r for r in rows}
    assert judged[("f1", "LEVIR-CD")]["verdict"] == "no_positive_gain"   # gated clean drop 0.0218
    assert judged[("test", "LEVIR-CD")]["verdict"] == "positive_gain"    # selector clean drop 0.0008
    assert "verdict" not in judged[("test", "DSIFN-CD")]
    assert {r["verdict"] for r in strict if "verdict" in r} == {"instrument_failed"}
    text = render(rows)
    assert "positive_gain (instrument_failed)" in text and "anchor_failed" in text


def test_external_whu_detector_check_and_install(tmp_path, monkeypatch, capsys) -> None:
    import tools.p27_check_whu_detector as tool

    (tmp_path / "configs" / "p21").mkdir(parents=True)
    recipe_text = (ROOT / "configs/p21/bit_whu.yaml").read_text(encoding="utf-8")
    (tmp_path / "configs/p21/bit_whu.yaml").write_text(recipe_text, encoding="utf-8")
    (tmp_path / "manifests" / "datasets").mkdir(parents=True)
    (tmp_path / "manifests/datasets/whu_cd.csv").write_text("sample_id,split\nwhu_train_r0c0,train\n", encoding="utf-8")
    manifest_sha = tool.sha256(tmp_path / "manifests/datasets/whu_cd.csv")
    recipe = yaml.safe_load(recipe_text)
    same = copy.deepcopy(recipe)
    same["data"]["dataset_root"] = "/elsewhere/WHU-CD"
    same["experiment"]["name"] = "teammate_bit_whu"
    changed = copy.deepcopy(same)
    changed["training"]["epochs"] = 200
    for name, config in (("same.pth", same), ("changed.pth", changed)):
        torch.save({"config": config, "model": {"w": torch.zeros(1)}, "epoch": 40}, tmp_path / name)
    monkeypatch.chdir(tmp_path)

    def run(*extra):
        monkeypatch.setattr("sys.argv", ["check", "--detector", "bit", *extra])
        tool.main()

    run("--checkpoint", "changed.pth")
    report = json.loads(capsys.readouterr().out)
    assert report["differences"] == [{"key": "training.epochs", "recipe": 40, "checkpoint": 200}]
    with pytest.raises(SystemExit, match="not proven identical"):
        run("--checkpoint", "same.pth", "--install")
    with pytest.raises(SystemExit, match="recipe differences"):
        run("--checkpoint", "changed.pth", "--install", "--manifest-sha256", manifest_sha)
    run("--checkpoint", "same.pth", "--install", "--manifest-sha256", manifest_sha)
    pointer = tmp_path / "outputs/p27_whu_detectors/bit/CHECKPOINT"
    assert pointer.read_text(encoding="utf-8").strip() == "outputs/p27_whu_detectors/bit/external_best.pth"
    assert json.loads((pointer.parent / "external_check.json").read_text(encoding="utf-8"))["differences"] == []
    with pytest.raises(SystemExit, match="already exists"):
        run("--checkpoint", "same.pth", "--install", "--manifest-sha256", manifest_sha)


def test_diffregcd_diagnosis_views_and_loop(tmp_path) -> None:
    from rs_cd.affine_f1 import scoring_mask
    from tools.p27_diffregcd_diagnosis import LEVELS, VIEWS, WIRINGS, diagnose, render, views

    image = _textured(64)
    matrix = perturbation_matrix(0.0, 1.1, 6.0, -4.0, width=64, height=64)
    shown = views(image, matrix)
    inside = scoring_mask([matrix], 64)
    assert np.array_equal(shown["clean"], image)
    assert np.all(shown["border_only"][~inside] == 0) and np.array_equal(shown["border_only"][inside], image[inside])
    assert np.array_equal(shown["resample_only"][~inside], image[~inside])
    assert np.array_equal(shown["resample_only"][inside], shown["oracle"][inside])
    assert np.array_equal(shown["raw"], perturb(image, matrix))

    _, _, evaluation, _, _ = _toy(tmp_path)
    spec = {**_test_split(evaluation)["datasets"][1]}

    def infer(a, b):
        difference = (a - b).abs().mean(dim=1, keepdim=True) * 4.0 - 1.0
        two = torch.cat([-difference, difference], dim=1)
        return {"official_unmodified": two, "intended_warped": -two}

    result = diagnose(spec, evaluation, infer, seed=3407, threshold=0.5, device=torch.device("cpu"), max_crops=2)
    assert result["crops"] == 2
    assert set(result["f1"]) == {f"{l}/{v}/{w}" for l in LEVELS for v in VIEWS for w in WIRINGS}
    assert result["f1"]["1/clean/official_unmodified"] == result["f1"]["3/clean/official_unmodified"]
    assert "intended_warped" in render("small", result)
