"""P30 evaluation: whole-tile geometry, border-free records, and the three commands."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from rs_cd.affine_crops import in_frame
from rs_cd.affine_registration import perturb, perturbation_matrix
from rs_cd.p30_geometry import BorderFreeGeometry

from test_p27_f1 import DifferenceDetector, _tiny_localizer
from test_p27_selector import _toy


def _tile(size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(3)
    small = rng.integers(20, 236, (size // 16, size // 16, 3), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)


def test_perturbed_crop_is_cut_from_the_warped_tile() -> None:
    tile = _tile()
    geometry = BorderFreeGeometry(tile, 64, 192, 256)
    matrix = perturbation_matrix(0.0, 0.80, 25.0, -25.0, width=256, height=256)
    perturbed = geometry.perturb(matrix)
    whole = cv2.warpAffine(tile, in_frame(matrix, 64, 192), (512, 512), flags=cv2.INTER_LINEAR)
    assert np.array_equal(perturbed, whole[64:320, 192:448])
    assert perturbed.min() >= 20                                   # real content up to every edge
    # Where the P27 crop warp has content, the two agree.
    crop = tile[64:320, 192:448]
    p27 = perturb(crop, matrix)
    inside = cv2.erode((p27.min(axis=2) > 0).astype(np.uint8), np.ones((5, 5), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    assert inside.mean() > 0.5
    assert np.abs(p27.astype(int) - perturbed.astype(int))[inside].max() <= 1
    assert (p27.min(axis=2) == 0).any()                            # the zero border P30 removes
    assert np.array_equal(geometry.perturb(np.eye(2, 3)), crop)
    with pytest.raises(SystemExit, match="outside the tile"):
        BorderFreeGeometry(tile, 0, 0, 256).perturb(matrix)


def test_correction_uses_the_tile_and_counts_uncovered_pixels() -> None:
    tile = _tile()
    geometry = BorderFreeGeometry(tile, 64, 192, 256)
    crop = tile[64:320, 192:448].astype(int)
    truth = perturbation_matrix(0.0, 1.20, -20.0, 15.0, width=256, height=256)
    views = geometry.path_images(None, truth, truth)
    assert list(views) == ["raw", "oracle", "estimated"]
    assert geometry.missing_pixels(truth) == 0
    assert views["oracle"].min() >= 20
    assert np.abs(views["oracle"].astype(int) - crop).mean() < 0.5 * np.abs(views["raw"].astype(int) - crop).mean()
    # An estimate reaching past the perturbed tile leaves pixels without content, and they are counted.
    far = perturbation_matrix(0.0, 1.0, 120.0, 0.0, width=256, height=256)
    estimated = geometry.path_images(None, truth, far)["estimated"]
    missing = geometry.missing_pixels(truth)
    assert missing > 0
    assert abs(int((estimated.max(axis=2) == 0).sum()) - missing) <= 2 * 256


class ScaledDetector(DifferenceDetector):
    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain

    def forward(self, t1, t2):
        return {"change_logits": (t1 - t2).abs().mean(dim=1, keepdim=True) * self.gain - 2.0}


def _paint_changes(evaluation) -> None:
    """The toy tiles change only in their top-left corner, outside every margin crop: add changes everywhere."""
    import csv

    from PIL import Image

    for spec in evaluation["datasets"]:
        root = Path(spec["dataset_root"])
        with open(spec["manifest"], newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            t2 = np.array(Image.open(root / row["t2_path"]))
            mask = np.zeros(t2.shape[:2], np.uint8)
            for top in range(80, t2.shape[0] - 40, 96):
                for left in range(80, t2.shape[1] - 40, 96):
                    mask[top:top + 30, left:left + 30] = 255
            t2[mask > 0] = 255 - t2[mask > 0]
            Image.fromarray(t2).save(root / row["t2_path"])
            Image.fromarray(mask).save(root / row["mask_path"])


def _setup(tmp_path, monkeypatch):
    import tools.run_p30 as tool

    selector, _, evaluation, localizer_config, detectors = _toy(tmp_path)
    _paint_changes(evaluation)
    names = [spec["name"] for spec in evaluation["datasets"]]
    checkpoints = {}
    for which in ("p30", "p27"):
        checkpoints[which] = tmp_path / f"localizer_{which}.pth"
        checkpoints[which].write_text(which)
    groups = {"original": detectors}
    pointers = {}
    for group, gain in (("p29_augmented", 3.0), ("p30_augmented", 5.0)):
        groups[group] = {name: {**detectors[name], "model": ScaledDetector(gain), "sha256": f"{group}_{name}"}
                         for name in names}
    for name in names:
        (tmp_path / f"aug_{name}.pth").write_text(name)
        pointers[name] = tmp_path / f"aug_{name}_CHECKPOINT"
        pointers[name].write_text(str(tmp_path / f"aug_{name}.pth"))
    config = {"crop_margin": 64, "selector_config_sha256": "recipe",
              "p27_selector_bundles": {"pointer": str(tmp_path / "p27_LATEST")},
              "detectors": {"p30_augmented": {name: {"checkpoint_pointer": str(pointers[name])} for name in names}}}
    monkeypatch.setattr(tool, "localizer_checkpoint", lambda config, which: checkpoints[which])
    monkeypatch.setattr(tool, "open_localizer", lambda config, which, device: (_tiny_localizer(), localizer_config))
    monkeypatch.setattr(tool, "open_detector", lambda config, group, name, device: groups[group][name])
    monkeypatch.setattr(tool, "MARKER_DIR", tmp_path / "markers")
    return tool, config, evaluation, selector, groups, checkpoints


def test_features_train_and_evaluate_once(tmp_path, monkeypatch) -> None:
    from tools.run_p27_selector import harness, open_dataset, read_records
    from tools.run_p27_step1 import sha256

    tool, config, evaluation, selector, groups, checkpoints = _setup(tmp_path, monkeypatch)
    device = torch.device("cpu")
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    summary = tool.run_features(config, evaluation, selector, device, features_dir)
    (features_dir / "features_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name, entry in summary["datasets"].items():
        assert entry["records"] == 4 * sum(entry["crops"].values())
        assert all(entry["crops"][split] > 0 for split in ("fit", "early_stop", "risk_calibration")), entry
        records = read_records(features_dir / f"features_{name}.csv")
        assert all("estimated_missing_px" in record for record in records)
        for record in records:           # every crop keeps the 64 px margin
            top, left = int(record["crop_id"][-9:-5]), int(record["crop_id"][-4:])
            assert top >= 64 and left >= 64

    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    tool.run_train(config, evaluation, selector, features_dir, bundle_dir)
    p27_bundles = tmp_path / "p27_train" / "bundles"
    p27_bundles.mkdir(parents=True)
    for spec in evaluation["datasets"]:
        bundle = torch.load(bundle_dir / f"selector_{spec['name']}.pth", weights_only=False)
        assert bundle["protocol"] == "p30" and bundle["localizer_sha256"] == sha256(checkpoints["p30"])
        bundle.pop("protocol")
        bundle["localizer_sha256"] = sha256(checkpoints["p27"])
        torch.save(bundle, p27_bundles / f"selector_{spec['name']}.pth")
    (tmp_path / "p27_LATEST").write_text(str(tmp_path / "p27_train"))

    config_file = tmp_path / "p30.yaml"
    config_file.write_text("p30")
    monkeypatch.setattr(tool, "load", lambda path: (dict(config), evaluation, selector))

    calls = iter(range(10))

    def evaluate(split: str) -> Path:
        output = tmp_path / f"evaluate_{split}_{next(calls)}"
        monkeypatch.setattr("sys.argv", ["run", "evaluate", "--config", str(config_file), "--bundle-dir",
                                         str(bundle_dir), "--split", split, "--device", "cpu",
                                         "--output-dir", str(output)])
        tool.main()
        return output

    with pytest.raises(SystemExit, match="validation"):
        evaluate("test")
    output = evaluate("val")
    decision = json.loads((output / "p30_val.json").read_text(encoding="utf-8"))
    assert decision["test_split_touched"] is False
    h = harness(evaluation)
    for spec in evaluation["datasets"]:
        name = spec["name"]
        entry = decision["datasets"][name]
        assert set(tool.rows_of(entry)) and all(set(row) == {"0", "1", "2", "3"} for row in tool.rows_of(entry).values())
        assert entry["original"] == {k: v["raw"] for k, v in entry["reference_p27"]["f1"].items()}
        dataset = open_dataset(spec, "val")
        crops = tool.margin_crops(dataset, spec, h["size"], 64)
        assert entry["crops"] == len(crops)
        again = tool.raw_scores(dataset, spec, crops, groups["original"][name], h, device)
        assert all(abs(again[level] - entry["original"][level]) < 1e-9 for level in again)
        assert entry["p30_augmented"] != entry["p29_augmented"]
        assert (output / f"records_proposed_{name}.csv").is_file()
        assert "reference_p27" in tool.render(name, entry)
    with pytest.raises(SystemExit, match="already"):
        evaluate("val")

    test_output = evaluate("test")
    assert json.loads((test_output / "p30_test.json").read_text(encoding="utf-8"))["test_split_touched"] is True
    with pytest.raises(SystemExit, match="used already"):
        evaluate("test")


def test_raw_scores_see_no_zero_border(tmp_path, monkeypatch) -> None:
    from tools.run_p27_selector import harness, open_dataset

    tool, config, evaluation, selector, groups, _ = _setup(tmp_path, monkeypatch)
    seen = []

    class Recording(DifferenceDetector):
        def forward(self, t1, t2):
            seen.append(t2.clone())
            return super().forward(t1, t2)

    spec = evaluation["datasets"][1]
    dataset = open_dataset(spec, "val")
    h = harness(evaluation)
    crops = tool.margin_crops(dataset, spec, h["size"], 64)
    detector = {**groups["original"][spec["name"]], "model": Recording()}
    tool.raw_scores(dataset, spec, crops, detector, h, torch.device("cpu"))
    zero = tool.normalize(np.zeros((1, 1, 3), np.uint8))[:, 0, 0]
    for batch in seen:
        black = (batch - zero[None, :, None, None]).abs().amax(dim=1) < 1e-6
        edges = torch.cat([black[:, 0], black[:, -1], black[:, :, 0], black[:, :, -1]], dim=1)
        assert not bool(edges.all(dim=1).any())      # no image has a black edge line
        assert black.float().mean() < 0.01
