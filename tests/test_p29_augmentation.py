"""P29: training-time misalignment augmentation matching the P27 test perturbation."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from rs_cd.affine_registration import AffineRanges, affine_parameters, perturb, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.levir import affine_misregistration_matrix
from rs_cd.losses import BinaryChangeLoss
from rs_cd.robustness import _stable_seed

from p27_helpers import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matrix_is_the_p27_test_perturbation() -> None:
    ranges = AffineRanges(0.0, 0.80, 1.25, 25.0, 3)
    for key in ("a|y0000x0000", "b|y0256x0512", "c|y0512x0256"):
        u = np.random.default_rng(_stable_seed(3407, key, "affine", 0, "shared")).random(4)
        for level in (1, 2, 3):
            angle, scale, tx, ty = affine_parameters(key, level, ranges, 3407)
            assert angle == 0.0
            expected = perturbation_matrix(angle, scale, tx, ty, width=256, height=256)
            actual = affine_misregistration_matrix(level, u[1], u[2], u[3], size=256, max_translation_px=25.0,
                                                   scale_range=(0.80, 1.25), levels=3)
            assert np.allclose(actual, expected, atol=1e-9)


def _dataset(source, **overrides) -> PairedChangeDataset:
    options = dict(split="train", image_size=None, normalization="none", augment=True, random_flip=False,
                   random_rotate90=False, affine_misregistration_probability=1.0)
    options.update(overrides)
    return PairedChangeDataset(source.dataset_root, source.manifest, **options)


def test_dataset_perturbs_t2_and_marks_pixels_without_content(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 2, 64)
    # A fixed shrink, so that part of the crop is left without T2 content.
    dataset = _dataset(source, affine_max_translation_px=6.0, affine_scale_range=(0.80, 0.80))
    plain = _dataset(source, augment=False, affine_misregistration_probability=0.0)
    random.seed(11)
    sample = dataset[0]
    random.seed(11)
    random.random()  # the probability draw
    level = random.randint(1, 3)
    matrix = affine_misregistration_matrix(level, random.random(), random.random(), random.random(), size=64,
                                           max_translation_px=6.0, scale_range=(0.80, 0.80), levels=3)
    reference = plain[0]
    t2 = reference["image_t2"].permute(1, 2, 0).numpy().astype(np.uint8)
    assert sample["affine_level"] == level
    assert np.array_equal(sample["image_t2"].permute(1, 2, 0).numpy().astype(np.uint8), perturb(t2, matrix))
    assert torch.equal(sample["image_t1"], reference["image_t1"]) and torch.equal(sample["mask"], reference["mask"])
    valid = sample["valid_mask"][0].numpy()
    assert set(np.unique(valid)) <= {0.0, 1.0} and 0.3 < valid.mean() < 1.0
    # Beyond the one-pixel bilinear edge, pixels outside the valid mask carry no T2 content.
    near = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert sample["image_t2"][:, ~near].abs().max() == 0


def test_augmentation_is_off_by_default_and_exclusive(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 1, 64)
    assert "valid_mask" not in _dataset(source, affine_misregistration_probability=0.0)[0]
    assert "valid_mask" not in _dataset(source, augment=False)[0]
    clean = _dataset(source, affine_misregistration_probability=1e-12)[0]
    assert clean["affine_level"] == 0 and bool(clean["valid_mask"].all())
    with pytest.raises(ValueError, match="either"):
        _dataset(source, misregistration_probability=0.5)


def test_masked_loss_ignores_pixels_without_content() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = (torch.rand(2, 1, 16, 16) > 0.7).float()
    criterion = BinaryChangeLoss(pos_weight=2.0)
    unmasked, _ = criterion({"change_logits": logits}, targets)
    full, _ = criterion({"change_logits": logits}, targets, valid=torch.ones_like(targets))
    assert torch.allclose(unmasked, full, atol=1e-6)
    valid = torch.ones_like(targets)
    valid[..., :4] = 0
    masked, _ = criterion({"change_logits": logits}, targets, valid=valid)
    masked.backward()
    assert logits.grad[..., :4].abs().max() == 0 and logits.grad[..., 4:].abs().max() > 0
    moved = logits.detach().clone()
    moved[..., :4] += 5.0
    again, _ = criterion({"change_logits": moved}, targets, valid=valid)
    assert torch.allclose(masked.detach(), again, atol=1e-6)


def test_training_loop_passes_the_valid_mask() -> None:
    from rs_cd.engine import train_one_epoch
    from rs_cd.runtime import create_grad_scaler

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, t1, t2, targets=None, batch=None):
            return {"change_logits": (t1 - t2).mean(dim=1, keepdim=True) + self.bias}

    seen = []

    class Criterion(BinaryChangeLoss):
        def forward(self, outputs, targets, valid=None):
            seen.append(valid)
            return super().forward(outputs, targets, valid=valid)

    batch = {"image_t1": torch.rand(2, 3, 8, 8), "image_t2": torch.rand(2, 3, 8, 8),
             "mask": torch.zeros(2, 1, 8, 8), "valid_mask": torch.ones(2, 1, 8, 8)}
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")
    for loader in ([batch], [{k: v for k, v in batch.items() if k != "valid_mask"}]):
        train_one_epoch(model, loader, optimizer, Criterion(), device, create_grad_scaler(device, enabled=False),
                        False, None, 1, 1)
    assert isinstance(seen[0], torch.Tensor) and seen[1] is None


def _without(config: dict, keys: dict[str, set[str]]) -> dict:
    return {block: {k: v for k, v in values.items() if k not in keys.get(block, set())}
            if isinstance(values, dict) else values for block, values in config.items()}


def test_training_configs_change_only_the_augmentation() -> None:
    augmentation = {"affine_misregistration_probability", "affine_max_translation_px", "affine_scale_range",
                    "affine_levels"}
    evaluation = yaml.safe_load((ROOT / "configs/p27_eval_perturbation.yaml").read_text(encoding="utf-8"))
    roots = {spec["name"]: spec for spec in evaluation["datasets"]}
    for name, original, dataset in (("levir", "p3b_dinov3_convnext_tiny_partial_ft", "LEVIR-CD"),
                                    ("dsifn", "p15_dsifn_p3b", "DSIFN-CD"),
                                    ("whu", "p8_whu_dinov3_partial_ft", "WHU-CD")):
        p29 = yaml.safe_load((ROOT / f"configs/p29/dinov3_aug_{name}.yaml").read_text(encoding="utf-8"))
        base = yaml.safe_load((ROOT / f"configs/{original}.yaml").read_text(encoding="utf-8"))
        assert p29["experiment"]["name"] == f"p29_dinov3_aug_{name}"
        assert p29["data"]["affine_misregistration_probability"] == 0.75
        assert p29["data"]["affine_max_translation_px"] == evaluation["perturbation"]["max_translation_px"]
        assert p29["data"]["affine_scale_range"] == evaluation["perturbation"]["scale_range"]
        assert p29["data"]["affine_levels"] == len(evaluation["perturbation"]["levels"])
        assert evaluation["perturbation"]["max_rotation_deg"] == 0.0
        assert (p29["data"]["dataset_root"], p29["data"]["manifest"], p29["data"]["mask_threshold"]) == \
            (roots[dataset]["dataset_root"], roots[dataset]["manifest"], roots[dataset]["mask_threshold"])
        ignored = {"experiment": {"name"}, "data": augmentation | {"dataset_root"}}
        assert _without(p29, ignored) == _without(base, ignored)


def test_dinov3aug_row_chains(monkeypatch) -> None:
    from tools.run_p27_f1 import PLUGIN_MARKER_ROOT, load_config, marker_dir
    from tools.run_p27_selector import load_config as load_selector_config
    from tools.run_p27_selector import load_test_config

    monkeypatch.chdir(ROOT)
    paths = {kind: ROOT / "configs/p27_plugins" / f"dinov3aug_{kind}.yaml" for kind in ("f1", "selector", "test")}
    f1_config, evaluation = load_config(paths["f1"])
    assert [spec["name"] for spec in evaluation["datasets"]] == ["LEVIR-CD", "DSIFN-CD", "WHU-CD"]
    assert f1_config["detectors"] == {name: {"checkpoint_pointer": f"outputs/p29_aug_detectors/{short}/CHECKPOINT"}
                                      for name, short in (("LEVIR-CD", "levir"), ("DSIFN-CD", "dsifn"),
                                                          ("WHU-CD", "whu"))}
    assert marker_dir(f1_config) == PLUGIN_MARKER_ROOT / "dinov3aug" / "frozen"
    dinov3 = yaml.safe_load((ROOT / "configs/p27_f1_gate.yaml").read_text(encoding="utf-8"))
    assert f1_config["decision"] == dinov3["decision"] and f1_config["instrument"] == dinov3["instrument"]
    config, chained_f1, _ = load_selector_config(paths["selector"])
    assert chained_f1 == f1_config
    dinov3_selector = yaml.safe_load((ROOT / "configs/p27_selector.yaml").read_text(encoding="utf-8"))
    for key in ("label", "weights", "classifier", "risk", "calibration", "evaluation", "training_crops",
                "localizer_run", "localizer_checkpoint", "seed", "anchor_tolerance"):
        assert config[key] == dinov3_selector[key], key
    load_test_config(paths["test"], _sha(paths["selector"]))


def test_preflight_flags_a_different_split(tmp_path, monkeypatch, capsys) -> None:
    import tools.p29_preflight as preflight

    source = make_dataset(tmp_path, "toy", 2, 64)
    monkeypatch.setitem(preflight.MANIFESTS, "toy", _sha(Path(source.manifest)))
    assert preflight.check_dataset("toy", Path(source.dataset_root), Path(source.manifest)) == []
    monkeypatch.setitem(preflight.MANIFESTS, "toy", "0" * 64)
    assert "split differs" in preflight.check_dataset("toy", Path(source.dataset_root), Path(source.manifest))[0]
    (Path(source.dataset_root) / "B" / "toy_1.png").unlink()
    problems = preflight.check_dataset("toy", Path(source.dataset_root), Path(source.manifest))
    assert any("missing" in p for p in problems)
