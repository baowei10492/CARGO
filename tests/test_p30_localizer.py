"""P30 localizer v2: border-free training targets and hold-out."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import cut_border_free, in_frame
from rs_cd.affine_registration import perturbation_matrix
from rs_cd.datasets.localizer_pairs import LocalizerPairDataset, PerturbationMixture

from p27_helpers import TinyBackbone, make_dataset, tile
from test_p27_training import NO_JITTER

ROOT = Path(__file__).resolve().parents[1]


def test_region_cut_equals_the_tile_warp_and_has_no_border() -> None:
    import cv2

    big = tile(512, 7)
    top, left, size, margin = 150, 120, 128, 48
    matrix = perturbation_matrix(0.0, 0.8, 12.0, -12.0, width=size, height=size)
    region = big[top - margin:top + size + margin, left - margin:left + size + margin]
    cut = cut_border_free(region, matrix, margin, size)
    warped = cv2.warpAffine(big, in_frame(matrix, top, left), (512, 512), flags=cv2.INTER_LINEAR)
    assert np.array_equal(cut, warped[top:top + size, left:left + size])
    assert (cut.sum(axis=2) == 0).sum() == 0


def test_config_differs_from_v1_only_in_the_margins() -> None:
    v1 = yaml.safe_load((ROOT / "configs/p27_localizer_v1.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((ROOT / "configs/p30_localizer_v2.yaml").read_text(encoding="utf-8"))
    assert v2["data"].pop("border_free_margin") == 96 and v2["holdout"].pop("border_free_margin") == 64
    assert v2 == v1
    mixture = v1["data"]["perturbation"]
    needed = (128 + mixture["large_translation"]) / min(mixture["large_scale"]) - 128
    assert needed < 96


def test_border_free_samples_have_no_zero_border(tmp_path) -> None:
    source = make_dataset(tmp_path, "big", 4, 512)
    mixture = PerturbationMixture(identity=0.0, small=0.0, large_translation=32.0, large_scale=(0.75, 1.333))
    dataset = LocalizerPairDataset([source], samples_per_epoch=6, crop_size=256, holdout_fraction=0.0, seed=1,
                                   mixture=mixture, jitter=NO_JITTER, border_free_margin=96)
    black = ((0.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])).astype(np.float32)
    for index in range(len(dataset)):
        sample = dataset[index]
        for key in ("target1", "target2"):
            image = sample[key].numpy()
            is_black = np.all(np.isclose(image, black[:, None, None], atol=1e-4), axis=0)
            assert is_black.sum() == 0, key
        assert not np.allclose(sample["matrix1"].numpy(), np.eye(2, 3))
    again = LocalizerPairDataset([source], samples_per_epoch=6, crop_size=256, holdout_fraction=0.0, seed=1,
                                 mixture=mixture, jitter=NO_JITTER, border_free_margin=96)
    assert torch.equal(dataset[3]["target1"], again[3]["target1"])


def test_v2_training_runs_with_a_border_free_holdout(tmp_path) -> None:
    from tools.train_similarity_localizer import build_holdout, train

    config = yaml.safe_load((ROOT / "configs/p30_localizer_v2.yaml").read_text(encoding="utf-8"))
    sources = [make_dataset(tmp_path, "big", 20, 1024), make_dataset(tmp_path, "small", 20, 512)]
    config["sources"] = [dict(s.__dict__) for s in sources]
    config["holdout"].update(fraction=0.3, crops_per_source=3)
    config["data"]["num_workers"] = 0
    config["model"].update(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1)
    config["loss"]["stage_weights"] = [0.5, 1.0]
    config["optim"].update(batch_size=2, warmup_steps=1, amp=False)
    config.update(log_every=1, eval_every=2)
    holdout = build_holdout(config)
    for items in holdout.values():
        for item in items:
            assert item["margin"] == 64 and item["region"].shape[:2] == (256 + 128, 256 + 128)
            top, left = (int(v) for v in item["crop_id"].rsplit("|", 1)[1][1:].split("x"))
            assert top >= 64 and left >= 64
    output = tmp_path / "run"
    output.mkdir()
    summary = train(config, output, torch.device("cpu"), backbone=TinyBackbone(), max_steps=2)
    assert summary["steps"] == 2
    result = json.loads((output / "holdout_step000002.json").read_text())
    assert set(result["summaries"]) == {"big", "small"} and np.isfinite(result["score"])
