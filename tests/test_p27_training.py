"""P27 version 1: training samples, loss and a short end-to-end training run."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rs_cd.affine_registration import perturb
from rs_cd.datasets.localizer_pairs import (
    LocalizerPairDataset,
    PerturbationMixture,
    Photometric,
    Source,
    SyntheticChange,
    in_holdout,
    normalize,
    paste_changes,
    photometric,
    sample_matrix,
)
from rs_cd.localizer_training import load_localizer, localizer_loss, misalignment_target
from rs_cd.similarity_geometry import grid

from p27_helpers import TinyBackbone, make_dataset, tile

ROOT = Path(__file__).resolve().parents[1]
NO_JITTER = Photometric(brightness=0.0, contrast=(1.0, 1.0), channel_gain=(1.0, 1.0), gamma=(1.0, 1.0),
                        blur_probability=0.0, noise_probability=0.0)


def test_perturbation_mixture_proportions_and_ranges() -> None:
    rng = np.random.default_rng(0)
    mixture = PerturbationMixture()
    matrices = [sample_matrix(rng, mixture, 256) for _ in range(3000)]
    identity = [np.allclose(m, np.eye(2, 3)) for m in matrices]
    assert 0.26 < np.mean(identity) < 0.34
    scales = np.array([m[0, 0] for m in matrices])
    assert scales.min() >= 0.75 - 1e-9 and scales.max() <= 1.333 + 1e-9
    assert all(abs(m[0, 1]) < 1e-12 and abs(m[1, 0]) < 1e-12 for m in matrices)  # no rotation


def test_paste_changes_respects_the_area_budget() -> None:
    rng = np.random.default_rng(1)
    target, donor = tile(256, 1), tile(256, 2)
    settings = SyntheticChange(probability=1.0, max_fraction=0.3, max_patches=10, min_size=40, max_size=90)
    pasted, mask = paste_changes(target, donor, rng, settings)
    assert mask.mean() <= 0.3 + 1e-9 and mask.any()
    assert np.array_equal(pasted[~mask], target[~mask])
    assert not np.array_equal(pasted[mask], target[mask])


def test_photometric_is_seeded_and_bounded() -> None:
    image = tile(64, 3)
    first = photometric(image, np.random.default_rng(5), Photometric())
    second = photometric(image, np.random.default_rng(5), Photometric())
    assert np.array_equal(first, second) and first.dtype == np.uint8 and first.shape == image.shape
    assert np.array_equal(photometric(image, np.random.default_rng(5), NO_JITTER), image)


def test_holdout_hash_is_stable_and_close_to_the_fraction() -> None:
    ids = [f"tile{i}" for i in range(4000)]
    flags = [in_holdout(i, 0.05, 3407) for i in ids]
    assert 0.035 < np.mean(flags) < 0.065
    assert flags == [in_holdout(i, 0.05, 3407) for i in ids]
    assert flags != [in_holdout(i, 0.05, 1) for i in ids]


def test_samples_are_deterministic_and_geometrically_exact(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 6, 512)
    dataset = LocalizerPairDataset([source], samples_per_epoch=10, holdout_fraction=0.0,
                                   synthetic=SyntheticChange(probability=0.0), jitter=NO_JITTER)
    first, again = dataset[3], dataset[3]
    for key in ("reference", "target1", "matrix1", "change"):
        assert torch.equal(first[key], again[key])
    assert first["reference"].shape == (3, 256, 256) and first["change"].shape == (1, 256, 256)
    # Without jitter, target1 is exactly the perturbed self-reference.
    t2 = (first["self_reference"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
          + np.array([0.485, 0.456, 0.406])) * 255.0
    t2 = np.clip(np.round(t2), 0, 255).astype(np.uint8)
    expected = normalize(perturb(t2, first["matrix1"].numpy().astype(np.float64)))
    assert torch.allclose(first["target1"], expected, atol=1e-5)
    with pytest.raises(ValueError):
        LocalizerPairDataset([Source(**{**source.__dict__, "split": "val"})], samples_per_epoch=1)


def test_synthetic_change_enters_the_label(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 4, 512)
    dataset = LocalizerPairDataset([source], samples_per_epoch=20, holdout_fraction=0.0,
                                   synthetic=SyntheticChange(probability=1.0, max_fraction=0.4), jitter=NO_JITTER)
    fractions = [float(dataset[i]["change"].mean()) for i in range(20)]
    # Real labels cover at most 50 x 50 px of a crop (3.8%); pasted patches add more.
    assert max(fractions) > 0.06


def test_holdout_tiles_never_reach_training(tmp_path) -> None:
    source = make_dataset(tmp_path, "toy", 40, 256)
    dataset = LocalizerPairDataset([source], samples_per_epoch=1, holdout_fraction=0.5)
    kept = {dataset.datasets[0].rows[i].sample_id for i in dataset.rows[0]}
    assert kept and all(not in_holdout(sample_id, 0.5, 3407) for sample_id in kept)
    assert len(kept) < 40


def _stages(matrix, count=3):
    return {"stage_matrices": [matrix.clone() for _ in range(count)],
            "change_logits": torch.zeros(matrix.shape[0], 4, 4), "misaligned_logit": torch.zeros(matrix.shape[0])}


def test_loss_vanishes_for_exact_estimates_and_ignores_a_shared_offset() -> None:
    rng = np.random.default_rng(2)
    m1 = torch.tensor(np.stack([sample_matrix(rng, PerturbationMixture(identity=0.0, small=0.0), 256)
                                for _ in range(4)])).float()
    m2 = torch.tensor(np.stack([sample_matrix(rng, PerturbationMixture(identity=0.0, small=0.0), 256)
                                for _ in range(4)])).float()
    points = grid(256, 256, 9)
    weights = {"relative": 1.0, "absolute": 0.0, "self": 0.0, "change": 0.0, "misaligned": 0.0}
    change = torch.zeros(4, 1, 16, 16)
    exact, _ = localizer_loss(_stages(m1), _stages(m2), _stages(m1), m1, m2, change,
                              weights=weights, stage_weights=[0.5, 0.75, 1.0], points=points)
    assert float(exact) < 1e-3
    # A shared intrinsic offset G (A_k = M_k o G) leaves the relative term unchanged.
    offset = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, -1.5]]]).repeat(4, 1, 1)
    from rs_cd.similarity_geometry import compose

    shifted, parts = localizer_loss(_stages(compose(m1, offset)), _stages(compose(m2, offset)), _stages(m1),
                                    m1, m2, change, weights=weights, stage_weights=[0.5, 0.75, 1.0], points=points)
    assert float(shifted) < 1e-3
    weights["absolute"] = 1.0
    with_absolute, parts = localizer_loss(_stages(compose(m1, offset)), _stages(compose(m2, offset)), _stages(m1),
                                          m1, m2, change, weights=weights, stage_weights=[0.5, 0.75, 1.0],
                                          points=points)
    assert parts["absolute"] > 1.0 and parts["relative"] < 1e-3
    targets = misalignment_target(torch.eye(2, 3)[None], points)
    assert float(targets) == 0.0


def test_training_runs_end_to_end_and_checkpoints_reload(tmp_path) -> None:
    from tools.train_similarity_localizer import train

    config = yaml.safe_load((ROOT / "configs/p27_localizer_v1.yaml").read_text(encoding="utf-8"))
    sources = [make_dataset(tmp_path, "big", 20, 1024), make_dataset(tmp_path, "small", 20, 512)]
    config["sources"] = [dict(s.__dict__) for s in sources]
    config["holdout"].update(fraction=0.3, crops_per_source=3)
    config["data"]["num_workers"] = 0
    config["model"].update(backbone_channels=[12, 24], coarse_dim=16, fine_dim=8, fine_iterations=1)
    config["loss"]["stage_weights"] = [0.5, 1.0]
    config["optim"].update(batch_size=2, warmup_steps=1, amp=False)
    config.update(log_every=1, eval_every=2)
    output = tmp_path / "run"
    output.mkdir()
    summary = train(config, output, torch.device("cpu"), backbone=TinyBackbone(), max_steps=2)
    assert summary["steps"] == 2 and summary["best"]["step"] == 2
    records = [json.loads(line) for line in (output / "train_log.jsonl").read_text().splitlines()]
    assert records and all(np.isfinite(r["total"]) for r in records)
    holdout = json.loads((output / "holdout_step000002.json").read_text())
    assert set(holdout["summaries"]) == {"big", "small"}
    assert np.isfinite(holdout["score"])
    model = load_localizer(str(output / "checkpoints" / "best.pth"), torch.device("cpu"), backbone=TinyBackbone())
    assert model.fine_iterations == 1
