"""DiffRegCD re-run: wiring of every part, the mechanism views, the loop and the readings."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_margin_crops
from rs_cd.affine_registration import perturb
from rs_cd.datasets import PairedChangeDataset
from rs_cd.p30_geometry import BorderFreeGeometry
from tools.p27_diffregcd_rerun import (
    MECHANISM_VIEWS,
    Views,
    border_reading,
    dates_reading,
    harness,
    matrices_for,
    run_part,
    with_border_only,
)

from p27_helpers import make_dataset

ROOT = Path(__file__).resolve().parents[1]


def _evaluation() -> dict:
    return yaml.safe_load((ROOT / "configs/p27_eval_perturbation.yaml").read_text(encoding="utf-8"))


def _spec(tmp_path, tiles: int = 3, size: int = 512) -> dict:
    source = make_dataset(tmp_path, "toy", tiles, size, split="test")
    return {"name": "LEVIR-CD", "dataset_root": source.dataset_root, "manifest": source.manifest, "split": "test",
            "tile_size": size, "mask_threshold": 127}


def _dataset(spec: dict) -> PairedChangeDataset:
    return PairedChangeDataset(spec["dataset_root"], spec["manifest"], split="test", normalization="unit",
                               mask_threshold=127)


def test_config_chain_and_parts() -> None:
    from tools.p27_diffregcd_rerun import load

    rerun, config, evaluation = load(ROOT / "configs/p27_diffregcd_rerun.yaml")
    assert rerun["parts"] == ["mechanism", "border_free", "bordered"] and rerun["crop_margin"] == 64
    assert config["split"] == "test" and set(rerun["mechanism_crops"]) == set(config["datasets"])


@pytest.mark.parametrize("part", ["bordered", "border_free"])
def test_the_perturbed_image_goes_to_a_and_t2_stays_in_b(tmp_path, part) -> None:
    spec, evaluation = _spec(tmp_path), _evaluation()
    views = Views(part, _dataset(spec), spec, evaluation, margin=64, cap=None)
    size, levels, ranges, seed = harness(evaluation)
    item = views.crops[1]
    t1, t2, _ = views.cache.crop(item, size)
    made, valid = views.make(1, t1, t2)
    matrices = matrices_for(item["crop_id"], levels, ranges, seed, size)
    keys = [key for key, _, _ in made]
    assert keys[:4] == ["0", "1", "2", "3"] and valid.shape == (size, size)
    assert made[0][1] is t1 and all(b is t2 for _, _, b in made)
    if part == "bordered":
        assert len(made) == 4
        for level in levels:
            assert np.array_equal(made[level][1], perturb(t1, matrices[level]))
    else:
        assert keys[4] == "bordered_3" and np.array_equal(made[4][1], perturb(t1, matrices[3]))
        geometry = BorderFreeGeometry(views.cache.arrays[0], item["top"], item["left"], size)
        for level in levels:
            assert np.array_equal(made[level][1], geometry.perturb(matrices[level]))
            assert made[level][1].max(axis=2)[[0, -1], :].min() > 0          # no black edge rows
        assert item["top"] >= 64 and item["left"] >= 64
        assert views.crops == select_margin_crops(_dataset(spec).rows, 512, size, 64)


def test_mechanism_views(tmp_path) -> None:
    spec, evaluation = _spec(tmp_path), _evaluation()
    views = Views("mechanism", _dataset(spec), spec, evaluation, margin=64, cap=None)
    size, levels, ranges, seed = harness(evaluation)
    index = 0
    item = views.crops[index]
    t1, t2, _ = views.cache.crop(item, size)
    made, _ = views.make(index, t1, t2)
    by_key = {key: (a, b) for key, a, b in made}
    assert tuple(by_key) == MECHANISM_VIEWS and all(b is t2 for _, b in by_key.values())
    top = matrices_for(item["crop_id"], levels, ranges, seed, size)[levels[-1]]
    assert np.array_equal(by_key["normal"][0], perturb(t1, top))
    assert np.array_equal(by_key["same_date"][0], perturb(t2, top))
    assert not by_key["black"][0].any()
    border = by_key["border_only"][0]
    covered = perturb(np.full_like(t1, 255), top).max(axis=2) > 0
    assert np.array_equal(border[covered], t1[covered]) and not border[~covered].any() and (~covered).any()
    other = views.other(index)
    assert other["row"] != item["row"]
    other_t1 = TileCache(_dataset(spec), 512).crop(other, size)[0]
    assert np.array_equal(by_key["unrelated"][0], perturb(other_t1, top))
    assert np.array_equal(with_border_only(t1, np.eye(2, 3)), t1)


def test_run_part_and_readings(tmp_path) -> None:
    spec, evaluation = _spec(tmp_path), _evaluation()

    def infer(a, b):
        change = (a - b).abs().mean(dim=1, keepdim=True) * 8.0 - 1.0
        logits = torch.cat([torch.zeros_like(change), change], dim=1)
        return {"official_unmodified": logits, "intended_warped": logits * 0.5}

    results = {}
    for part in ("mechanism", "border_free", "bordered"):
        results[part] = run_part(part, spec, evaluation, infer, margin=64, cap=None, seed=3407, threshold=0.5,
                                 device=torch.device("cpu"), max_crops=3)
        assert results[part]["crops"] == 3
        assert all(0.0 <= value <= 1.0 for value in results[part]["f1"].values())
    assert {key.split("/")[0] for key in results["border_free"]["f1"]} == {"0", "1", "2", "3", "bordered_3"}
    reading = border_reading({"a": results["border_free"], "b": copy.deepcopy(results["border_free"])}, 3)
    assert reading["verdict"] in {"border_dependent", "border_independent", "partial"}
    assert set(dates_reading(results["mechanism"])) == {"normal_f1", "same_date_f1", "ratio", "verdict"}

    def rows(bordered, free):
        return {"f1": {"bordered_3/official_unmodified/scoring": bordered, "3/official_unmodified/scoring": free}}

    assert border_reading({"x": rows(0.90, 0.60), "y": rows(0.80, 0.70)}, 3)["verdict"] == "border_dependent"
    assert border_reading({"x": rows(0.80, 0.79), "y": rows(0.70, 0.70)}, 3)["verdict"] == "border_independent"
    assert border_reading({"x": rows(0.80, 0.77), "y": rows(0.70, 0.67)}, 3)["verdict"] == "partial"

    def dates(normal, same):
        return dates_reading({"f1": {"normal/official_unmodified/scoring": normal,
                                     "same_date/official_unmodified/scoring": same}})["verdict"]

    assert dates(0.86, 0.80) == "ignores_dates"
    assert dates(0.86, 0.10) == "uses_dates"
    assert dates(0.86, 0.40) == "partial"
    assert dates(0.0, 0.0) == "undefined"
