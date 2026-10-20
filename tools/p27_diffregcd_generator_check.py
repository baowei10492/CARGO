"""DiffRegCD on its own perturbation generator (docs/P27_DIFFREGCD_GENERATOR_CHECK.md).

Runs in the DiffRegCD environment, with the weights, config and diffusion seeds of the
P27 DiffRegCD run. Every view is made with the authors' generator
(scripts/prepare_gt_flow.py: the same draws in the same order, the same matrix and the
same cv2.warpAffine call with its constant zero border); the label is not warped, as in
their generator. Both input orders and both author wirings are scored, on all pixels
(their protocol) and on the pixels where the warped image has content. Diagnosis only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from rs_cd.affine_crops import TileCache, select_crops
from rs_cd.affine_f1 import add_counts, confusion_counts
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_diffregcd import stable_inference_seed
from rs_cd.metrics.binary import BinaryChangeMetrics
from tools.run_p27_diffregcd_f1 import audit_weights, change_logit, load_config, unit, weight_paths

RANGES = {                      # paper Table 1; the P27 level-3 ranges; the script defaults
    "paper": {"angle": 30.0, "scale": (0.80, 1.25), "trans": 25.0},
    "paper_norot": {"angle": 0.0, "scale": (0.80, 1.25), "trans": 25.0},
    "repo_default": {"angle": 20.0, "scale": (0.90, 1.15), "trans": 16.0},
}
ORDERS = {"b_warped": ("paper", "paper_norot", "repo_default"),   # A = T1, B = warp(T2): training code, P27
          "a_warped": ("paper", "paper_norot")}                   # A = warp(T1), B = T2: README
WIRINGS = ("official_unmodified", "intended_warped")
SCORINGS = ("all", "content")
CROPS = {"LEVIR-CD": 600, "DSIFN-CD": None, "WHU-CD": 600}
VERDICT_DATASET = "LEVIR-CD"


def generator_matrix(rng: np.random.RandomState, size: int, *, angle: float, scale: tuple[float, float],
                     trans: float) -> np.ndarray:
    """scripts/prepare_gt_flow.py, lines 65-71: angle, scale, tx, ty drawn in this order."""
    drawn_angle = rng.uniform(-angle, angle)
    drawn_scale = rng.uniform(*scale)
    tx = rng.uniform(-trans, trans)
    ty = rng.uniform(-trans, trans)
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), drawn_angle, drawn_scale)
    matrix[:, 2] += [tx, ty]
    return matrix


def generator_warp(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """scripts/prepare_gt_flow.py, line 73: OpenCV's default constant zero border."""
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def content(matrix: np.ndarray, size: int) -> np.ndarray:
    """Pixels of the warped image that carry content."""
    ones = np.ones((size, size), np.uint8)
    return cv2.warpAffine(ones, matrix, (size, size), flags=cv2.INTER_NEAREST) > 0


def draw_seed(seed: int, crop_id: str, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{crop_id}|{name}".encode("utf-8")).digest()[:4], "little")


def views(t1: np.ndarray, t2: np.ndarray, crop_id: str, seed: int) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(key, image A, image B, content pixels): the clean pair, then every order and range set.

    Both orders use the same matrix for a range set, so they differ only in which image is warped.
    """
    size = t1.shape[0]
    out = [("clean", t1, t2, np.ones((size, size), bool))]
    matrices = {name: generator_matrix(np.random.RandomState(draw_seed(seed, crop_id, name)), size, **spec)
                for name, spec in RANGES.items()}
    for order, names in ORDERS.items():
        for name in names:
            matrix = matrices[name]
            if order == "b_warped":
                out.append((f"{order}/{name}", t1, generator_warp(t2, matrix), content(matrix, size)))
            else:
                out.append((f"{order}/{name}", generator_warp(t1, matrix), t2, content(matrix, size)))
    return out


def keys() -> list[str]:
    return ["clean", *(f"{order}/{name}" for order, names in ORDERS.items() for name in names)]


def check(spec: dict[str, Any], infer: Callable[[torch.Tensor, torch.Tensor], dict], *, size: int, seed: int,
          threshold: float, device: torch.device, cap: int | None) -> dict[str, Any]:
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    tile = int(spec["tile_size"])
    everything = len(dataset.rows) * (tile // size) ** 2
    crops = select_crops(dataset.rows, tile, size, cap if cap is not None else max(everything, 1))
    cache = TileCache(dataset, tile)
    totals = {f"{key}/{wiring}/{scoring}": BinaryChangeMetrics(threshold=threshold)
              for key in keys() for wiring in WIRINGS for scoring in SCORINGS}
    everywhere = torch.ones(1, 1, size, size, device=device)
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        crop_seed = stable_inference_seed(seed, item["crop_id"], 0)   # as the P27 run: shared by the crop's views
        for key, image_a, image_b, inside in views(t1, t2, item["crop_id"], seed):
            torch.manual_seed(crop_seed)
            np.random.seed(crop_seed % (2**32))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(crop_seed)
            outputs = infer(unit(image_a)[None].to(device) * 2.0 - 1.0, unit(image_b)[None].to(device) * 2.0 - 1.0)
            keep = {"all": everywhere, "content": torch.from_numpy(inside.astype(np.float32))[None, None].to(device)}
            for wiring in WIRINGS:
                logit = change_logit(outputs[wiring])
                for scoring in SCORINGS:
                    add_counts(totals[f"{key}/{wiring}/{scoring}"],
                               confusion_counts(logit, target, keep[scoring], threshold)[0])
        if index % 100 == 0 or index == len(crops):
            print(f"[diffregcd generator check {spec['name']} {index}/{len(crops)}]", flush=True)
    computed = {key: metrics.compute() for key, metrics in totals.items()}
    return {"crops": len(crops),
            "f1": {key: float(values["f1"]) for key, values in computed.items()},
            "precision": {key: float(values["precision"]) for key, values in computed.items()},
            "recall": {key: float(values["recall"]) for key, values in computed.items()}}


def reading(name: str, f1: dict[str, float]) -> dict[str, Any]:
    clean = f1["clean/official_unmodified/all"]
    drops = {order: clean - f1[f"{order}/paper/official_unmodified/all"] for order in ORDERS}
    if name != VERDICT_DATASET:
        verdict = "reported"
    elif clean < 0.85:
        verdict = "sanity_failed"
    elif any(drop <= 0.05 for drop in drops.values()):
        verdict = "released_robust"
    elif all(drop >= 0.20 for drop in drops.values()):
        verdict = "released_not_robust"
    else:
        verdict = "intermediate"
    return {"clean_f1": clean, "drop_paper_official_all": drops, "verdict": verdict}


def render(name: str, result: dict[str, Any]) -> str:
    lines = [f"[{name}] {result['crops']} crops; F1 % (recall %), official test path | training-time warp; "
             "all pixels / content pixels"]
    for key in keys():
        cells = []
        for wiring in WIRINGS:
            for scoring in SCORINGS:
                full = f"{key}/{wiring}/{scoring}"
                cells.append(f"{result['f1'][full] * 100:5.1f} ({result['recall'][full] * 100:4.1f})")
        lines.append(f"  {key:<24} " + "  ".join(cells[:2]) + "  |  " + "  ".join(cells[2:]))
    r = result["reading"]
    lines.append(f"  drop clean -> paper (official, all pixels): "
                 + ", ".join(f"{order} {drop * 100:.1f}" for order, drop in r["drop_paper_official_all"].items())
                 + f"; verdict {r['verdict']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_diffregcd.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, evaluation = load_config(Path(args.config))
    root = Path(config["external"]["root"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != config["external"]["commit"]:
        raise SystemExit(f"DiffRegCD commit {commit} is not {config['external']['commit']}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    from tools.evaluate_diffregcd_external import ExternalDiffRegCD

    device = torch.device("cuda")
    size = int(evaluation["perturbation"]["crop_size"])
    specs = {spec["name"]: {**spec, "split": config["split"]} for spec in evaluation["datasets"]}
    summary = {}
    for name, cap in CROPS.items():
        audit = audit_weights(config, name)
        adapter = ExternalDiffRegCD(root, config_path=root / config["datasets"][name]["config_json"],
                                    checkpoint_paths=weight_paths(config, name))
        result = check(specs[name], lambda a, b: adapter.infer(a, b), size=size, seed=int(config["seed"]),
                       threshold=float(config["probability_threshold"]), device=device, cap=cap)
        result.update(dataset=name, weights=audit, external_commit=commit, reading=reading(name, result["f1"]))
        (output_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary[name] = result["reading"]
        print(render(name, result), flush=True)
        if result["reading"]["verdict"] == "sanity_failed":
            raise SystemExit("clean LEVIR-CD F1 below 0.85 on these crops: the adapter is broken, stop")
        del adapter
        torch.cuda.empty_cache()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
