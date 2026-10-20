"""DiffRegCD (WACV 2026) on the P27 affine protocol (configs/p27_diffregcd.yaml).

Runs in the DiffRegCD environment. For each dataset the released backbone and
that dataset's heads; A = the T1 crop, B = the perturbed T2 crop (the order
the network requires, main record 191.3), both in [-1, 1]; the change map is
the official head without feature warp (P13's author path), class 1 >= 0.5,
scored in the T1 frame on the P27 scoring pixels of every crop. The diffusion
noise is seeded per crop and shared by its levels. Two paths per level:

    diffregcd         the perturbed pair (the method as released)
    diffregcd_oracle  T2 put back with the true matrix (its detector's ceiling)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_crops
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    AffineRanges,
    affine_parameters,
    align,
    crop_origins,
    perturb,
    perturbation_matrix,
)
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_diffregcd import stable_inference_seed
from rs_cd.metrics.binary import BinaryChangeMetrics

PATHS = ("diffregcd", "diffregcd_oracle")


def all_crops(dataset: PairedChangeDataset, tile_size: int, size: int) -> list[dict[str, Any]]:
    """Every crop, in the order of tools.run_p27_f1.all_crops (that module is not importable here)."""

    total = len(dataset.rows) * len(crop_origins(tile_size, size))
    return select_crops(dataset.rows, tile_size, size, max(total, 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("DiffRegCD config is not marked preregistered")
    evaluation_path = Path(config["evaluation_config"])
    if sha256(evaluation_path) != config["evaluation_config_sha256"]:
        raise SystemExit(f"{evaluation_path} changed")
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
    known = {spec["name"] for spec in evaluation["datasets"]}
    if not set(config["datasets"]) <= known:
        raise SystemExit(f"unknown datasets {set(config['datasets']) - known}")
    return config, evaluation


def weight_paths(config: dict[str, Any], name: str) -> dict[str, Path]:
    root = Path(config["external"]["root"])
    entry = config["datasets"][name]
    return {
        "backbone": root / config["external"]["backbone"]["path"],
        "registration": root / entry["registration"]["path"],
        "change_detection": root / entry["change_detection"]["path"],
    }


def audit_weights(config: dict[str, Any], name: str) -> dict[str, Any]:
    entry = config["datasets"][name]
    expected = {"backbone": config["external"]["backbone"]["sha256"],
                "registration": entry["registration"]["sha256"],
                "change_detection": entry["change_detection"]["sha256"]}
    audit = {}
    for key, path in weight_paths(config, name).items():
        if not path.is_file():
            raise SystemExit(f"{name}: missing DiffRegCD weight {path}")
        actual = sha256(path)
        if actual != expected[key]:
            raise SystemExit(f"{name}: {path} has sha256 {actual}, expected {expected[key]}")
        audit[key] = {"path": str(path), "sha256": actual}
    return audit


def unit(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def change_logit(output: torch.Tensor) -> torch.Tensor:
    """Two-class logits -> one logit whose sigmoid is the class-1 softmax."""

    return output[:, 1:2].float() - output[:, 0:1].float()


def evaluate_dataset(spec: dict[str, Any], evaluation: dict[str, Any], infer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     *, seed: int, threshold: float, device: torch.device, max_crops: int | None = None) -> dict[str, Any]:
    perturbation = evaluation["perturbation"]
    size = int(perturbation["crop_size"])
    levels = [int(level) for level in perturbation["levels"]]
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels))
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    crops = all_crops(dataset, int(spec["tile_size"]), size)
    if max_crops is not None:
        crops = crops[:max_crops]
    cache = TileCache(dataset, int(spec["tile_size"]))
    totals = {str(level): {path: BinaryChangeMetrics(threshold=threshold) for path in PATHS}
              for level in [CLEAN_LEVEL, *levels]}
    start = time.perf_counter()
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        matrices = {level: perturbation_matrix(*affine_parameters(item["crop_id"], level, ranges, int(evaluation["seed"])),
                                               width=size, height=size)
                    for level in [CLEAN_LEVEL, *levels]}
        valid = scoring_mask([matrices[level] for level in levels], size)
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        image_a = (unit(t1)[None].to(device) * 2.0 - 1.0)
        crop_seed = stable_inference_seed(seed, item["crop_id"], 0)
        for level, matrix in matrices.items():
            perturbed = t2 if level == CLEAN_LEVEL else perturb(t2, matrix)
            # The clean oracle is the clean pair itself.
            views = {"diffregcd": perturbed} if level == CLEAN_LEVEL else {
                "diffregcd": perturbed, "diffregcd_oracle": align(perturbed, matrix)}
            for path, view in views.items():
                image_b = unit(view)[None].to(device) * 2.0 - 1.0
                torch.manual_seed(crop_seed)
                np.random.seed(crop_seed % (2**32))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(crop_seed)
                counts = confusion_counts(change_logit(infer(image_a, image_b)), target, keep, threshold)[0]
                targets = PATHS if level == CLEAN_LEVEL else (path,)
                for name in targets:
                    add_counts(totals[str(level)][name], counts)
        if index % 200 == 0 or index == len(crops):
            print(f"[p27 diffregcd {spec['name']} {index}/{len(crops)}]", flush=True)
    table = {level: {path: totals[level][path].compute() for path in PATHS} for level in totals}
    return {
        "crops": len(crops),
        "tiles": len(dataset.rows),
        "seconds": time.perf_counter() - start,
        "table": table,
        "f1": {level: {path: float(values["f1"]) for path, values in row.items()} for level, row in table.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_diffregcd.yaml")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, evaluation = load_config(Path(args.config))
    root = Path(config["external"]["root"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != config["external"]["commit"]:
        raise SystemExit(f"DiffRegCD commit {commit} is not {config['external']['commit']}")
    names = args.datasets or list(config["datasets"])
    audits = {name: audit_weights(config, name) for name in names}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from tools.evaluate_diffregcd_external import ExternalDiffRegCD

    device = torch.device("cuda")
    specs = {spec["name"]: {**spec, "split": config["split"]} for spec in evaluation["datasets"]}
    for name in names:
        path = output_dir / f"{name}.json"
        if path.is_file():
            continue
        adapter = ExternalDiffRegCD(root, config_path=root / config["datasets"][name]["config_json"],
                                    checkpoint_paths=weight_paths(config, name))
        result = evaluate_dataset(specs[name], evaluation, lambda a, b: adapter.infer(a, b)["official_unmodified"],
                                  seed=int(config["seed"]), threshold=float(config["probability_threshold"]),
                                  device=device)
        result.update(dataset=name, split=config["split"], weights=audits[name], external_commit=commit,
                      config_sha256=sha256(Path(args.config)))
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        f1 = result["f1"]
        print(f"[{name}] " + "  ".join(f"L{level}: {f1[level]['diffregcd']:.4f} (oracle {f1[level]['diffregcd_oracle']:.4f})"
                                       for level in f1), flush=True)
        del adapter
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
