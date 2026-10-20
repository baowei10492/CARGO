"""Why DiffRegCD's oracle path falls far below its clean score (P27 test crops).

Runs in the DiffRegCD environment, on the first --levir-crops LEVIR-CD test
crops and all DSIFN-CD test crops, at levels 1 and 3. For each crop and level
the second image is shown to DiffRegCD in several forms, all scored on the
P27 scoring pixels:

    clean          T2 itself
    border_only    T2 with the pixels the level's matrix pushes outside set to zero
    resample_only  T2 perturbed and put back (bilinear twice), the lost border
                   refilled from T2 so that only the resampling differs
    oracle         T2 perturbed and put back, zero border (the main run's oracle)
    raw            the perturbed T2 (the main run's path)

Both author wirings are recorded for every view: the released test path
(official, no feature warp) and the training-time path (features of B warped
with the predicted flow). Diagnosis only; nothing is a verdict.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import AffineRanges, affine_parameters, align, perturb, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_diffregcd import stable_inference_seed
from rs_cd.metrics.binary import BinaryChangeMetrics
from tools.run_p27_diffregcd_f1 import all_crops, audit_weights, change_logit, load_config, unit, weight_paths

VIEWS = ("clean", "border_only", "resample_only", "oracle", "raw")
WIRINGS = ("official_unmodified", "intended_warped")
LEVELS = (1, 3)


def views(t2: np.ndarray, matrix: np.ndarray) -> dict[str, np.ndarray]:
    size = t2.shape[0]
    inside = scoring_mask([matrix], size)[..., None]
    perturbed = perturb(t2, matrix)
    restored = align(perturbed, matrix)
    return {
        "clean": t2,
        "border_only": np.where(inside, t2, 0).astype(np.uint8),
        "resample_only": np.where(inside, restored, t2).astype(np.uint8),
        "oracle": restored,
        "raw": perturbed,
    }


def diagnose(spec: dict[str, Any], evaluation: dict[str, Any], infer: Callable[[torch.Tensor, torch.Tensor], dict],
             *, seed: int, threshold: float, device: torch.device, max_crops: int | None) -> dict[str, Any]:
    perturbation = evaluation["perturbation"]
    size = int(perturbation["crop_size"])
    levels = [int(level) for level in perturbation["levels"]]
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels))
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    crops = all_crops(dataset, int(spec["tile_size"]), size)[:max_crops]
    cache = TileCache(dataset, int(spec["tile_size"]))
    totals = {f"{level}/{view}/{wiring}": BinaryChangeMetrics(threshold=threshold)
              for level in LEVELS for view in VIEWS for wiring in WIRINGS}
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        matrices = {level: perturbation_matrix(*affine_parameters(item["crop_id"], level, ranges, int(evaluation["seed"])),
                                               width=size, height=size) for level in levels}
        valid = scoring_mask(list(matrices.values()), size)
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        image_a = unit(t1)[None].to(device) * 2.0 - 1.0
        crop_seed = stable_inference_seed(seed, item["crop_id"], 0)
        for level in LEVELS:
            for view, image in views(t2, matrices[level]).items():
                torch.manual_seed(crop_seed)
                np.random.seed(crop_seed % (2**32))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(crop_seed)
                outputs = infer(image_a, unit(image)[None].to(device) * 2.0 - 1.0)
                for wiring in WIRINGS:
                    counts = confusion_counts(change_logit(outputs[wiring]), target, keep, threshold)[0]
                    add_counts(totals[f"{level}/{view}/{wiring}"], counts)
        if index % 100 == 0 or index == len(crops):
            print(f"[p27 diffregcd diagnosis {spec['name']} {index}/{len(crops)}]", flush=True)
    f1 = {key: float(metrics.compute()["f1"]) for key, metrics in totals.items()}
    return {"crops": len(crops), "f1": f1,
            "precision": {key: float(m.compute()["precision"]) for key, m in totals.items()},
            "recall": {key: float(m.compute()["recall"]) for key, m in totals.items()}}


def render(name: str, result: dict[str, Any]) -> str:
    lines = [f"[{name}] {result['crops']} crops  (F1 / precision / recall)"]
    for wiring in WIRINGS:
        for level in LEVELS:
            cells = []
            for view in VIEWS:
                key = f"{level}/{view}/{wiring}"
                cells.append(f"{view} {result['f1'][key]:.3f}/{result['precision'][key]:.2f}/{result['recall'][key]:.2f}")
            lines.append(f"  {wiring:<20} L{level}  " + "  ".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_diffregcd.yaml")
    parser.add_argument("--levir-crops", type=int, default=300)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, evaluation = load_config(Path(args.config))
    root = Path(config["external"]["root"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != config["external"]["commit"]:
        raise SystemExit(f"DiffRegCD commit {commit} is not {config['external']['commit']}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from tools.evaluate_diffregcd_external import ExternalDiffRegCD

    device = torch.device("cuda")
    specs = {spec["name"]: {**spec, "split": config["split"]} for spec in evaluation["datasets"]}
    for name, limit in (("LEVIR-CD", args.levir_crops), ("DSIFN-CD", None)):
        audit = audit_weights(config, name)
        adapter = ExternalDiffRegCD(root, config_path=root / config["datasets"][name]["config_json"],
                                    checkpoint_paths=weight_paths(config, name))
        result = diagnose(specs[name], evaluation, lambda a, b: adapter.infer(a, b), seed=int(config["seed"]),
                          threshold=float(config["probability_threshold"]), device=device, max_crops=limit)
        result.update(dataset=name, weights=audit, external_commit=commit)
        (output_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(render(name, result), flush=True)
        del adapter
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
