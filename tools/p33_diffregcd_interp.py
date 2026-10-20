"""DiffRegCD at level 0 with the interpolation control (P33, docs/P33_REVIEWER_EXPERIMENTS.md).

Runs in the DiffRegCD environment, like tools.p27_diffregcd_rerun (same config, weights, wiring, seeds, border-free
margin crops and scoring pixels). DiffRegCD's input convention: A = the earlier image, perturbed; B = the later image.
Views per crop:

    0           A = T1 as is (the re-run's level 0)
    0_interp    A = T1 resampled on its whole tile by the per-crop sub-pixel transform of rs_cd.p33_protocol
                (<= 0.5 px, +-0.5 % scale): interpolated like levels 1-3, without their misalignment
    3           A = T1 perturbed at level 3 on its whole tile (the re-run's level 3)

Levels 0 and 3 must reproduce the border-free part of the frozen re-run (anchor). If DiffRegCD's low level-0 F1 on
DSIFN-CD comes from never having seen an un-resampled earlier image, 0_interp recovers it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rs_cd.affine_crops import TileCache, select_margin_crops
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import affine_parameters, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_diffregcd import stable_inference_seed
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p30_geometry import BorderFreeGeometry
from rs_cd.p33_protocol import INTERP_LEVEL, INTERP_RANGES
from tools.p27_diffregcd_rerun import PRIMARY, WIRINGS, harness, load, matrices_for
from tools.run_p27_diffregcd_f1 import audit_weights, change_logit, unit, weight_paths

VIEWS = ("0", "0_interp", "3")
RERUN_POINTER = Path("reports/p27_localizer/diffregcd/rerun/LATEST")


def run_dataset(spec: dict[str, Any], evaluation: dict[str, Any], infer, *, margin: int, seed: int, threshold: float,
                device: torch.device, max_crops: int | None) -> dict[str, Any]:
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    size, levels, ranges, draw_seed = harness(evaluation)
    tile = int(spec["tile_size"])
    cache = TileCache(dataset, tile)
    crops = select_margin_crops(dataset.rows, tile, size, margin)
    crops = crops[:max_crops] if max_crops is not None else crops
    totals: dict[str, BinaryChangeMetrics] = {}
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        matrices = matrices_for(item["crop_id"], levels, ranges, draw_seed, size)
        valid = scoring_mask([matrices[level] for level in levels], size)
        geometry = BorderFreeGeometry(cache.arrays[0], item["top"], item["left"], size)
        tiny = perturbation_matrix(*affine_parameters(item["crop_id"], INTERP_LEVEL, INTERP_RANGES, draw_seed),
                                   width=size, height=size)
        views = {"0": t1, "0_interp": geometry.perturb(tiny), "3": geometry.perturb(matrices[levels[-1]])}
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        crop_seed = stable_inference_seed(seed, item["crop_id"], 0)     # as the re-run: shared by the crop's views
        for key, image_a in views.items():
            torch.manual_seed(crop_seed)
            np.random.seed(crop_seed % (2**32))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(crop_seed)
            outputs = infer(unit(image_a)[None].to(device) * 2.0 - 1.0, unit(t2)[None].to(device) * 2.0 - 1.0)
            for wiring in WIRINGS:
                metrics = totals.setdefault(f"{key}/{wiring}", BinaryChangeMetrics(threshold=threshold))
                add_counts(metrics, confusion_counts(change_logit(outputs[wiring]), target, keep, threshold)[0])
        if index % 50 == 0 or index == len(crops):
            print(f"[diffregcd interp {spec['name']} {index}/{len(crops)}]", flush=True)
    computed = {key: metrics.compute() for key, metrics in totals.items()}
    return {"crops": len(crops),
            "f1": {key: float(values["f1"]) for key, values in computed.items()},
            "precision": {key: float(values["precision"]) for key, values in computed.items()},
            "recall": {key: float(values["recall"]) for key, values in computed.items()}}


def anchor(name: str, result: dict[str, Any]) -> dict[str, Any]:
    rerun = Path(RERUN_POINTER.read_text(encoding="utf-8").strip())
    frozen = json.loads((rerun / "border_free" / f"{name}.json").read_text(encoding="utf-8"))["f1"]
    differences = {key: abs(result["f1"][f"{key}/{PRIMARY}"] - float(frozen[f"{key}/{PRIMARY}/scoring"]))
                   for key in ("0", "3")}
    return {"rerun": str(rerun), "differences": differences, "passed": max(differences.values()) <= 1e-3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/p27_diffregcd_rerun.yaml")
    parser.add_argument("--output-dir", default="reports/p33/diffregcd_interp")
    parser.add_argument("--max-crops", type=int, default=None, help="smoke runs only")
    args = parser.parse_args()
    rerun, config, evaluation = load(Path(args.config))
    root = Path(config["external"]["root"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != config["external"]["commit"]:
        raise SystemExit(f"DiffRegCD commit {commit} is not {config['external']['commit']}")

    from tools.evaluate_diffregcd_external import ExternalDiffRegCD

    device = torch.device("cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = {spec["name"]: {**spec, "split": config["split"]} for spec in evaluation["datasets"]}
    summary = {}
    for name in config["datasets"]:
        path = output / f"{name}.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            adapter = ExternalDiffRegCD(root, config_path=root / config["datasets"][name]["config_json"],
                                        checkpoint_paths=weight_paths(config, name))
            result = run_dataset(specs[name], evaluation, lambda a, b: adapter.infer(a, b),
                                 margin=int(rerun["crop_margin"]), seed=int(config["seed"]),
                                 threshold=float(config["probability_threshold"]), device=device,
                                 max_crops=args.max_crops)
            result.update(dataset=name, weights=audit_weights(config, name), external_commit=commit)
            if args.max_crops is None:
                result["anchor"] = anchor(name, result)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            del adapter
            torch.cuda.empty_cache()
        summary[name] = {key: result["f1"][f"{key}/{PRIMARY}"] for key in VIEWS}
        summary[name]["anchor_passed"] = result.get("anchor", {}).get("passed")
        print(f"[diffregcd interp {name}] F1 " + ", ".join(f"{key} {summary[name][key] * 100:.1f}" for key in VIEWS)
              + f"; anchor {summary[name]['anchor_passed']}", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
