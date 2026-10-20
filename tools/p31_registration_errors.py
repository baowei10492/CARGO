"""Per-crop registration errors on the border-free test crops (main record 192.71).

The localizer variants and matching baselines of the P31 localizer ablation (tools/p31_localizer_ablation.py), run
again without the detector so that every crop's transfer errors are kept: the P30 test crops and draws, perturbation
of the whole T2 tile (rs_cd.p30_geometry). The success rates must reproduce the ablation run. The rows feed the
success-by-change-fraction bins and the success-versus-tolerance curve (analysed locally). Resumable per variant.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.localizer_training import step1_rows
from rs_cd.p30_geometry import BorderFreeGeometry
from tools import run_p30
from tools.p31_localizer_ablation import build
from tools.run_p27_selector import harness, open_dataset, with_split
from tools.run_p31_plugins import load

VARIANTS = ("v2_full", "v2_coarse_only", "v2_one_fine_step", "v2_no_reliability", "sift_similarity", "roma_similarity")
FIELDS = ("dataset", "sample_id", "crop_id", "level", "change_fraction", "perturbation_px", "ate_raw", "ate_diff",
          "success", "converged", "trigger", "misaligned_probability", "estimate_displacement_px")
# Success is thresholded at 2 px, so float differences between runs flip crops that sit on the threshold (the first
# overnight run, v2_full: 1 of 2048 LEVIR, 1 of 192 DSIFN, up to 3 of 2552 WHU crops). The check is for instrument
# errors (other crops, draws or weights), which move far more crops: per dataset and level it allows
# max(minimum, fraction x crops) flipped crops; RoMa also samples its matches.
FLIPPED = {"roma_similarity": (10, 0.02), "roma_ts_ransac": (10, 0.02), "roma_ts_huber": (10, 0.02)}
DEFAULT_FLIPPED = (3, 0.005)


def allowed_flips(variant: str, crops: int) -> int:
    minimum, fraction = FLIPPED.get(variant, DEFAULT_FLIPPED)
    return max(minimum, math.ceil(fraction * crops))


@torch.no_grad()
def crop_errors(spec: dict[str, Any], evaluation: dict[str, Any], localizer, *, margin: int,
                max_crops: int | None = None) -> list[dict[str, Any]]:
    h = harness(evaluation)
    size = h["size"]
    dataset = open_dataset(spec, spec["split"])
    crops = run_p30.margin_crops(dataset, spec, size, margin)
    crops = crops[:max_crops] if max_crops is not None else crops
    cache = TileCache(dataset, int(spec["tile_size"]))
    out = []
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], size)
        truth_mask = mask > int(spec["mask_threshold"])
        rows = step1_rows(localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2,
                          change_fraction=float(truth_mask.mean()), ranges=h["ranges"], levels=h["levels"],
                          seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
                          trigger_probability=h["trigger_probability"], make_target=geometry.perturb)
        for row in rows:
            out.append({"sample_id": item.get("sample_id", ""), **{k: row[k] for k in FIELDS if k in row}})
        if index % 500 == 0 or index == len(crops):
            print(f"[registration errors {spec['name']} {index}/{len(crops)}]", flush=True)
    return out


def success_by_level(rows: list[dict[str, Any]]) -> dict[str, float]:
    levels = sorted({int(r["level"]) for r in rows if int(r["level"]) != 0})
    return {str(level): float(np.mean([bool(r["success"]) for r in rows if int(r["level"]) == level])) for level in levels}


def anchor(variant: str, success: dict[str, dict[str, float]], reference: dict[str, Any]) -> dict[str, Any]:
    """Success rates ({dataset: {level: rate}}) against the localizer ablation run of the same variant."""
    differences, flipped, over = {}, {}, []
    for name, levels in success.items():
        entry = reference["datasets"][name]
        crops = int(entry["crops"])
        for level, value in levels.items():
            key = f"{name}/{level}"
            differences[key] = abs(value - float(entry["localization_success"][level]))
            flipped[key] = int(round(differences[key] * crops))
            if flipped[key] > allowed_flips(variant, crops):
                over.append(key)
    return {"max_abs_difference": max(differences.values()), "max_flipped_crops": max(flipped.values()),
            "passed": not over, "over_limit": over, "differences": differences, "flipped_crops": flipped,
            "rule": {"minimum_and_fraction": list(FLIPPED.get(variant, DEFAULT_FLIPPED))}}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *FIELDS], restval="")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p31_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output-dir", required=True, help="resumable: finished variants are kept")
    args = parser.parse_args()
    config, p30, evaluation, _ = load(Path(args.config))
    spec = config["ablation"]
    evaluation = with_split(evaluation, spec["split"])
    ablation_root = run_p30.pointed("reports/p31/localizer_ablation/LATEST")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in args.variants:
        done = output_dir / f"{variant}.json"
        reference = json.loads((ablation_root / f"{variant}.json").read_text(encoding="utf-8"))
        if done.is_file():
            result = json.loads(done.read_text(encoding="utf-8"))
            result["anchor"] = anchor(variant, result["success"], reference)      # re-checked with the current rule
            done.write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary[variant] = result
            print(f"[{variant}] kept; anchor {result['anchor']['max_flipped_crops']} flipped crops "
                  f"({'passed' if result['anchor']['passed'] else 'FAILED'})", flush=True)
            continue
        torch.manual_seed(int(evaluation["seed"]))
        localizer = build(variant, p30, evaluation, device)
        rows_by_dataset = {}
        for dataset_spec in evaluation["datasets"]:
            rows_by_dataset[dataset_spec["name"]] = crop_errors(dataset_spec, evaluation, localizer,
                                                                margin=int(p30["crop_margin"]))
        write_rows(output_dir / f"errors_{variant}.csv", [r for rows in rows_by_dataset.values() for r in rows])
        success = {name: success_by_level(rows) for name, rows in rows_by_dataset.items()}
        check = anchor(variant, success, reference)
        result = {"variant": variant, "ablation_run": str(ablation_root), "anchor": check, "success": success}
        done.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary[variant] = result
        print(f"[{variant}] anchor {check['max_flipped_crops']} flipped crops, max difference "
              f"{check['max_abs_difference']:.4f} ({'passed' if check['passed'] else 'FAILED'})", flush=True)
        del localizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({v: s["anchor"]["flipped_crops"] for v, s in summary.items()}, indent=1))
    failed = [v for v, s in summary.items() if not s["anchor"]["passed"]]
    if failed:      # every variant is still run and kept; an instrument error shows as many flipped crops
        raise SystemExit(f"success rates differ from the localizer ablation run beyond threshold flips: {failed}")


if __name__ == "__main__":
    main()
