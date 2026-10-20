"""P31 localizer ablation under the border-free protocol (docs/P31_BORDER_FREE_REMAINDER.md).

The DINOv3 detectors, the P30 test crops, draws and scoring pixels; perturbation and correction act on the whole
tile (rs_cd.p30_geometry). Only the localizer changes:

    v2_full, v2_coarse_only, v2_one_fine_step, v2_no_reliability   the P30 localizer and its inference-time variants
    v1_full                                                        the P27 localizer
    roma_similarity, sift_similarity                               matching baselines, P27 settings

Paths: raw, oracle, estimated (every pair corrected), estimated_gated. The raw path must reproduce the raw row of
the P30 test run. Finished variants are skipped, so the tool can be resumed. Reporting only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import DETECTOR_PATHS, add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import CLEAN_LEVEL, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.localizer_training import step1_rows
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p27_selector import matrix_of
from rs_cd.p30_geometry import BorderFreeGeometry
from rs_cd.p33_protocol import RETRAINED_LOCALIZERS
from tools import p27_localizer_ablation, run_p30
from tools.run_p27_selector import harness, open_dataset, with_split
from tools.run_p31_plugins import load

PATHS = ("raw", "oracle", "estimated", "estimated_gated")
P27_NAMES = {"v2_full": "p27_full", "v2_coarse_only": "p27_coarse_only", "v2_one_fine_step": "p27_one_fine_step",
             "v2_no_reliability": "p27_no_reliability", "v1_full": "p27_full"}


def build(variant: str, p30: dict[str, Any], evaluation: dict[str, Any], device: torch.device):
    if variant in P27_NAMES:
        checkpoint = run_p30.localizer_checkpoint(p30, "p27" if variant == "v1_full" else "p30")
        return p27_localizer_ablation.build(P27_NAMES[variant], checkpoint, evaluation, device)
    if variant in RETRAINED_LOCALIZERS:     # P33: registration modules retrained, full inference
        checkpoint = run_p30.pointed(RETRAINED_LOCALIZERS[variant]) / "checkpoints" / "best.pth"
        return p27_localizer_ablation.build("p27_full", checkpoint, evaluation, device)
    return p27_localizer_ablation.build(variant, Path("unused"), evaluation, device)


@torch.no_grad()
def evaluate(spec: dict[str, Any], evaluation: dict[str, Any], localizer, detector: dict[str, Any], device: torch.device,
             *, margin: int, max_crops: int | None = None) -> dict[str, Any]:
    h = harness(evaluation)
    size = h["size"]
    dataset = open_dataset(spec, spec["split"])
    crops = run_p30.margin_crops(dataset, spec, size, margin)
    crops = crops[:max_crops] if max_crops is not None else crops
    cache = TileCache(dataset, int(spec["tile_size"]))
    keys = [str(level) for level in (CLEAN_LEVEL, *h["levels"])]
    threshold = detector["threshold"]
    totals = {key: {path: BinaryChangeMetrics(threshold=threshold) for path in PATHS} for key in keys}
    success = {key: [] for key in keys[1:]}
    triggers = {key: [] for key in keys}
    seconds, missing = [], 0
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], size)
        truth_mask = mask > int(spec["mask_threshold"])
        rows = step1_rows(localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2,
                          change_fraction=float(truth_mask.mean()), ranges=h["ranges"], levels=h["levels"],
                          seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
                          trigger_probability=h["trigger_probability"], make_target=geometry.perturb)
        truths = [perturbation_matrix(float(r["angle"]), float(r["scale"]), float(r["tx"]), float(r["ty"]),
                                      width=size, height=size) for r in rows]
        estimates = [matrix_of(r) for r in rows]
        valid = scoring_mask([m for r, m in zip(rows, truths) if int(r["level"]) != CLEAN_LEVEL], size)
        images = []
        for truth, estimate in zip(truths, estimates):     # the order of rs_cd.p27_selector.crop_records
            images.extend(normalize(view) for view in geometry.path_images(None, truth, estimate).values())
            missing += int(geometry.missing_pixels(truth) > 0)
        t2_batch = torch.stack(images).to(device)
        t1_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
        logits = detector["model"](t1_batch, t2_batch)["change_logits"]
        target = torch.from_numpy(truth_mask.astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        counts = confusion_counts(logits, target, keep, threshold)
        step = len(DETECTOR_PATHS)
        for k, row in enumerate(rows):
            key = str(int(row["level"]))
            for j, path in enumerate(DETECTOR_PATHS):
                add_counts(totals[key][path], counts[k * step + j])
            chosen = "estimated" if row["trigger"] else "raw"
            add_counts(totals[key]["estimated_gated"], counts[k * step + DETECTOR_PATHS.index(chosen)])
            triggers[key].append(bool(row["trigger"]))
            if key in success:
                success[key].append(bool(row["success"]))
            seconds.append(float(row["seconds"]))
        if index % 500 == 0 or index == len(crops):
            print(f"[p31 ablation {spec['name']} {index}/{len(crops)}]", flush=True)
    f1 = {key: {path: float(totals[key][path].compute()["f1"]) for path in PATHS} for key in keys}
    return {"crops": len(crops), "f1": f1,
            "localization_success": {key: float(np.mean(values)) for key, values in success.items()},
            "trigger_rate": {key: float(np.mean(values)) for key, values in triggers.items()},
            "median_seconds": statistics.median(seconds),
            "estimated_view_with_uncovered_pixels": missing,
            "clean_drop": {path: f1["0"]["raw"] - f1["0"][path] for path in ("estimated", "estimated_gated")},
            "gain": {key: {path: f1[key][path] - f1[key]["raw"] for path in PATHS} for key in keys[1:]}}


def raw_anchor(result: dict[str, Any], anchor: dict[str, Any], tolerance: float) -> dict[str, Any]:
    """The raw path against the P30 test run's raw row, per dataset and level."""
    differences = {f"{name}/{level}": abs(entry["f1"][level]["raw"] - anchor["datasets"][name]["original"][level])
                   for name, entry in result["datasets"].items() for level in entry["f1"]}
    largest = max(differences.values())
    return {"max_abs_difference": largest, "tolerance": tolerance, "passed": largest <= tolerance}


def render(results: dict[str, dict[str, Any]]) -> str:
    lines = []
    names = sorted({name for result in results.values() for name in result["datasets"]})
    for name in names:
        lines.append(f"\n[{name}]  success L1/L2/L3 | always corrected: clean change, gain L1/L2/L3 | gated: clean change, gain L3")
        for variant, result in results.items():
            entry = result["datasets"][name]
            success = "/".join(f"{entry['localization_success'][k] * 100:4.1f}" for k in ("1", "2", "3"))
            gains = "/".join(f"{entry['gain'][k]['estimated'] * 100:+5.1f}" for k in ("1", "2", "3"))
            lines.append(f"  {variant:<18} {success} | {-entry['clean_drop']['estimated'] * 100:+5.1f}, {gains} | "
                         f"{-entry['clean_drop']['estimated_gated'] * 100:+5.1f}, {entry['gain']['3']['estimated_gated'] * 100:+5.1f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p31_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, help="resumable: finished variants are kept")
    parser.add_argument("--variants", nargs="+", default=None, help="default: the variants of the config")
    args = parser.parse_args()
    config, p30, evaluation, _ = load(Path(args.config))
    spec = dict(config["ablation"])
    if args.variants:
        spec["variants"] = list(args.variants)
    evaluation = with_split(evaluation, spec["split"])
    anchor_root = run_p30.pointed(spec["raw_anchor"])
    anchor = json.loads((anchor_root / "p30_test.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detectors = {s["name"]: run_p30.open_detector(p30, "original", s["name"], device) for s in evaluation["datasets"]}
    results = {}
    for variant in spec["variants"]:
        path = output_dir / f"{variant}.json"
        if path.is_file():
            results[variant] = json.loads(path.read_text(encoding="utf-8"))
            continue
        localizer = build(variant, p30, evaluation, device)
        result: dict[str, Any] = {"variant": variant, "split": spec["split"], "datasets": {}}
        for dataset_spec in evaluation["datasets"]:
            name = dataset_spec["name"]
            result["datasets"][name] = {**evaluate(dataset_spec, evaluation, localizer, detectors[name], device,
                                                   margin=int(p30["crop_margin"])),
                                        "detector_sha256": detectors[name]["sha256"]}
            print(f"[{variant}] {name} done", flush=True)
        result["raw_anchor"] = raw_anchor(result, anchor, float(spec["raw_anchor_tolerance"]))
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[variant] = result
        if not result["raw_anchor"]["passed"]:
            raise SystemExit(f"{variant}: the raw path does not reproduce the P30 test run "
                             f"(max difference {result['raw_anchor']['max_abs_difference']:.5f}); instrument error, stop")
        del localizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    text = render(results)
    # --variants adds variants to an existing run; keep that run's summary of the configured variants intact
    summary_name = "localizer_ablation.txt" if not args.variants else "localizer_ablation_extra.txt"
    (output_dir / summary_name).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
