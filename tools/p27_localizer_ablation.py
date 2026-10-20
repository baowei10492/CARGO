"""P27 localizer ablation and matching baselines (docs/P27_EXPERIMENTS_AMENDMENT.md).

The frozen DINOv3 detectors of the F1 step, the same crops, perturbations and
scoring pixels; only the localizer changes. For every variant and dataset the
tool reports the localization success per level and the detector F1 for the
raw / oracle / always-corrected / localizer-gated paths. Variants that are
already finished (their JSON exists) are skipped, so the tool can be resumed.

    p27_full                  the frozen P27 localizer
    p27_coarse_only           no fine refinement
    p27_one_fine_step         one fine iteration instead of two
    p27_no_reliability        correspondences keep their confidence weight only
    roma_similarity           RoMa (CVPR 2024) + RANSAC similarity, P25 settings
    sift_similarity           OpenCV SIFT + ratio test + the same RANSAC fit

The gate of the inference-time P27 variants still comes from the misalignment
head, whose inputs those variants change; the always-corrected path isolates
localization quality.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from rs_cd.localizer_training import load_localizer
from rs_cd.p27_baseline_localizers import SiftSimilarity, SimilarityBaseline, p27_variant, roma_estimator
from tools.run_p27_f1 import evaluate_dataset, open_detectors
from tools.run_p27_f1 import load_config as load_f1_config
from tools.run_p27_selector import with_split

VARIANTS = ("p27_full", "p27_coarse_only", "p27_one_fine_step", "p27_no_reliability",
            "roma_similarity", "sift_similarity")
PATHS = ("raw", "oracle", "estimated", "estimated_gated")
# Reviewer check (2026-10-08): matching baselines fitted in the registration module's family (translation + isotropic
# scale): RoMa by RANSAC at 3 px or by certainty-weighted least squares with Huber reweighting, SIFT by RANSAC at 3 px
# (rs_cd.translation_scale_baselines).
ROMA_TRANSLATION_SCALE = {
    "roma_ts_ransac": {"robust": "ransac", "reprojection_px": 3.0},
    "roma_ts_huber": {"robust": "huber", "reprojection_px": 2.0},
}
SIFT_TRANSLATION_SCALE_PX = 3.0
_ROMA_SAMPLERS: dict[tuple, Any] = {}


def shared_roma_sampler(evaluation: dict[str, Any], device: torch.device):
    """One RoMa model and match cache per process, so the two translation-scale fits share every RoMa pass."""
    from rs_cd.translation_scale_baselines import RomaMatchSampler
    from tools.run_p25_affine_step1 import build_localizer as build_roma

    key = (str(device), int(evaluation["seed"]), int(evaluation["localizer"]["num_matches"]))
    if key not in _ROMA_SAMPLERS:
        _ROMA_SAMPLERS[key] = RomaMatchSampler(build_roma(evaluation, device))
    return _ROMA_SAMPLERS[key]


def build(variant: str, checkpoint: Path, evaluation: dict[str, Any], device: torch.device):
    if variant.startswith("p27_"):
        model = load_localizer(str(checkpoint), device)
        options = {
            "p27_full": {},
            "p27_coarse_only": {"fine_iterations": 0},
            "p27_one_fine_step": {"fine_iterations": 1},
            "p27_no_reliability": {"reliability_weighting": False},
        }[variant]
        return p27_variant(model, device, **options)
    spec = evaluation["localizer"]
    trigger = float(spec["trigger_mean_displacement_px"])
    size = int(evaluation["perturbation"]["crop_size"])
    if variant == "roma_similarity":
        from tools.run_p25_affine_step1 import build_localizer as build_roma

        return SimilarityBaseline(roma_estimator(build_roma(evaluation, device)), trigger_px=trigger, size=size)
    if variant in ROMA_TRANSLATION_SCALE:
        from rs_cd.translation_scale_baselines import RomaTranslationScale

        estimator = RomaTranslationScale(
            shared_roma_sampler(evaluation, device), max_iters=int(spec["ransac_max_iters"]),
            confidence=float(spec["ransac_confidence"]), refine_iters=int(spec["refine_iters"]),
            min_inliers=int(spec["min_inliers"]), **ROMA_TRANSLATION_SCALE[variant],
        )
        return SimilarityBaseline(estimator, trigger_px=trigger, size=size)
    if variant == "sift_ts_ransac":
        from rs_cd.translation_scale_baselines import SiftTranslationScale

        sift = SiftTranslationScale(
            seed=int(evaluation["seed"]), reprojection_px=SIFT_TRANSLATION_SCALE_PX,
            max_iters=int(spec["ransac_max_iters"]), confidence=float(spec["ransac_confidence"]),
            refine_iters=int(spec["refine_iters"]), min_inliers=int(spec["min_inliers"]),
        )
        return SimilarityBaseline(sift, trigger_px=trigger, size=size)
    if variant in ("phasecorr_fm", "ecc_ts"):
        # P33 reviewer request: classical global registration in the module's family (translation + isotropic scale)
        from rs_cd.global_registration_baselines import EccScaleTranslation, FourierMellinScaleTranslation

        estimator = FourierMellinScaleTranslation(size=size) if variant == "phasecorr_fm" else EccScaleTranslation(size=size)
        return SimilarityBaseline(estimator, trigger_px=trigger, size=size)
    if variant == "sift_similarity":
        sift = SiftSimilarity(
            seed=int(evaluation["seed"]), reprojection_px=float(spec["ransac_reprojection_px"]),
            max_iters=int(spec["ransac_max_iters"]), confidence=float(spec["ransac_confidence"]),
            refine_iters=int(spec["refine_iters"]), min_inliers=int(spec["min_inliers"]),
        )
        return SimilarityBaseline(sift, trigger_px=trigger, size=size)
    raise SystemExit(f"unknown variant {variant}")


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    levels = result["localizer"]["levels"]
    f1 = result["f1"]
    return {
        "crops": result["crops"],
        "localization_success": {key: value.get("success_rate") for key, value in levels.items()},
        "median_error_diff": {key: value.get("median_ate_diff") for key, value in levels.items()},
        "trigger_rate": {key: value["trigger_rate"] for key, value in levels.items()},
        "median_seconds": levels["0"]["median_seconds"],
        "f1": f1,
        "clean_drop": {path: f1["0"]["raw"] - f1["0"][path] for path in ("estimated", "estimated_gated")},
        "gain": {level: {path: f1[level][path] - f1[level]["raw"] for path in PATHS}
                 for level in f1 if level != "0"},
    }


def run_variant(variant: str, checkpoint: Path, f1_config: dict[str, Any], evaluation: dict[str, Any],
                detectors: dict[str, dict[str, Any]], device: torch.device, *,
                max_crops: int | None = None, localizer=None) -> dict[str, Any]:
    localizer = localizer if localizer is not None else build(variant, checkpoint, evaluation, device)
    out: dict[str, Any] = {"variant": variant, "split": evaluation["datasets"][0]["split"], "datasets": {}}
    for spec in evaluation["datasets"]:
        detector = detectors[spec["name"]]
        start = time.perf_counter()
        result = evaluate_dataset(spec, evaluation, localizer, detector["model"], detector["threshold"], device,
                                  max_crops=max_crops)
        out["datasets"][spec["name"]] = {**summarize(result), "seconds": time.perf_counter() - start,
                                         "detector_sha256": detector["sha256"]}
        print(f"[{variant}] {spec['name']} done", flush=True)
    return out


def render(results: dict[str, dict[str, Any]]) -> str:
    lines = []
    names = sorted({name for result in results.values() for name in result["datasets"]})
    for name in names:
        lines.append(f"\n[{name}]  success L1/L2/L3 | always: clean drop, gain L2/L3 | gated: clean drop, gain L2/L3")
        for variant, result in results.items():
            entry = result["datasets"].get(name)
            if entry is None:
                continue
            success = "/".join("-" if entry["localization_success"].get(k) is None
                               else f"{entry['localization_success'][k]:.2f}" for k in ("1", "2", "3"))
            always = (f"{entry['clean_drop']['estimated']:+.4f}, {entry['gain']['2']['estimated']:+.4f}/"
                      f"{entry['gain']['3']['estimated']:+.4f}")
            gated = (f"{entry['clean_drop']['estimated_gated']:+.4f}, {entry['gain']['2']['estimated_gated']:+.4f}/"
                     f"{entry['gain']['3']['estimated_gated']:+.4f}")
            lines.append(f"  {variant:<20} {success} | {always} | {gated}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1-config", default="configs/p27_f1_gate.yaml")
    parser.add_argument("--localizer-checkpoint", default="outputs/p27_localizer/20260916T164212Z/checkpoints/best.pth")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    f1_config, evaluation = load_f1_config(Path(args.f1_config))
    evaluation = with_split(evaluation, args.split)
    device = torch.device(args.device)
    detectors = open_detectors(f1_config, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in args.variants:
        path = output_dir / f"{variant}.json"
        if path.is_file():
            results[variant] = json.loads(path.read_text(encoding="utf-8"))
            continue
        results[variant] = run_variant(variant, Path(args.localizer_checkpoint), f1_config, evaluation,
                                       detectors, device)
        path.write_text(json.dumps(results[variant], indent=2), encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    text = render(results)
    (output_dir / "localizer_ablation.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
