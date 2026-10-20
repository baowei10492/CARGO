"""P25 step 1: label-free check that RoMa + a similarity fit undoes the
DiffRegCD affine perturbation (docs/P25_AFFINE_PERTURBATION_PROTOCOL.md).

For each dataset's validation split, tiles are cut into 256 px crops and a
fixed, evenly spaced subset is kept. Every crop is estimated clean and at the
three perturbation levels; no detector runs. A self-check first perturbs T1
against itself, which fixes signs, units and the half-pixel convention.
Masks are read only for the descriptive change-fraction column. Test splits
are refused.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    AffineRanges,
    RomaSimilarityLocalizer,
    affine_parameters,
    compose,
    crop_origins,
    dataset_summary,
    grid_points,
    is_success,
    mean_displacement,
    perturb,
    perturbation_matrix,
    selfcheck_passed,
    transfer_error,
)
from rs_cd.datasets.levir import PairedChangeDataset
from rs_cd.runtime import save_json, seed_everything, select_device
from tools.build_whu_scale_probe import select_evenly
from tools.run_p23_whu_strong_localizer import require_fused_local_corr, weights_manifest


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("P25 config is not marked preregistered")
    for spec in config["datasets"]:
        if spec["split"] != "val":
            raise SystemExit(f"step 1 reads validation splits only, got {spec['split']!r}")
    return config


def ranges_from(config: dict[str, Any]) -> AffineRanges:
    perturbation = config["perturbation"]
    low, high = perturbation["scale_range"]
    return AffineRanges(
        max_rotation_deg=float(perturbation["max_rotation_deg"]),
        scale_low=float(low),
        scale_high=float(high),
        max_translation_px=float(perturbation["max_translation_px"]),
        levels=len(perturbation["levels"]),
    )


def build_localizer(config: dict[str, Any], device: torch.device) -> RomaSimilarityLocalizer:
    spec = config["localizer"]
    if spec["kind"] != "roma_outdoor_similarity":
        raise SystemExit(f"unknown localizer {spec['kind']!r}")
    if spec["use_custom_corr"]:
        require_fused_local_corr(str(spec["fused_local_corr_version"]))
    return RomaSimilarityLocalizer(
        num_matches=int(spec["num_matches"]),
        reprojection_px=float(spec["ransac_reprojection_px"]),
        max_iters=int(spec["ransac_max_iters"]),
        confidence=float(spec["ransac_confidence"]),
        refine_iters=int(spec["refine_iters"]),
        min_inliers=int(spec["min_inliers"]),
        device=device,
        seed=int(config["seed"]),
        model_kwargs={"use_custom_corr": bool(spec["use_custom_corr"])},
    )


def select_crops(dataset: PairedChangeDataset, tile_size: int, crop_size: int, cap: int) -> list[dict[str, Any]]:
    crops = [
        {"row": index, "sample_id": row.sample_id, "top": top, "left": left,
         "crop_id": f"{row.sample_id}|y{top:04d}x{left:04d}"}
        for index, row in enumerate(dataset.rows)
        for top, left in crop_origins(tile_size, crop_size)
    ]
    return select_evenly(crops, cap)


def selfcheck_crops(crops, cache, size: int, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """The most textured of the first selected crops (grey-level spread of T1).

    A textureless crop cannot test signs or scale, so it would fail the check
    for a reason unrelated to the conventions it is meant to verify.
    """

    pool = crops[: int(settings["selfcheck_pool"])]
    spreads = [float(cache.crop(item, size)[0].astype(np.float32).mean(axis=2).std()) for item in pool]
    order = sorted(range(len(pool)), key=lambda index: (-spreads[index], index))
    return [pool[index] for index in order[: int(settings["selfcheck_crops"])]]


class TileCache:
    def __init__(self, dataset: PairedChangeDataset, tile_size: int) -> None:
        self.dataset = dataset
        self.tile_size = tile_size
        self.index: int | None = None
        self.arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def crop(self, item: dict[str, Any], size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.index != item["row"]:
            arrays = self.dataset._load_arrays(self.dataset.rows[item["row"]])
            if arrays[0].shape[:2] != (self.tile_size, self.tile_size):
                raise SystemExit(
                    f"{item['sample_id']} is {arrays[0].shape[:2]}, config says {self.tile_size}"
                )
            self.index, self.arrays = item["row"], arrays
        top, left = item["top"], item["left"]
        return tuple(
            np.ascontiguousarray(array[top : top + size, left : left + size])
            for array in self.arrays
        )


def timed(device: torch.device, function, *args, **kwargs):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def run(config: dict[str, Any], device: torch.device, output_dir: Path) -> dict[str, Any]:
    perturbation = config["perturbation"]
    settings = config["step1"]
    size = int(perturbation["crop_size"])
    seed = int(config["seed"])
    ranges = ranges_from(config)
    levels = [int(level) for level in perturbation["levels"]]
    points = grid_points(size, size, int(settings["grid_points"]))
    trigger_px = float(config["localizer"]["trigger_mean_displacement_px"])
    tolerance = float(settings["success_ate_px"])
    localizer = build_localizer(config, device)
    save_json(output_dir / "weights_manifest.json", weights_manifest())

    decision: dict[str, Any] = {
        "gate": "p25_step1",
        "labels_read": "validation masks, change_fraction column only (descriptive)",
        "test_split_touched": False,
        "settings": settings,
        "perturbation": perturbation,
        "localizer": config["localizer"],
        "datasets": {},
    }
    all_rows: list[dict[str, Any]] = []
    selfcheck_rows: list[dict[str, Any]] = []
    for spec in config["datasets"]:
        name = spec["name"]
        dataset = PairedChangeDataset(
            spec["dataset_root"], spec["manifest"], split=spec["split"], normalization="unit",
            mask_threshold=int(spec["mask_threshold"]),
        )
        crops = select_crops(dataset, int(spec["tile_size"]), size, int(settings["max_crops_per_dataset"]))
        cache = TileCache(dataset, int(spec["tile_size"]))

        own_selfcheck = []
        for item in selfcheck_crops(crops, cache, size, settings):
            t1, _, _ = cache.crop(item, size)
            matrix = perturbation_matrix(
                *affine_parameters(item["crop_id"], levels[-1], ranges, seed), width=size, height=size
            )
            estimate = localizer.estimate(t1, perturb(t1, matrix), sample_key=f"selfcheck|{item['crop_id']}")
            own_selfcheck.append(
                {
                    "dataset": name, "crop_id": item["crop_id"], "converged": estimate.converged,
                    "ate_raw": transfer_error(estimate.matrix, matrix, points, width=size, height=size),
                }
            )
        selfcheck = selfcheck_passed(own_selfcheck, float(settings["selfcheck_tolerance_px"]))
        selfcheck_rows.extend(own_selfcheck)
        print(f"[selfcheck] {name}: {selfcheck}")
        if not selfcheck["passed"]:
            decision["datasets"][name] = {"verdict": "selfcheck_failed", "selfcheck": selfcheck}
            continue

        rows = []
        for index, item in enumerate(crops, start=1):
            t1, t2, mask = cache.crop(item, size)
            change = float((mask > int(spec["mask_threshold"])).mean())
            clean_estimate = None
            for level in [CLEAN_LEVEL, *levels]:
                angle, scale, tx, ty = affine_parameters(item["crop_id"], level, ranges, seed)
                matrix = perturbation_matrix(angle, scale, tx, ty, width=size, height=size)
                target = t2 if level == CLEAN_LEVEL else perturb(t2, matrix)
                estimate, seconds = timed(
                    device, localizer.estimate, t1, target, sample_key=f"{item['crop_id']}|L{level}"
                )
                if level == CLEAN_LEVEL:
                    clean_estimate = estimate
                ate_raw = transfer_error(estimate.matrix, matrix, points, width=size, height=size)
                ate_diff = None
                if level != CLEAN_LEVEL and clean_estimate is not None and clean_estimate.converged:
                    ate_diff = transfer_error(
                        estimate.matrix, compose(matrix, clean_estimate.matrix), points,
                        width=size, height=size,
                    )
                displacement = mean_displacement(estimate.matrix, points)
                row = {
                    "dataset": name, "crop_id": item["crop_id"], "level": level,
                    "angle": angle, "scale": scale, "tx": tx, "ty": ty,
                    "perturbation_px": mean_displacement(matrix, points),
                    "converged": estimate.converged, "inliers": estimate.inliers,
                    "inlier_fraction": estimate.inlier_fraction,
                    "a11": estimate.matrix[0, 0], "a12": estimate.matrix[0, 1], "a13": estimate.matrix[0, 2],
                    "a21": estimate.matrix[1, 0], "a22": estimate.matrix[1, 1], "a23": estimate.matrix[1, 2],
                    "estimate_displacement_px": displacement,
                    "trigger": estimate.converged and displacement > trigger_px,
                    "ate_raw": ate_raw, "ate_diff": ate_diff,
                    "seconds": seconds, "change_fraction": change,
                }
                row["success"] = is_success(row, tolerance) if level != CLEAN_LEVEL else None
                rows.append(row)
            if index % 50 == 0 or index == len(crops):
                print(f"[step1 {name} {index}/{len(crops)}]")
        summary = dataset_summary(rows, settings)
        summary.update(selfcheck=selfcheck, crops=len(crops), tiles=len(dataset.rows))
        decision["datasets"][name] = summary
        all_rows.extend(rows)

    _write_csv(output_dir / "step1_estimates.csv", all_rows)
    _write_csv(output_dir / "selfcheck.csv", selfcheck_rows)
    verdicts = {name: entry["verdict"] for name, entry in decision["datasets"].items()}
    decision["verdicts"] = verdicts
    decision["proceed"] = [name for name, verdict in verdicts.items() if verdict == "proceed_to_step2"]
    save_json(output_dir / "step1_decision.json", decision)
    return decision


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p25_affine_protocol.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(int(config["seed"]))
    result = run(config, select_device(args.device), output_dir)
    print(json.dumps(result["verdicts"], indent=2))


if __name__ == "__main__":
    main()
