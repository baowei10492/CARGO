"""P29b: does a detector read the zero border of the synthetic perturbation?

docs/P29B_BORDER_CUE_DIAGNOSIS.md. Validation crops of the P27 evaluation, levels
1-3; for each crop and level the second image is given to both detectors of a
dataset (the original DINOv3 detector and the P29 augmented one) as:

    clean          T2
    raw            T2 warped with the level's matrix, zero outside (P27 raw path)
    border_raw     T2 not moved, zeroed where raw is zero
    oracle         raw put back with the true matrix (P27 oracle path)
    border_oracle  T2 not moved, zeroed where oracle is zero
    context_raw    the tile warped with the matrix (reflected at the tile edge),
                   the crop cut out: the misalignment without a zero border

All scored on the P27 scoring pixels. Diagnosis only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from rs_cd.affine_crops import TileCache, in_frame, select_margin_crops, source_inside_tile
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import AffineRanges, affine_parameters, align, perturb, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.metrics.binary import BinaryChangeMetrics
from tools.run_p27_f1 import all_crops, load_config, open_detectors

VIEWS = ("raw", "border_raw", "oracle", "border_oracle", "context_raw")
LEVELS = (1, 2, 3)
DETECTOR_CONFIGS = {"original": Path("configs/p27_f1_gate.yaml"),
                    "augmented": Path("configs/p27_plugins/dinov3aug_f1.yaml")}


def covered(matrix: np.ndarray, size: int, *, inverse: bool = False) -> np.ndarray:
    """Pixels that receive content when a size x size image is warped (nearest)."""
    flags = cv2.INTER_NEAREST | (cv2.WARP_INVERSE_MAP if inverse else 0)
    ones = np.ones((size, size), dtype=np.uint8)
    return cv2.warpAffine(ones, matrix, (size, size), flags=flags, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool)


def tile_matrix(matrix: np.ndarray, top: int, left: int) -> np.ndarray:
    """The crop-frame matrix expressed in the tile frame: p -> M (p - o) + o."""
    return in_frame(matrix, top, left)


def views(t2: np.ndarray, tile_t2: np.ndarray, top: int, left: int, matrix: np.ndarray) -> dict[str, np.ndarray]:
    size = t2.shape[0]
    raw = perturb(t2, matrix)
    oracle = align(raw, matrix)
    raw_region = covered(matrix, size)  # p with M^-1 p inside the crop
    oracle_region = covered(matrix, size, inverse=True)  # p with M p inside the crop
    height, width = tile_t2.shape[:2]
    warped_tile = cv2.warpAffine(tile_t2, tile_matrix(matrix, top, left), (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
    return {
        "raw": raw,
        "border_raw": np.where(raw_region[..., None], t2, 0).astype(np.uint8),
        "oracle": oracle,
        "border_oracle": np.where(oracle_region[..., None], t2, 0).astype(np.uint8),
        "context_raw": np.ascontiguousarray(warped_tile[top:top + size, left:left + size]),
    }


def crops_for(dataset: PairedChangeDataset, spec: dict[str, Any], size: int, margin: int | None) -> list[dict[str, Any]]:
    """P27 crops, or (P30) crops keeping ``margin`` real pixels to every tile edge."""
    if margin is None:
        return all_crops(dataset, int(spec["tile_size"]), size)
    return select_margin_crops(dataset.rows, int(spec["tile_size"]), size, margin)


def check_no_padding(item: dict[str, Any], matrix: np.ndarray, size: int, tile_size: int) -> None:
    if not source_inside_tile(matrix, item["top"], item["left"], size, tile_size):
        raise SystemExit(f"{item['crop_id']}: the perturbation reaches outside the tile; the margin is too small")


@torch.no_grad()
def diagnose(spec: dict[str, Any], evaluation: dict[str, Any], detectors: dict[str, dict[str, Any]],
             device: torch.device, max_crops: int | None = None, margin: int | None = None) -> dict[str, Any]:
    perturbation = evaluation["perturbation"]
    size = int(perturbation["crop_size"])
    levels = [int(level) for level in perturbation["levels"]]
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels))
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    crops = crops_for(dataset, spec, size, margin)[:max_crops]
    cache = TileCache(dataset, int(spec["tile_size"]))
    keys = ["clean"] + [f"{level}/{view}" for level in LEVELS for view in VIEWS]
    totals = {name: {key: BinaryChangeMetrics(threshold=entry["threshold"]) for key in keys}
              for name, entry in detectors.items()}
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        tile_t2 = cache.arrays[1]
        matrices = {level: perturbation_matrix(*affine_parameters(item["crop_id"], level, ranges,
                                                                  int(evaluation["seed"])), width=size, height=size)
                    for level in levels}
        valid = scoring_mask(list(matrices.values()), size)
        if margin is not None:
            for matrix in matrices.values():
                check_no_padding(item, matrix, size, int(spec["tile_size"]))
        images = [t2]
        for level in LEVELS:
            level_views = views(t2, tile_t2, item["top"], item["left"], matrices[level])
            images += [level_views[view] for view in VIEWS]
        t2_batch = torch.stack([normalize(image) for image in images]).to(device)
        t1_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        for name, entry in detectors.items():
            logits = entry["model"](t1_batch, t2_batch)["change_logits"]
            for key, counts in zip(keys, confusion_counts(logits, target, keep, entry["threshold"])):
                add_counts(totals[name][key], counts)
        if index % 200 == 0 or index == len(crops):
            print(f"[p29b {spec['name']} {index}/{len(crops)}]", flush=True)
    return {"crops": len(crops),
            "f1": {name: {key: float(m.compute()["f1"]) for key, m in by_key.items()} for name, by_key in totals.items()},
            "recall": {name: {key: float(m.compute()["recall"]) for key, m in by_key.items()}
                       for name, by_key in totals.items()}}


def reading(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The rule of docs/P29B_BORDER_CUE_DIAGNOSIS.md: level-3 drops from clean, mean over datasets."""
    def drop(detector: str, view: str) -> float:
        values = [(r["f1"][detector]["clean"] - r["f1"][detector][f"3/{view}"]) * 100 for r in results.values()]
        return sum(values) / len(values)

    d = {name: {view: drop(name, view) for view in VIEWS} for name in DETECTOR_CONFIGS}
    augmented = d["augmented"]
    context_extra = augmented["context_raw"] - augmented["raw"]
    if augmented["border_raw"] >= 5 and context_extra >= 5 and d["original"]["border_raw"] <= 2:
        verdict = "confirmed"
    elif augmented["border_raw"] <= 2 and context_extra <= 2:
        verdict = "ruled_out"
    else:
        verdict = "partial"
    return {"verdict": verdict, "level3_drop_points": d, "context_minus_raw_augmented": context_extra}


def render(name: str, result: dict[str, Any]) -> str:
    lines = [f"[{name}] {result['crops']} crops, F1 %"]
    for detector, f1 in result["f1"].items():
        lines.append(f"  {detector:<9} clean {f1['clean'] * 100:5.1f}")
        for level in LEVELS:
            lines.append(f"    L{level} " + "  ".join(f"{view} {f1[f'{level}/{view}'] * 100:5.1f}" for view in VIEWS))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-crops", type=int, default=None, help="per dataset (default: all validation crops)")
    parser.add_argument("--margin", type=int, default=None,
                        help="P30: crops keeping this many real pixels to the tile edge (no padding at all)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    configs = {name: load_config(path) for name, path in DETECTOR_CONFIGS.items()}
    evaluation = configs["original"][1]
    if configs["augmented"][1] != evaluation:
        raise SystemExit("the two detector configs read different evaluation settings")
    opened = {name: open_detectors(config, device) for name, (config, _) in configs.items()}
    results = {}
    for spec in evaluation["datasets"]:
        if spec["split"] != "val":
            raise SystemExit("the diagnosis runs on validation crops only")
        detectors = {name: opened[name][spec["name"]] for name in DETECTOR_CONFIGS}
        results[spec["name"]] = diagnose(spec, evaluation, detectors, device, args.max_crops, args.margin)
        results[spec["name"]]["detector_sha256"] = {name: d["sha256"] for name, d in detectors.items()}
        print(render(spec["name"], results[spec["name"]]), flush=True)
    summary = {"results": results, "reading": reading(results), "margin": args.margin}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["reading"], indent=2))


if __name__ == "__main__":
    main()
