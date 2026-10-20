"""P30 pre-check: the frozen P27 localizer and original detectors without the zero border.

docs/P30_PRECHECK.md. Validation crops only. For every crop and level 1-3 the
misaligned T2 is made twice, `bordered` (the P27 perturbation) and
`border_free` (the same matrix applied to the whole tile, the crop cut out);
the P27 localizer estimates the transform with the P27 step-1 routine and the
original DINOv3 detector scores raw / oracle / estimated / gated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import add_counts, confusion_counts, gated_source, path_images, scoring_mask
from rs_cd.affine_registration import AffineRanges, grid_points, perturb, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.localizer_training import TrainedLocalizer, load_localizer, step1_rows
from rs_cd.metrics.binary import BinaryChangeMetrics
from tools.p29_border_diagnosis import check_no_padding, crops_for, tile_matrix
from tools.run_p27_f1 import load_config, open_detectors

MODES = ("bordered", "border_free")
PATHS = ("raw", "oracle", "estimated", "gated")
LEVELS = (0, 1, 2, 3)


def border_free(tile_t2: np.ndarray, top: int, left: int, size: int, matrix: np.ndarray) -> np.ndarray:
    height, width = tile_t2.shape[:2]
    warped = cv2.warpAffine(tile_t2, tile_matrix(matrix, top, left), (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(warped[top:top + size, left:left + size])


def _matrix(row: dict[str, Any]) -> np.ndarray:
    return np.array([[row["a11"], row["a12"], row["a13"]], [row["a21"], row["a22"], row["a23"]]])


@torch.no_grad()
def check(spec: dict[str, Any], evaluation: dict[str, Any], localizer: TrainedLocalizer, detector: dict[str, Any],
          device: torch.device, max_crops: int | None = None, margin: int | None = None) -> dict[str, Any]:
    perturbation, settings = evaluation["perturbation"], evaluation["step1"]
    size = int(perturbation["crop_size"])
    levels = [int(level) for level in perturbation["levels"]]
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels))
    points = grid_points(size, size, int(settings["grid_points"]))
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    crops = crops_for(dataset, spec, size, margin)[:max_crops]
    cache = TileCache(dataset, int(spec["tile_size"]))
    totals = {mode: {f"{level}/{path}": BinaryChangeMetrics(threshold=detector["threshold"])
                     for level in LEVELS for path in PATHS} for mode in MODES}
    success = {mode: {level: [] for level in levels} for mode in MODES}
    for index, item in enumerate(crops, start=1):
        t1, t2, mask = cache.crop(item, size)
        tile_t2 = cache.arrays[1]
        makers = {"bordered": None,
                  "border_free": lambda m: border_free(tile_t2, item["top"], item["left"], size, m)}
        images, keys = [], []
        valid = None
        for mode, maker in makers.items():
            rows = step1_rows(localizer, dataset=spec["name"], crop_id=item["crop_id"], t1=t1, t2=t2,
                              change_fraction=0.0, ranges=ranges, levels=levels, seed=int(evaluation["seed"]),
                              points=points, tolerance=float(settings["success_ate_px"]),
                              trigger_probability=float(evaluation["p27"]["trigger_probability"]), make_target=maker)
            truths = {int(r["level"]): perturbation_matrix(float(r["angle"]), float(r["scale"]), float(r["tx"]),
                                                           float(r["ty"]), width=size, height=size) for r in rows}
            if valid is None:
                valid = scoring_mask([truths[level] for level in levels], size)
                if margin is not None:
                    for level in levels:
                        check_no_padding(item, truths[level], size, int(spec["tile_size"]))
            for row in rows:
                level = int(row["level"])
                if level:
                    success[mode][level].append(bool(row["success"]))
                moved = t2 if level == 0 else (perturb(t2, truths[level]) if maker is None else maker(truths[level]))
                views = path_images(moved, truths[level], _matrix(row))
                views["gated"] = views[gated_source(bool(row["trigger"]))]
                for path in PATHS:
                    images.append(normalize(views[path]))
                    keys.append((mode, f"{level}/{path}"))
        t2_batch = torch.stack(images).to(device)
        t1_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
        logits = detector["model"](t1_batch, t2_batch)["change_logits"]
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
        for (mode, key), counts in zip(keys, confusion_counts(logits, target, keep, detector["threshold"])):
            add_counts(totals[mode][key], counts)
        if index % 200 == 0 or index == len(crops):
            print(f"[p30 precheck {spec['name']} {index}/{len(crops)}]", flush=True)
    return {"crops": len(crops),
            "success": {mode: {str(level): float(np.mean(v)) for level, v in by_level.items()}
                        for mode, by_level in success.items()},
            "f1": {mode: {key: float(m.compute()["f1"]) for key, m in by_key.items()} for mode, by_key in totals.items()}}


def reading(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mean = {mode: sum(r["success"][mode]["3"] for r in results.values()) / len(results) * 100 for mode in MODES}
    drop = mean["bordered"] - mean["border_free"]
    return {"level3_success_percent": mean, "drop_points": drop,
            "verdict": "no_border_dependence" if drop <= 5 else "retrain_localizer"}


def render(name: str, result: dict[str, Any]) -> str:
    lines = [f"[{name}] {result['crops']} crops"]
    for mode in MODES:
        s = result["success"][mode]
        f1 = result["f1"][mode]
        lines.append(f"  {mode:<11} success L1/L2/L3 " + " / ".join(f"{s[str(l)] * 100:.1f}" for l in (1, 2, 3)))
        for level in LEVELS:
            lines.append(f"    L{level} " + "  ".join(f"{path} {f1[f'{level}/{path}'] * 100:5.1f}" for path in PATHS))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--margin", type=int, default=None,
                        help="P30: crops keeping this many real pixels to the tile edge (no padding at all)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    f1_config, evaluation = load_config(Path("configs/p27_f1_gate.yaml"))
    selector = yaml.safe_load(Path("configs/p27_selector.yaml").read_text(encoding="utf-8"))
    checkpoint = Path(selector["localizer_run"]) / "checkpoints" / selector["localizer_checkpoint"]
    localizer = TrainedLocalizer(load_localizer(str(checkpoint), device), device)
    detectors = open_detectors(f1_config, device)
    results = {}
    for spec in evaluation["datasets"]:
        if spec["split"] != "val":
            raise SystemExit("the pre-check runs on validation crops only")
        results[spec["name"]] = check(spec, evaluation, localizer, detectors[spec["name"]], device, args.max_crops,
                                      args.margin)
        print(render(spec["name"], results[spec["name"]]), flush=True)
    summary = {"results": results, "reading": reading(results), "margin": args.margin}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["reading"], indent=2))


if __name__ == "__main__":
    main()
