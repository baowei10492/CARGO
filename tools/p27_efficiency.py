"""P27 efficiency: parameters and per-pair latency of the deployed pipeline.

Pipeline for one 256 px pair: localizer T1->T2 and T2->T1 (the selector's
forward-backward feature), warp, detector on the raw and the corrected pair,
selector features and MLP. Each part is timed on the GPU with synchronisation;
the detector of every row is timed separately. RoMa (P25 settings) is timed as
the reference localizer when --roma is given.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from rs_cd.affine_f1 import scoring_mask
from rs_cd.affine_registration import align, grid_points
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.localizer_training import TrainedLocalizer, load_localizer
from rs_cd.p27_detectors import open_detector
from rs_cd.affine_crops import TileCache, select_crops
from rs_cd.p27_selector import (
    TASK_FEATURES,
    feature_matrix,
    registration_features,
    selector_probability,
    task_features,
)

ROWS = {
    "DINOv3 (ours)": ("configs/p27_f1_gate.yaml", "reports/p27_localizer/selector/train/LATEST"),
    "BIT": ("configs/p27_plugins/bit_f1.yaml", "reports/p27_localizer/plugins/bit/selector_train/LATEST"),
    "CD-Lamba": ("configs/p27_plugins/cdlamba_f1.yaml", "reports/p27_localizer/plugins/cdlamba/selector_train/LATEST"),
    "CDxLSTM": ("configs/p27_plugins/cdxlstm_f1.yaml", "reports/p27_localizer/plugins/cdxlstm/selector_train/LATEST"),
}


def count(module: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def timed(device: torch.device, function: Callable[[], Any], repeats: int, warmup: int) -> tuple[Any, dict[str, float]]:
    result = None
    for _ in range(warmup):
        result = function()
    samples = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, {"median_ms": statistics.median(samples), "mean_ms": statistics.fmean(samples)}


def first_crop(spec: dict[str, Any], size: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    item = select_crops(dataset.rows, int(spec["tile_size"]), size, 1)[0]
    t1, t2, _ = TileCache(dataset, int(spec["tile_size"])).crop(item, size)
    return t1, t2


@torch.no_grad()
def measure(localizer: TrainedLocalizer, detector: torch.nn.Module, bundle: dict[str, Any] | None,
            t1: np.ndarray, t2: np.ndarray, device: torch.device, *, repeats: int, warmup: int,
            threshold: float = 0.5) -> dict[str, Any]:
    size = t1.shape[0]
    points = grid_points(size, size, 17)
    (forward, backward), loc = timed(device, lambda: (localizer.estimate_many([t1], [t2])[0],
                                                      localizer.estimate_many([t2], [t1])[0]), repeats, warmup)
    aligned, warp = timed(device, lambda: align(t2, forward.matrix), repeats, warmup)
    t1_batch = normalize(t1).unsqueeze(0).repeat(2, 1, 1, 1).to(device)
    t2_batch = torch.stack([normalize(t2), normalize(aligned)]).to(device)
    logits, det = timed(device, lambda: detector(t1_batch, t2_batch)["change_logits"], repeats, warmup)
    timings = {"localizer_both_directions": loc, "warp": warp, "detector_two_passes": det}
    if bundle is not None:
        def select():
            inside = torch.from_numpy(scoring_mask([forward.matrix], size).astype(np.float32))[None, None].to(device)
            task = task_features(logits[:1].float(), logits[1:].float(), inside, threshold).cpu().tolist()[0]
            record = registration_features(forward.matrix, backward.matrix, forward.misaligned_probability,
                                           points, size)
            record.update(zip(TASK_FEATURES, task))
            return selector_probability(bundle, feature_matrix([record], bundle["feature_names"]))
        _, timings["selector"] = timed(device, select, repeats, warmup)
    timings["total_median_ms"] = sum(value["median_ms"] for value in timings.values() if isinstance(value, dict))
    return timings


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localizer-run", default="outputs/p27_localizer/20260916T164212Z")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--roma", action="store_true", help="also time RoMa + similarity fit (P25 settings)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    evaluation = yaml.safe_load(Path("configs/p27_eval_perturbation.yaml").read_text(encoding="utf-8"))
    spec = next(s for s in evaluation["datasets"] if s["name"] == "LEVIR-CD")
    size = int(evaluation["perturbation"]["crop_size"])
    t1, t2 = first_crop(spec, size)
    model = load_localizer(str(Path(args.localizer_run) / "checkpoints" / "best.pth"), device)
    localizer = TrainedLocalizer(model, device)
    result: dict[str, Any] = {"crop_size": size, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                              "localizer_parameters": count(model), "rows": {}}
    for label, (f1_path, train_pointer) in ROWS.items():
        f1_config = yaml.safe_load(Path(f1_path).read_text(encoding="utf-8"))
        manifest = f1_config.get("p21_freeze_manifest")
        try:
            detector, _, digest, _ = open_detector(f1_config["detectors"]["LEVIR-CD"], device,
                                                   manifest_path=None if manifest is None else Path(manifest))
        except SystemExit as error:
            result["rows"][label] = {"skipped": str(error)}
            continue
        bundle = None
        pointer = Path(train_pointer)
        if pointer.is_file():
            bundle_path = Path(pointer.read_text(encoding="utf-8").strip()) / "bundles" / "selector_LEVIR-CD.pth"
            if bundle_path.is_file():
                bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        entry = {"detector_sha256": digest, "detector_parameters": count(detector),
                 "selector_parameters": None if bundle is None else sum(v.numel() for v in bundle["model_state"].values()),
                 "timing": measure(localizer, detector, bundle, t1, t2, device,
                                   repeats=args.repeats, warmup=args.warmup)}
        _, entry["timing"]["detector_single_pass"] = timed(
            device, lambda: detector(normalize(t1)[None].to(device), normalize(t2)[None].to(device)),
            args.repeats, args.warmup)
        result["rows"][label] = entry
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.roma:
        from tools.run_p25_affine_step1 import build_localizer as build_roma

        roma = build_roma(evaluation, device)
        _, result["roma_similarity_ms"] = timed(device, lambda: roma.estimate(t1, t2, sample_key="timing"),
                                                args.repeats, args.warmup)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
