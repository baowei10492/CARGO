"""Run the P22 gates on WHU-CD (docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md).

gate0  Label-free, train-raster blocks only. Can the frozen LEVIR localizer,
       reading 6x box-downsampled 3072 px windows, recover the physical
       family (native 12/24/48 px) to within one coarse pixel? The native
       2/4/8 px control is measured with the same instrument.
gate1  Validation tiles with labels and the frozen P8 WHU detector.
       anchor  tile-level injection must reproduce the P8 Gate 1 numbers
               (raw on its one-sided support, oracle on paired support),
               proving that rasters, seeds, crops and metrics match P8;
       main    window-level injection with real surrounding context, four
               paths per scenario on one common interior support: raw,
               oracle inverse, estimated alignment, estimated alignment
               behind the q gate.

The WHU test raster is never opened: raster loading refuses it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rs_cd.datasets.levir import IMAGENET_MEAN, IMAGENET_STD
from rs_cd.geometry import interior_valid_mask, warp_reference_from_target
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.physical_shift import (
    CLEAN_KEY,
    anchor_check,
    box_downsample,
    crop,
    gate0_summary,
    gate1_decision,
    p22_scenarios,
    scenario_translation,
    val_tiles,
    window_origin,
)
from rs_cd.robustness import (
    _from_unit,
    _paired_shift_valid_mask,
    _to_unit,
    _translate_reflect,
)
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.build_p21_dualpath_checkpoint import localizer_config_from_payload
from tools.build_whu_scale_probe import blocks_for_scale
from tools.build_whu_tiles import EARLY_YEAR, LATE_YEAR, TWO_PERIOD_DIR, _load_raster

PATHS = ("raw", "oracle", "estimated", "estimated_gated")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _first(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {pattern} under {directory}")
    return matches[0]


def load_rasters(
    raw_root: Path, raster_split: str, *, with_label: bool
) -> dict[str, np.ndarray]:
    if raster_split != "train":
        raise ValueError(
            "P22 reads the train raster only; the WHU test raster stays sealed"
        )
    two_period = raw_root / TWO_PERIOD_DIR
    arrays = {
        "t1": _load_raster(
            _first(two_period / EARLY_YEAR / "whole_image" / raster_split / "image", "*.tif"),
            expect_rgb=True,
        ),
        "t2": _load_raster(
            _first(two_period / LATE_YEAR / "whole_image" / raster_split / "image", "*.tif"),
            expect_rgb=True,
        ),
    }
    if with_label:
        arrays["label"] = _load_raster(
            _first(two_period / "change_label" / raster_split, "change_label.tif"),
            expect_rgb=False,
        )
    shapes = {key: value.shape[:2] for key, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"raster shapes disagree: {shapes}")
    return arrays


def unit_region(
    array: np.ndarray, left: int, top: int, size: int, device: torch.device
) -> torch.Tensor:
    """RGB region in [0, 1], computed exactly as PairedChangeDataset does."""

    block = np.ascontiguousarray(array[top : top + size, left : left + size])
    return torch.from_numpy(block).to(device).permute(2, 0, 1).float() / 255.0


def normalize_cpu(unit: torch.Tensor) -> torch.Tensor:
    return (unit.cpu() - IMAGENET_MEAN) / IMAGENET_STD


def normalize_device(unit: torch.Tensor) -> torch.Tensor:
    return (unit - IMAGENET_MEAN.to(unit)) / IMAGENET_STD.to(unit)


def label_tile(array: np.ndarray, left: int, top: int, size: int) -> torch.Tensor:
    region = array[top : top + size, left : left + size]
    return torch.from_numpy((region > 127).astype(np.float32)).unsqueeze(0)


def load_localizer(
    path: Path, expected_sha256: str | None, device: torch.device
) -> tuple[torch.nn.Module, str, float]:
    """Frozen localizer, its digest and its search boundary in coarse px.

    The boundary is derived from the checkpoint (search_radius descriptor
    cells x the stride-2 descriptor) rather than typed into the config, so
    the clamping annotation cannot drift from the instrument.
    """

    digest = _sha256(path)
    if expected_sha256 and digest != expected_sha256:
        raise SystemExit(
            f"localizer {path} has sha256 {digest}, expected {expected_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = localizer_config_from_payload(payload)
    model = build_model({"model": model_config})
    model.load_state_dict(payload["model"], strict=True)
    boundary = 2.0 * float(model_config.get("search_radius", 4))
    return model.to(device).eval(), digest, boundary


def load_detector(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_model(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), config, _sha256(path)


@torch.no_grad()
def estimate_offset(
    localizer: torch.nn.Module,
    coarse_t1: torch.Tensor,
    t2_unit_window: torch.Tensor,
    *,
    factor: int,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Conditional and gated offsets in native pixels, plus the q gate."""

    coarse_t2 = normalize_device(box_downsample(t2_unit_window.unsqueeze(0), factor))
    with torch.autocast(
        device_type=coarse_t1.device.type,
        enabled=amp and coarse_t1.device.type == "cuda",
    ):
        registration = localizer.estimate_registration(coarse_t1, coarse_t2)
    conditional = registration["conditional_hard_offset_pixels"].float()[0] * factor
    applied = registration["applied_offset_pixels"].float()[0] * factor
    gate = bool(registration["alignment_sample_gate"][0] >= 0.5)
    return conditional, applied, gate


def _shift(window: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return window
    return _translate_reflect(window, dx=dx, dy=dy)[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _offset_row(
    sample_id: str,
    scenario: Any,
    dx: int,
    dy: int,
    conditional: torch.Tensor,
    applied: torch.Tensor,
    gate: bool,
    factor: int,
) -> dict[str, Any]:
    error = math.hypot(float(conditional[0]) - dx, float(conditional[1]) - dy)
    return {
        "id": sample_id,
        "scenario": scenario.key,
        "family": scenario.family,
        "severity": scenario.severity,
        "gt_dx": float(dx),
        "gt_dy": float(dy),
        "est_dx": float(conditional[0]),
        "est_dy": float(conditional[1]),
        "applied_dx": float(applied[0]),
        "applied_dy": float(applied[1]),
        "q_gate": bool(gate),
        "native_error": error,
        "coarse_error": error / factor,
    }


def run_gate0(config: dict[str, Any], device: torch.device, output_dir: Path) -> dict[str, Any]:
    geometry = config["geometry"]
    factor = int(geometry["downsample_factor"])
    window = int(geometry["window_size"])
    seed = int(config["seed"])
    rasters = load_rasters(Path(config["raw_root"]), "train", with_label=False)
    region = geometry["train_region"]
    blocks, stride = blocks_for_scale(
        int(region["x_max"]),
        int(region["y_max"]),
        window,
        max_blocks=int(config["gate0"]["blocks"]),
    )
    localizer, localizer_sha, boundary = load_localizer(
        Path(config["checkpoints"]["localizer"]),
        config["checkpoints"].get("localizer_sha256"),
        device,
    )
    scenarios = p22_scenarios(factor)
    amp = bool(config.get("amp", True))
    records: list[dict[str, Any]] = []
    for index, (left, top) in enumerate(blocks, start=1):
        sample_id = f"whu_p22_block_y{top:05d}_x{left:05d}"
        t1 = unit_region(rasters["t1"], left, top, window, device)
        t2 = unit_region(rasters["t2"], left, top, window, device)
        coarse_t1 = normalize_device(box_downsample(t1.unsqueeze(0), factor))
        for scenario in scenarios:
            dx, dy = scenario_translation(sample_id, scenario, seed)
            conditional, applied, gate = estimate_offset(
                localizer, coarse_t1, _shift(t2, dx, dy), factor=factor, amp=amp
            )
            records.append(
                _offset_row(sample_id, scenario, dx, dy, conditional, applied, gate, factor)
            )
        print(f"[gate0 {index}/{len(blocks)}] {sample_id}")

    summary = gate0_summary(
        records, factor=factor, thresholds=config["gate0"], boundary_coarse_px=boundary
    )
    summary.update(
        {
            "gate": "p22_gate0",
            "blocks": len(blocks),
            "block_stride": stride,
            "localizer_sha256": localizer_sha,
            "labels_read": False,
            "validation_split_touched": False,
            "test_raster_touched": False,
            "physical_shift_m": [
                pixels * float(geometry["gsd_m"]) for pixels in (2 * factor, 4 * factor, 8 * factor)
            ],
        }
    )
    _write_csv(output_dir / "gate0_offsets.csv", records)
    save_json(output_dir / "gate0_decision.json", summary)
    return summary


def _p8_protocol(config: dict[str, Any]) -> tuple[int, int | None]:
    """Seed and batch size of the P8 Gate 1 run the anchor reproduces.

    Both are read from the P8 protocol file rather than retyped: the seed
    fixes every injected direction, and the batch size fixes the cuDNN
    algorithm selection, which is the only remaining source of numeric
    difference between two fp32 evaluations of identical inputs.
    """

    protocol = yaml.safe_load(
        Path(config["checkpoints"]["p8_oracle_protocol"]).read_text(encoding="utf-8")
    )
    seed = int(protocol.get("seed", 3407))
    entries = [entry for entry in protocol["corruptions"] if entry.get("name") == "shift"]
    if not entries:
        raise SystemExit("P8 oracle protocol has no shift corruption")
    severities = [int(value) for value in entries[0].get("severities", [1, 2, 3])]
    modes = list(entries[0].get("modes", ["t1", "t2", "common"]))
    if severities != [1, 2, 3] or "t2" not in modes:
        raise SystemExit(f"P8 oracle protocol shift entry is {entries[0]}")
    if seed != int(config["seed"]):
        raise SystemExit(f"P8 seed {seed} differs from the P22 seed {config['seed']}")
    batch_size = protocol.get("batch_size")
    return seed, (int(batch_size) if batch_size is not None else None)


@torch.no_grad()
def run_anchor(
    detector: torch.nn.Module,
    rasters: dict[str, np.ndarray],
    tiles: list[dict[str, Any]],
    *,
    tile_size: int,
    seed: int,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Tile-level reflect injection, exactly as P8 Gate 1 evaluated it.

    ``raw`` is scored on the one-sided support P8's raw run used and
    ``oracle`` on paired support, reproducing main record 71.1 as it was
    measured. ``raw_paired`` (same logits, paired support) is reported in
    addition, because main record 153.3 found that P8's headroom mixed the
    two supports; it is not part of the anchor comparison.
    """

    observed: dict[str, float] = {}
    scenarios = [s for s in p22_scenarios(1) if s.family in {"clean", "pixel"}]
    for scenario in scenarios:
        paths = ["raw"] if scenario.family == "clean" else ["raw", "raw_paired", "oracle"]
        metrics = {path: BinaryChangeMetrics(threshold=threshold) for path in paths}
        buffers: dict[str, list[tuple[torch.Tensor, ...]]] = {path: [] for path in paths}

        def flush(path: str) -> None:
            if not buffers[path]:
                return
            t1, t2, mask, valid = (torch.stack(items) for items in zip(*buffers[path]))
            logits = detector(t1.to(device), t2.to(device))["change_logits"]
            metrics[path].update(
                logits, mask.to(device), from_logits=True, valid_mask=valid.to(device)
            )
            buffers[path].clear()

        for tile in tiles:
            left, top = tile["left"], tile["top"]
            t1 = normalize_cpu(unit_region(rasters["t1"], left, top, tile_size, torch.device("cpu")))
            t2 = normalize_cpu(unit_region(rasters["t2"], left, top, tile_size, torch.device("cpu")))
            mask = label_tile(rasters["label"], left, top, tile_size)
            dx, dy = scenario_translation(tile["sample_id"], scenario, seed)
            if scenario.family == "clean":
                buffers["raw"].append((t1, t2, mask, torch.ones_like(mask)))
            else:
                shifted, one_sided = _translate_reflect(_to_unit(t2, "imagenet"), dx=dx, dy=dy)
                raw_t2 = _from_unit(shifted, "imagenet")
                oracle_t2, _ = _translate_reflect(raw_t2, dx=-dx, dy=-dy)
                paired = _paired_shift_valid_mask(tile_size, tile_size, dx, dy)
                buffers["raw"].append((t1, raw_t2, mask, one_sided))
                buffers["raw_paired"].append((t1, raw_t2, mask, paired))
                buffers["oracle"].append((t1, oracle_t2, mask, paired))
            for path in paths:
                if len(buffers[path]) == batch_size:
                    flush(path)
        for path in paths:
            flush(path)
        if scenario.family == "clean":
            observed["clean"] = float(metrics["raw"].compute()["f1"])
        else:
            for path in paths:
                observed[f"{path}:{scenario.key}"] = float(metrics[path].compute()["f1"])
        print(f"[anchor] {scenario.key} " + " ".join(f"{k}={v:.5f}" for k, v in observed.items() if scenario.key in k or k == "clean"))
    return observed


@torch.no_grad()
def run_main(
    detector: torch.nn.Module,
    localizer: torch.nn.Module,
    rasters: dict[str, np.ndarray],
    tiles: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    geometry = config["geometry"]
    factor = int(geometry["downsample_factor"])
    window = int(geometry["window_size"])
    tile_size = int(geometry["tile_size"])
    seed = int(config["seed"])
    amp = bool(config.get("amp", True))
    height, width = rasters["t1"].shape[:2]
    scenarios = p22_scenarios(factor)
    # One common interior support for every scenario and path (the P12 rule
    # "same valid pixels per scenario"). The border must cover the largest
    # injected shift, so no reflected or border-lost pixel is ever scored and
    # damage, headroom and recovery are all differences on identical pixels.
    border = int(config["gate1"]["evaluation_border_pixels"])
    largest_shift = max(scenario.pixels for scenario in scenarios)
    if border < largest_shift:
        raise SystemExit(
            f"evaluation_border_pixels {border} is smaller than the largest "
            f"injected shift {largest_shift}; supports would differ"
        )
    valid = interior_valid_mask(1, tile_size, tile_size, border, device=device)
    metrics = {
        scenario.key: {path: BinaryChangeMetrics(threshold=threshold) for path in PATHS}
        for scenario in scenarios
    }
    offset_rows: list[dict[str, Any]] = []
    for index, tile in enumerate(tiles, start=1):
        left, top = tile["left"], tile["top"]
        window_left, window_top = window_origin(
            left, top, tile_size=tile_size, window_size=window,
            raster_width=width, raster_height=height,
        )
        box = dict(tile_left=left, tile_top=top, window_left=window_left,
                   window_top=window_top, tile_size=tile_size)
        t1_window = unit_region(rasters["t1"], window_left, window_top, window, device)
        t2_window = unit_region(rasters["t2"], window_left, window_top, window, device)
        coarse_t1 = normalize_device(box_downsample(t1_window.unsqueeze(0), factor))
        t1_tile = normalize_cpu(crop(t1_window, **box))
        mask = label_tile(rasters["label"], left, top, tile_size).unsqueeze(0).to(device)
        for scenario in scenarios:
            dx, dy = scenario_translation(tile["sample_id"], scenario, seed)
            shifted = _shift(t2_window, dx, dy)
            oracle_window = _shift(shifted, -dx, -dy)
            conditional, applied, gate = estimate_offset(
                localizer, coarse_t1, shifted, factor=factor, amp=amp
            )
            aligned = warp_reference_from_target(
                shifted.unsqueeze(0), conditional.unsqueeze(0).to(shifted)
            )[0]
            candidates = {
                "raw": crop(shifted, **box),
                "oracle": crop(oracle_window, **box),
                "estimated": crop(aligned, **box),
            }
            candidates["estimated_gated"] = candidates["estimated"] if gate else candidates["raw"]
            t2_batch = normalize_cpu(torch.stack([candidates[path] for path in PATHS]))
            t1_batch = t1_tile.unsqueeze(0).repeat(len(PATHS), 1, 1, 1)
            logits = detector(t1_batch.to(device), t2_batch.to(device))["change_logits"]
            for position, path in enumerate(PATHS):
                metrics[scenario.key][path].update(
                    logits[position : position + 1], mask, from_logits=True, valid_mask=valid
                )
            offset_rows.append(
                _offset_row(tile["sample_id"], scenario, dx, dy, conditional, applied, gate, factor)
            )
        print(f"[gate1 {index}/{len(tiles)}] {tile['sample_id']} window=({window_left},{window_top})")
    table = {
        key: {path: metrics[key][path].compute() for path in PATHS} for key in metrics
    }
    return table, offset_rows


def run_gate1(config: dict[str, Any], device: torch.device, output_dir: Path) -> dict[str, Any]:
    geometry = config["geometry"]
    tile_size = int(geometry["tile_size"])
    seed, protocol_batch = _p8_protocol(config)
    rasters = load_rasters(Path(config["raw_root"]), "train", with_label=True)
    tiles = val_tiles(list(geometry["val_columns"]), int(geometry["val_rows"]), tile_size)
    detector, detector_config, detector_sha = load_detector(
        Path(config["checkpoints"]["detector"]), device
    )
    threshold = float(detector_config["evaluation"].get("threshold", 0.5))
    gate1 = config["gate1"]
    # Same resolution order as tools/evaluate_robustness.py, so the anchor
    # batches exactly as the P8 run did.
    anchor_batch = (
        protocol_batch
        if protocol_batch is not None
        else int(
            detector_config["evaluation"].get(
                "batch_size", detector_config["training"]["batch_size"]
            )
        )
    )

    observed = run_anchor(
        detector, rasters, tiles, tile_size=tile_size, seed=seed,
        threshold=threshold, batch_size=anchor_batch, device=device,
    )
    reference = {"clean": gate1["anchor_reference"]["clean"]}
    for key, value in gate1["anchor_reference"]["raw"].items():
        reference[f"raw:{key}"] = value
    for key, value in gate1["anchor_reference"]["oracle"].items():
        reference[f"oracle:{key}"] = value
    anchor = anchor_check(observed, reference, float(gate1["anchor_tolerance"]))
    anchor["detector_sha256"] = detector_sha
    anchor["anchor_batch_size"] = anchor_batch
    anchor["raw_paired_not_compared"] = {
        key: value for key, value in observed.items() if key.startswith("raw_paired:")
    }
    save_json(output_dir / "gate1_anchor.json", anchor)
    print(json.dumps(anchor, indent=2))
    if not anchor["passed"]:
        decision = {"gate": "p22_gate1", "verdict": "anchor_failed", "anchor": anchor}
        save_json(output_dir / "gate1_decision.json", decision)
        return decision

    localizer, localizer_sha, boundary = load_localizer(
        Path(config["checkpoints"]["localizer"]),
        config["checkpoints"].get("localizer_sha256"),
        device,
    )
    table, offset_rows = run_main(
        detector, localizer, rasters, tiles, config=config, threshold=threshold, device=device
    )
    rows = [
        {"scenario": key, "path": path, **values}
        for key, by_path in table.items()
        for path, values in by_path.items()
    ]
    _write_csv(output_dir / "gate1_scenario_metrics.csv", rows)
    _write_csv(output_dir / "gate1_offsets.csv", offset_rows)
    f1 = {key: {path: float(values["f1"]) for path, values in by_path.items()} for key, by_path in table.items()}
    support = {
        key: {path: int(v["tp"] + v["fp"] + v["fn"] + v["tn"]) for path, v in by_path.items()}
        for key, by_path in table.items()
    }
    support_values = {value for by_path in support.values() for value in by_path.values()}
    if len(support_values) != 1:
        raise SystemExit(f"supports differ across scenarios or paths: {support}")
    decision = gate1_decision(f1, anchor_passed=True, thresholds=gate1)
    offsets = gate0_summary(
        offset_rows, factor=int(geometry["downsample_factor"]),
        thresholds=config["gate0"], boundary_coarse_px=boundary,
    )
    decision.update(
        {
            "gate": "p22_gate1",
            "f1": f1,
            "anchor": anchor,
            "validation_offsets": offsets["scenarios"],
            "tiles": len(tiles),
            "detector_sha256": detector_sha,
            "localizer_sha256": localizer_sha,
            "test_raster_touched": False,
            "support": {
                "rule": "common interior border for every scenario and path",
                "border_pixels": int(gate1["evaluation_border_pixels"]),
                "valid_pixels_per_scenario": support_values.pop(),
            },
        }
    )
    save_json(output_dir / "gate1_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=["gate0", "gate1"])
    parser.add_argument("--config", default="configs/p22_whu_physical_shift.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("P22 config is not marked preregistered")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(int(config["seed"]))
    device = select_device(args.device)
    runner = run_gate0 if args.gate == "gate0" else run_gate1
    result = runner(config, device, output_dir)
    print(f"{args.gate} verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
