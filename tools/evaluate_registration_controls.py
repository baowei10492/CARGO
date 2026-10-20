from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.geometry import interior_valid_mask, warp_reference_from_target
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.registration_controls import (
    RegistrationControlResult,
    ecc_translation_offset,
    phase_correlation_offset,
)
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_hash(protocol: dict[str, Any]) -> str:
    text = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _offset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    errors = np.asarray([float(row["epe_pixels"]) for row in rows])
    return {
        "samples": len(rows),
        "mean_epe_pixels": float(errors.mean()),
        "median_epe_pixels": float(np.median(errors)),
        "p90_epe_pixels": float(np.quantile(errors, 0.90)),
        "p95_epe_pixels": float(np.quantile(errors, 0.95)),
        "pck_at_1": float(np.mean(errors <= 1.0)),
        "pck_at_2": float(np.mean(errors <= 2.0)),
        "pck_at_4": float(np.mean(errors <= 4.0)),
        "exact_offset_rate": float(np.mean(errors <= 1e-6)),
        "converged_rate": float(
            np.mean([bool(row["converged"]) for row in rows])
        ),
        "fallback_rate": float(
            np.mean([bool(row["used_fallback"]) for row in rows])
        ),
    }


def _detector_logits(
    model: torch.nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    offsets: torch.Tensor | None,
) -> torch.Tensor:
    """Run the frozen detector without accidentally invoking registration.

    P7 registration models implement ``forward_with_forced_offset``.
    Calling their ordinary forward after externally warping T2 would run
    a second learned registration pass and contaminate this control.
    """

    forced_forward = getattr(model, "forward_with_forced_offset", None)
    if callable(forced_forward):
        forced_offsets = (
            torch.zeros(
                (int(t1.shape[0]), 2),
                device=t1.device,
                dtype=t1.dtype,
            )
            if offsets is None
            else offsets
        )
        outputs = forced_forward(t1, t2, forced_offsets)
        key = (
            "raw_change_logits"
            if offsets is None
            else "aligned_change_logits"
        )
    else:
        detector_t2 = (
            t2
            if offsets is None
            else warp_reference_from_target(t2, offsets)
        )
        outputs = model(t1, detector_t2)
        key = "change_logits"
    logits = outputs[key]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Model did not return Tensor logits at {key}")
    return logits


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Phase Correlation and ECC as frozen registration "
            "controls before a change detector."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p7_registration_controls_t2.yaml",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Registration-control protocol must be a mapping")
    methods = tuple(str(value) for value in protocol["methods"])
    allowed = {"raw", "phase", "ecc", "oracle"}
    if not set(methods).issubset(allowed):
        raise ValueError(f"Unsupported control method in {methods}")
    if "raw" not in methods or "oracle" not in methods:
        raise ValueError("Control protocol must include raw and oracle")
    if "ecc" in methods:
        try:
            import cv2  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "Protocol requests ECC. Install with "
                "pip install -e '.[geometry]'"
            ) from error

    device = select_device(args.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = copy.deepcopy(checkpoint["config"])
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    seed = int(protocol["seed"])
    seed_everything(seed)
    split = str(args.split or protocol.get("split", "val"))
    data = config["data"]
    normalization = str(data.get("normalization", "imagenet"))
    base_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=split,
        image_size=data.get("eval_crop_size"),
        normalization=normalization,
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    specs = [
        CorruptionSpec(
            name="clean",
            severity=0,
            mode="common",
            seed=seed,
        ),
        *[
            CorruptionSpec(
                name="shift",
                severity=int(severity),
                mode="t2",
                seed=seed,
            )
            for severity in protocol["shift_severities"]
        ],
    ]
    segmentation_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    max_abs_shift = int(protocol["max_abs_shift"])
    ecc_config = protocol.get("ecc", {})

    for scenario_index, spec in enumerate(specs):
        print(f"[{scenario_index + 1}/{len(specs)}] {spec.key}")
        dataset = CorruptedPairDataset(
            base_dataset,
            spec,
            normalization=normalization,
        )
        loader = _loader(
            dataset=dataset,
            batch_size=int(protocol["batch_size"]),
            shuffle=False,
            num_workers=int(protocol["num_workers"]),
            seed=seed + scenario_index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        accumulators = {
            method: BinaryChangeMetrics(
                threshold=float(protocol["threshold"])
            )
            for method in methods
        }
        sample_count = 0
        valid_pixel_count = 0

        for batch in loader:
            t1_cpu = batch["image_t1"]
            t2_cpu = batch["image_t2"]
            true_offset_cpu = torch.stack(
                (
                    torch.as_tensor(batch["shift_dx"]),
                    torch.as_tensor(batch["shift_dy"]),
                ),
                dim=1,
            ).float()
            predictions: dict[str, torch.Tensor] = {
                "oracle": true_offset_cpu.clone()
            }
            details: dict[str, list[RegistrationControlResult]] = {}
            if "phase" in methods or "ecc" in methods:
                phase_results = [
                    phase_correlation_offset(
                        t1_cpu[index],
                        t2_cpu[index],
                        normalization=normalization,
                        max_abs_shift=max_abs_shift,
                    )
                    for index in range(int(t1_cpu.shape[0]))
                ]
                predictions["phase"] = torch.tensor(
                    [[result.dx, result.dy] for result in phase_results],
                    dtype=torch.float32,
                )
                details["phase"] = phase_results
            if "ecc" in methods:
                ecc_results: list[RegistrationControlResult] = []
                ecc_offsets: list[list[float]] = []
                for index, phase in enumerate(details["phase"]):
                    result = ecc_translation_offset(
                        t1_cpu[index],
                        t2_cpu[index],
                        normalization=normalization,
                        initial_offset=(phase.dx, phase.dy),
                        iterations=int(ecc_config.get("iterations", 100)),
                        epsilon=float(ecc_config.get("epsilon", 1e-6)),
                    )
                    outside = (
                        abs(result.dx) > max_abs_shift
                        or abs(result.dy) > max_abs_shift
                    )
                    use_fallback = (
                        (not result.converged or outside)
                        and bool(ecc_config.get("fallback_to_phase", True))
                    )
                    chosen = phase if use_fallback else result
                    ecc_results.append(result)
                    ecc_offsets.append([chosen.dx, chosen.dy])
                predictions["ecc"] = torch.tensor(
                    ecc_offsets,
                    dtype=torch.float32,
                )
                details["ecc"] = ecc_results

            t1 = t1_cpu.to(device, non_blocking=True)
            t2 = t2_cpu.to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = batch.get("valid_mask")
            if isinstance(valid, torch.Tensor):
                valid = valid.to(device, non_blocking=True)
            else:
                valid = torch.ones_like(mask)
            interior = interior_valid_mask(
                batch_size=int(t1.shape[0]),
                height=int(t1.shape[-2]),
                width=int(t1.shape[-1]),
                border=int(protocol["evaluation_border_pixels"]),
                device=device,
            )
            valid = valid * interior
            method_logits: dict[str, torch.Tensor] = {}
            if "raw" in methods:
                method_logits["raw"] = _detector_logits(
                    model,
                    t1,
                    t2,
                    None,
                )
            for method in methods:
                if method == "raw":
                    continue
                offsets = predictions[method].to(
                    device=device,
                    dtype=t2.dtype,
                )
                method_logits[method] = _detector_logits(
                    model,
                    t1,
                    t2,
                    offsets,
                )

            for method, logits in method_logits.items():
                accumulators[method].update(
                    logits,
                    mask,
                    from_logits=True,
                    valid_mask=valid,
                )
            batch_size = int(t1.shape[0])
            sample_count += batch_size
            valid_pixel_count += int(valid.sum())

            for method, predicted in predictions.items():
                errors = torch.linalg.vector_norm(
                    predicted - true_offset_cpu,
                    dim=1,
                )
                for index, sample_id in enumerate(batch["id"]):
                    control = (
                        details.get(method, [])[index]
                        if method in details
                        else RegistrationControlResult(
                            dx=float(predicted[index, 0]),
                            dy=float(predicted[index, 1]),
                            score=1.0,
                        )
                    )
                    chosen_dx = float(predicted[index, 0])
                    chosen_dy = float(predicted[index, 1])
                    used_fallback = (
                        method == "ecc"
                        and (
                            not control.converged
                            or abs(control.dx) > max_abs_shift
                            or abs(control.dy) > max_abs_shift
                        )
                        and bool(
                            ecc_config.get("fallback_to_phase", True)
                        )
                    )
                    offset_rows.append(
                        {
                            "id": str(sample_id),
                            "scenario": spec.key,
                            "method": method,
                            "target_dx": float(true_offset_cpu[index, 0]),
                            "target_dy": float(true_offset_cpu[index, 1]),
                            "predicted_dx": chosen_dx,
                            "predicted_dy": chosen_dy,
                            "epe_pixels": float(errors[index]),
                            "score": float(control.score),
                            "converged": bool(control.converged),
                            "used_fallback": used_fallback,
                        }
                    )

        for method, accumulator in accumulators.items():
            values = accumulator.compute()
            segmentation_rows.append(
                {
                    "scenario": spec.key,
                    "severity": spec.severity,
                    "shift_pixels": spec.value,
                    "method": method,
                    **values,
                    "samples": sample_count,
                    "valid_pixels": valid_pixel_count,
                }
            )
        print(
            " ".join(
                f"{method}={float(accumulators[method].compute()['f1']):.6f}"
                for method in methods
            )
        )

    segmentation_by_key = {
        (str(row["scenario"]), str(row["method"])): row
        for row in segmentation_rows
    }
    for spec in specs:
        raw_f1 = float(segmentation_by_key[(spec.key, "raw")]["f1"])
        oracle_f1 = float(
            segmentation_by_key[(spec.key, "oracle")]["f1"]
        )
        denominator = oracle_f1 - raw_f1
        for method in methods:
            row = segmentation_by_key[(spec.key, method)]
            method_f1 = float(row["f1"])
            row["oracle_gap_recovery"] = (
                (method_f1 - raw_f1) / denominator
                if denominator > 1e-12
                else None
            )

    offset_summaries: list[dict[str, Any]] = []
    for spec in specs:
        for method in ("phase", "ecc", "oracle"):
            if method not in methods:
                continue
            subset = [
                row
                for row in offset_rows
                if row["scenario"] == spec.key and row["method"] == method
            ]
            offset_summaries.append(
                {
                    "scenario": spec.key,
                    "severity": spec.severity,
                    "shift_pixels": spec.value,
                    "method": method,
                    **_offset_summary(subset),
                }
            )

    classical_decision: dict[str, Any] = {}
    clean_raw = float(
        segmentation_by_key[("clean:s0:common", "raw")]["f1"]
    )
    for method in ("phase", "ecc"):
        if method not in methods:
            continue
        clean_f1 = float(
            segmentation_by_key[("clean:s0:common", method)]["f1"]
        )
        recoveries = {}
        for pixels in (4, 8):
            scenario = next(
                spec.key
                for spec in specs
                if spec.name == "shift" and int(spec.value) == pixels
            )
            recovery = segmentation_by_key[(scenario, method)][
                "oracle_gap_recovery"
            ]
            recoveries[str(pixels)] = recovery
        solved = (
            clean_f1 >= clean_raw - 0.001
            and all(
                value is not None and float(value) >= 0.80
                for value in recoveries.values()
            )
        )
        classical_decision[method] = {
            "clean_f1_drop": clean_raw - clean_f1,
            "recovery_by_shift_pixels": recoveries,
            "solves_synthetic_global_translation": solved,
        }

    summary = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": str(protocol_path),
        "protocol_sha256": _protocol_hash(protocol),
        "split": split,
        "coordinate_convention": (
            "reference(x,y) corresponds to target(x+dx,y+dy)"
        ),
        "common_support_border_pixels": int(
            protocol["evaluation_border_pixels"]
        ),
        "classical_decision": classical_decision,
        "segmentation_results": segmentation_rows,
        "offset_summaries": offset_summaries,
    }
    save_json(output_dir / "results.json", summary)
    _write_csv(output_dir / "segmentation_results.csv", segmentation_rows)
    _write_csv(output_dir / "offset_summaries.csv", offset_summaries)
    _write_csv(output_dir / "offsets_per_sample.csv", offset_rows)
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "protocol_sha256": summary["protocol_sha256"],
                "classical_decision": classical_decision,
            },
            indent=2,
        )
    )
    print(f"Saved registration controls to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
