from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.robustness import CorruptedPairDataset
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader
from tools.train_evaluate_task_benefit_router import (
    _scenario_specs,
    _write_csv,
)


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _difference(
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _rate(errors: list[float], threshold: float) -> float:
    if not errors:
        return 0.0
    return sum(error <= threshold for error in errors) / len(errors)


def offset_consistency_summary(
    records: list[dict[str, Any]],
    *,
    clean_scenario: str,
    shift_scenarios: list[str],
    boundary_threshold: float | None = None,
) -> dict[str, Any]:
    """Label-free discrimination of transfer-failure mode.

    Uses only predicted conditional offsets and injected shift ground
    truth. If ``est(shift) - est(clean)`` tracks the injected offset,
    the descriptor/localizer is healthy and the failure is a stable
    per-image intrinsic offset (task-coordinate mismatch). If it does
    not, the descriptor itself is weak on this dataset.
    """

    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_scenario[record["scenario"]][record["id"]] = record
    clean = by_scenario[clean_scenario]
    if not clean:
        raise RuntimeError("No clean records available")

    summary: dict[str, Any] = {"per_shift": {}}
    residuals: dict[str, dict[str, tuple[float, float]]] = {}
    for scenario in shift_scenarios:
        rows = by_scenario[scenario]
        ids = [key for key in rows if key in clean]
        if not ids:
            raise RuntimeError(
                f"No overlapping ids for scenario {scenario}"
            )
        raw_errors: list[float] = []
        delta_errors: list[float] = []
        residual_vs_clean: list[float] = []
        residuals[scenario] = {}
        for key in ids:
            est = (
                float(rows[key]["est_dx"]),
                float(rows[key]["est_dy"]),
            )
            target = (
                float(rows[key]["gt_dx"]),
                float(rows[key]["gt_dy"]),
            )
            est_clean = (
                float(clean[key]["est_dx"]),
                float(clean[key]["est_dy"]),
            )
            raw_errors.append(_norm(_difference(est, target)))
            delta = _difference(est, est_clean)
            delta_errors.append(_norm(_difference(delta, target)))
            residual = _difference(est, target)
            residuals[scenario][key] = residual
            residual_vs_clean.append(
                _norm(_difference(residual, est_clean))
            )
        entry = {
            "samples": len(ids),
            "raw_pck1": _rate(raw_errors, 1.0),
            "raw_pck2": _rate(raw_errors, 2.0),
            "raw_mean_epe": sum(raw_errors) / len(raw_errors),
            # NOTE: delta_* and residual_matches_clean_* are the SAME
            # quantity. delta = ||(est - est_clean) - gt|| and
            # residual_vs_clean = ||(est - gt) - est_clean||, which are
            # algebraically identical. Both are kept for backward
            # compatibility, but they are ONE piece of evidence, not two.
            "delta_pck1": _rate(delta_errors, 1.0),
            "delta_pck2": _rate(delta_errors, 2.0),
            "delta_mean_epe": sum(delta_errors) / len(delta_errors),
            "residual_matches_clean_pck1": _rate(
                residual_vs_clean, 1.0
            ),
            "residual_matches_clean_pck2": _rate(
                residual_vs_clean, 2.0
            ),
        }
        if boundary_threshold is not None:
            def _chebyshev(dx_key: str, dy_key: str) -> list[str]:
                return [
                    key
                    for key in ids
                    if max(
                        abs(float(rows[key][dx_key])),
                        abs(float(rows[key][dy_key])),
                    )
                    >= boundary_threshold
                ]

            estimate_saturated = _chebyshev("est_dx", "est_dy")
            # The ground truth's own boundary rate. Without it the
            # estimate's boundary rate is uninterpretable: at a severity
            # whose injected offset already sits on the search boundary,
            # a PERFECT estimator saturates the statistic (LEVIR Shift-8
            # measures 0.984 while its PCK@1 is 0.922). Only the excess
            # of estimate over truth can indicate clamping, and a large
            # negative excess indicates the opposite failure -- an
            # estimator that never reaches the true magnitude at all.
            truth_saturated = _chebyshev("gt_dx", "gt_dy")
            entry["boundary_fraction"] = (
                len(estimate_saturated) / len(ids)
            )
            entry["gt_boundary_fraction"] = (
                len(truth_saturated) / len(ids)
            )
            entry["excess_boundary_fraction"] = (
                entry["boundary_fraction"]
                - entry["gt_boundary_fraction"]
            )
        summary["per_shift"][scenario] = entry

    pairwise: dict[str, Any] = {}
    for index, first in enumerate(shift_scenarios):
        for second in shift_scenarios[index + 1 :]:
            shared = [
                key
                for key in residuals[first]
                if key in residuals[second]
            ]
            errors = [
                _norm(
                    _difference(
                        residuals[first][key],
                        residuals[second][key],
                    )
                )
                for key in shared
            ]
            pairwise[f"{first}|{second}"] = {
                "samples": len(shared),
                "pck1": _rate(errors, 1.0),
                "pck2": _rate(errors, 2.0),
                "mean_epe": (
                    sum(errors) / len(errors) if errors else 0.0
                ),
            }
    summary["residual_pairwise_consistency"] = pairwise

    clean_magnitudes = [
        _norm((float(row["est_dx"]), float(row["est_dy"])))
        for row in clean.values()
    ]
    clean_magnitudes.sort()
    half = len(clean_magnitudes) // 2
    summary["clean_conditional_offset"] = {
        "samples": len(clean_magnitudes),
        "mean_magnitude": sum(clean_magnitudes)
        / len(clean_magnitudes),
        "median_magnitude": clean_magnitudes[half],
        "fraction_leq_2p1": _rate(clean_magnitudes, 2.1),
        "fraction_leq_4p1": _rate(clean_magnitudes, 4.1),
    }
    return summary


@torch.no_grad()
def collect_offset_records(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    *,
    protocol: dict[str, Any],
    normalization: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    seed = int(protocol["seed"])
    records: list[dict[str, Any]] = []
    specs = _scenario_specs(protocol)
    for scenario_index, spec in enumerate(specs):
        print(f"[offsets {scenario_index + 1}/{len(specs)}] {spec.key}")
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
        for batch in loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(protocol.get("amp", True))
                and device.type == "cuda",
            ):
                registration = model.estimate_registration(t1, t2)
            offsets = registration[
                "conditional_hard_offset_pixels"
            ].float().cpu()
            q_gate = (
                registration["alignment_sample_gate"] >= 0.5
            ).cpu()
            for index, sample_id in enumerate(batch["id"]):
                records.append(
                    {
                        "id": str(sample_id),
                        "scenario": spec.key,
                        "est_dx": float(offsets[index, 0]),
                        "est_dy": float(offsets[index, 1]),
                        "gt_dx": float(batch["shift_dx"][index]),
                        "gt_dy": float(batch["shift_dy"][index]),
                        "q_gate": bool(q_gate[index]),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label-free offset-consistency diagnostic for the Stage C "
            "transfer failure: discriminates task-coordinate intrinsic "
            "misalignment from descriptor collapse. Uses injected "
            "offsets only; never reads change masks."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p7_stage_c_s2looking_transfer.yaml",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Override the protocol's validation_split (used by the "
            "scale probe, whose splits are scale0512/scale1024/...)."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Centre-crop every evaluation image to this size (the "
            "dataset applies a centre crop whenever augment is off). "
            "Use it to match image size across datasets of different "
            "native tile sizes: a global offset estimator has less "
            "support to vote with on a smaller image, so comparing a "
            "512 px dataset against a 1024 px one confounds dataset "
            "properties with measurement configuration."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(
        Path(args.protocol).expanduser().read_text(encoding="utf-8")
    )
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a mapping")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = select_device(args.device)
    seed_everything(int(protocol["seed"]))

    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    config = copy.deepcopy(checkpoint["config"])
    config["model"]["registration_only"] = False
    config["model"]["freeze_registration"] = True
    config["model"]["offset_decoding"] = "hard"
    config["model"]["alignment_gate"] = "threshold_nonzero"
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    data = config["data"]
    normalization = str(data.get("normalization", "imagenet"))
    dataset = PairedChangeDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split=str(args.split or protocol["validation_split"]),
        image_size=(
            int(args.image_size)
            if args.image_size is not None
            else None
        ),
        normalization=normalization,
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode="none",
    )
    records = collect_offset_records(
        model,
        dataset,
        protocol=protocol,
        normalization=normalization,
        device=device,
    )
    _write_csv(output_dir / "offset_records.csv", records)

    specs = _scenario_specs(protocol)
    clean_scenario = next(
        spec.key for spec in specs if spec.name == "clean"
    )
    shift_scenarios = [
        spec.key for spec in specs if spec.name == "shift"
    ]
    summary = offset_consistency_summary(
        records,
        clean_scenario=clean_scenario,
        shift_scenarios=shift_scenarios,
        boundary_threshold=(
            float(protocol["maximum_offset_pixels"]) - 0.1
        ),
    )
    q_rates = {}
    for spec in specs:
        rows = [
            record
            for record in records
            if record["scenario"] == spec.key
        ]
        q_rates[spec.key] = sum(
            record["q_gate"] for record in rows
        ) / max(1, len(rows))
    summary["q_trigger_rate"] = q_rates
    # Provenance: an offset summary is only comparable to another one
    # measured at the same image size (see --image-size).
    summary["evaluation_image_size"] = (
        int(args.image_size) if args.image_size is not None else "native"
    )
    summary["dataset_root"] = str(args.dataset_root)
    save_json(output_dir / "offset_consistency_summary.json", summary)
    print("===== OFFSET CONSISTENCY SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved offset diagnostic to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
