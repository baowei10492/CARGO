from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.registration_controls import (
    ecc_translation_offset,
    phase_correlation_offset,
    sift_translation_offset,
)
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train_evaluate_task_benefit_router import (
    _loader,
    _scenario_specs,
    _valid_mask,
)


CLASSICAL_METHODS = ("phase", "ecc", "sift")
FLOAT_TOLERANCE = 1e-9


def _estimate_classical(
    method: str,
    t1: torch.Tensor,
    t2: torch.Tensor,
    *,
    normalization: str,
    max_abs_shift: float,
    ecc_fallback_to_phase: bool,
) -> tuple[list[tuple[float, float]], list[bool], list[bool]]:
    offsets: list[tuple[float, float]] = []
    converged: list[bool] = []
    used_fallback: list[bool] = []
    for index in range(t1.shape[0]):
        reference = t1[index]
        target = t2[index]
        fallback = False
        if method == "phase":
            result = phase_correlation_offset(
                reference,
                target,
                normalization=normalization,
                max_abs_shift=int(max_abs_shift),
            )
        elif method == "ecc":
            initial = phase_correlation_offset(
                reference,
                target,
                normalization=normalization,
                max_abs_shift=int(max_abs_shift),
            )
            result = ecc_translation_offset(
                reference,
                target,
                normalization=normalization,
                initial_offset=(initial.dx, initial.dy),
            )
            unusable = (
                not result.converged
                or abs(result.dx) > max_abs_shift
                or abs(result.dy) > max_abs_shift
            )
            if unusable and ecc_fallback_to_phase:
                result = initial
                fallback = True
        elif method == "sift":
            result = sift_translation_offset(
                reference,
                target,
                normalization=normalization,
                max_abs_shift=max_abs_shift,
            )
        else:
            raise ValueError(f"Unknown classical method: {method}")
        offsets.append((float(result.dx), float(result.dy)))
        converged.append(bool(result.converged))
        used_fallback.append(fallback)
    return offsets, converged, used_fallback


def baseline_decision(
    rows: list[dict[str, Any]],
    *,
    reference_shift_mean_gain: float,
    reference_clean_drop: float,
    margin: float,
    clean_drop_max: float,
) -> dict[str, Any]:
    """Apply the preregistered classical-baseline comparison.

    The honest question this answers is not "do we beat classical
    registration at recovering a synthetic global translation" -- on a
    dataset whose intrinsic misalignment is ~0, dense correlation
    methods are expected to be very good at exactly that. The question
    is whether a classical estimator, given the same frozen detector and
    the same warp, reaches the learned selective router's operating
    point WITHOUT damaging clean pairs.

    A classical win is an explicitly allowed outcome and is not to be
    hidden: it would mean the localizer is not the contribution on this
    dataset, and the claim must be restated around selective routing,
    the applicability criterion, and the task-versus-photometric
    coordinate evidence.
    """

    by_method: dict[str, dict[str, float]] = {}
    for row in rows:
        key = f"{row['method']}:{row['variant']}"
        entry = by_method.setdefault(key, {})
        entry[str(row["scenario"])] = float(row["f1"])
    raw = {
        str(row["scenario"]): float(row["f1"])
        for row in rows
        if row["method"] == "raw"
    }
    clean_key = "clean:s0:common"
    shift_keys = sorted(key for key in raw if key.startswith("shift:"))
    if clean_key not in raw or not shift_keys:
        raise ValueError("Raw rows must cover clean and shift scenarios")

    summaries: dict[str, dict[str, Any]] = {}
    for key, scores in by_method.items():
        if key.startswith("raw:"):
            continue
        gains = [scores[name] - raw[name] for name in shift_keys]
        summary = {
            "clean_drop": raw[clean_key] - scores[clean_key],
            "shift_mean_gain": sum(gains) / len(gains),
            "per_scenario_gain": {
                name: scores[name] - raw[name] for name in shift_keys
            },
        }
        # FLOAT_TOLERANCE only makes "exactly at the bar" behave as the
        # protocol text says (">= bar"). Summing per-scenario gains puts
        # an exactly-matching method at 0.07709999999999999 against a bar
        # of 0.0771, which would fail on representation noise alone. This
        # does not loosen the threshold: 1e-9 is ten million times
        # smaller than the 0.005 margin it sits inside.
        summary["matches_reference"] = bool(
            summary["shift_mean_gain"]
            >= float(reference_shift_mean_gain)
            - float(margin)
            - FLOAT_TOLERANCE
            and summary["clean_drop"]
            <= float(clean_drop_max) + FLOAT_TOLERANCE
        )
        summaries[key] = summary

    matching = sorted(
        key for key, value in summaries.items() if value["matches_reference"]
    )
    best = max(
        summaries,
        key=lambda key: summaries[key]["shift_mean_gain"],
        default=None,
    )
    return {
        "status": "completed",
        "reference": {
            "shift_mean_gain": float(reference_shift_mean_gain),
            "clean_drop": float(reference_clean_drop),
            "margin": float(margin),
            "clean_drop_max": float(clean_drop_max),
        },
        "raw_f1": raw,
        "method_summaries": summaries,
        "best_classical_or_gated": best,
        "methods_matching_reference": matching,
        "verdict": (
            "classical_sufficient_restate_contribution"
            if matching
            else "learned_selective_router_justified"
        ),
        "note": (
            "Classical methods estimate photometric displacement; the "
            "learned localizer estimates task-coordinate displacement. "
            "CORRECTED 2026-07-26 (main record 86.2): the preregistered "
            "expectation that these coincide on LEVIR was WRONG. LEVIR "
            "pairs carry a real ~3 px photometric displacement, so "
            "classical estimators report injection + intrinsic while "
            "the task-trained localizer reports the injection alone. "
            "'Intrinsic misalignment ~0' was a statement about the "
            "task-trained localizer's behaviour, not about photometric "
            "alignment -- which is exactly the distinction this paper "
            "is about."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare classical registration baselines (phase "
            "correlation, ECC, SIFT+RANSAC) against the learned "
            "localizer under one frozen detector and one warp."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(
        Path(args.protocol).expanduser().read_text(encoding="utf-8")
    )
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
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    config["model"]["registration_only"] = False
    config["model"]["freeze_registration"] = True
    config["model"]["offset_decoding"] = "hard"
    config["model"]["alignment_gate"] = "threshold_nonzero"
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    estimate = getattr(model, "estimate_registration")

    data = config["data"]
    normalization = str(data.get("normalization", "imagenet"))
    threshold = float(protocol["threshold"])
    gate_pixels = float(protocol["gate_pixels"])
    max_abs_shift = float(protocol["maximum_offset_pixels"])
    methods = tuple(str(name) for name in protocol["methods"])
    base_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=str(protocol["validation_split"]),
        image_size=None,
        normalization=normalization,
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode="none",
    )

    variants = ["always", "gated"]
    scenario_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = {name: [] for name in methods}
    timing["ours"] = []

    specs = _scenario_specs(protocol)
    for scenario_index, spec in enumerate(specs):
        print(f"[baselines {scenario_index + 1}/{len(specs)}] {spec.key}")
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
            seed=int(protocol["seed"]) + scenario_index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        keys = ["raw"] + [
            f"{name}:{variant}"
            for name in (*methods, "ours")
            for variant in variants
        ]
        accumulators = {
            key: BinaryChangeMetrics(threshold=threshold) for key in keys
        }
        selected = {key: 0 for key in keys}
        sample_count = 0

        for batch in loader:
            t1_cpu = batch["image_t1"]
            t2_cpu = batch["image_t2"]
            t1 = t1_cpu.to(device, non_blocking=True)
            t2 = t2_cpu.to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = _valid_mask(
                batch,
                mask,
                border=int(protocol["evaluation_border_pixels"]),
                device=device,
            )
            # CorruptedPairDataset records the INJECTED shift under
            # shift_dx / shift_dy. misregistration_dx belongs to the base
            # dataset's own training-time augmentation and is identically
            # zero at evaluation, so reading it silently compares every
            # estimate against zero (main record 85.1).
            truth = torch.stack(
                [
                    batch["shift_dx"].float(),
                    batch["shift_dy"].float(),
                ],
                dim=1,
            )

            estimates: dict[str, torch.Tensor] = {}
            gates: dict[str, torch.Tensor] = {}
            estimator_converged: dict[str, list[bool]] = {}
            estimator_fallback: dict[str, list[bool]] = {}
            for name in methods:
                started = time.perf_counter()
                offsets, converged, used_fallback = _estimate_classical(
                    name,
                    t1_cpu,
                    t2_cpu,
                    normalization=normalization,
                    max_abs_shift=max_abs_shift,
                    ecc_fallback_to_phase=bool(
                        protocol.get("ecc_fallback_to_phase", True)
                    ),
                )
                timing[name].append(
                    (time.perf_counter() - started) / max(1, len(offsets))
                )
                tensor = torch.tensor(offsets, dtype=torch.float32)
                estimates[name] = tensor
                estimator_converged[name] = converged
                estimator_fallback[name] = used_fallback
                gates[name] = (
                    tensor.abs().amax(dim=1) >= gate_pixels
                )

            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                enabled=bool(protocol.get("amp", True))
                and device.type == "cuda",
            ):
                registration = estimate(t1, t2)
            timing["ours"].append(
                (time.perf_counter() - started) / max(1, t1.shape[0])
            )
            estimates["ours"] = registration[
                "conditional_hard_offset_pixels"
            ].float().cpu()
            estimator_converged["ours"] = [True] * int(t1.shape[0])
            estimator_fallback["ours"] = [False] * int(t1.shape[0])
            gates["ours"] = (
                registration["alignment_sample_gate"] >= 0.5
            ).cpu()

            # forward_with_forced_offset returns the frozen detector on
            # the UNWARPED pair (raw) and on the pair warped by exactly
            # the offset we pass. Calling model(t1, t2) instead would run
            # the full P7 forward, which applies the model's OWN
            # registration -- that makes "raw" secretly the q-gated
            # aligned path and double-warps every classical method
            # (main record 85.1).
            raw_logits: torch.Tensor | None = None
            for name in (*methods, "ours"):
                offsets = estimates[name].to(device=device, dtype=t2.dtype)
                for variant in variants:
                    key = f"{name}:{variant}"
                    if variant == "always":
                        chosen = torch.ones(
                            offsets.shape[0],
                            dtype=torch.bool,
                            device=device,
                        )
                    else:
                        chosen = gates[name].to(device)
                    applied = offsets * chosen[:, None].to(offsets)
                    with torch.autocast(
                        device_type=device.type,
                        enabled=bool(protocol.get("amp", True))
                        and device.type == "cuda",
                    ):
                        forced = model.forward_with_forced_offset(
                            t1,
                            t2,
                            applied,
                        )
                    aligned = forced["aligned_change_logits"].float()
                    if raw_logits is None:
                        raw_logits = forced["raw_change_logits"].float()
                        accumulators["raw"].update(
                            raw_logits,
                            mask,
                            from_logits=True,
                            valid_mask=valid,
                        )
                    logits = torch.where(
                        chosen[:, None, None, None],
                        aligned,
                        raw_logits,
                    )
                    accumulators[key].update(
                        logits,
                        mask,
                        from_logits=True,
                        valid_mask=valid,
                    )
                    selected[key] += int(chosen.sum())

            for name in (*methods, "ours"):
                errors = torch.linalg.vector_norm(
                    estimates[name] - truth,
                    dim=1,
                )
                for index, sample_id in enumerate(batch["id"]):
                    offset_rows.append(
                        {
                            "id": str(sample_id),
                            "scenario": spec.key,
                            "method": name,
                            "est_dx": float(estimates[name][index, 0]),
                            "est_dy": float(estimates[name][index, 1]),
                            "gt_dx": float(truth[index, 0]),
                            "gt_dy": float(truth[index, 1]),
                            "error": float(errors[index]),
                            "gate": bool(gates[name][index]),
                            "converged": estimator_converged[name][index],
                            "used_fallback": estimator_fallback[name][index],
                        }
                    )
            sample_count += int(t1.shape[0])

        for key in keys:
            values = accumulators[key].compute()
            method, _, variant = key.partition(":")
            scenario_rows.append(
                {
                    "scenario": spec.key,
                    "method": method,
                    "variant": variant or "raw",
                    **values,
                    "aligned_selection_rate": (
                        selected[key] / max(1, sample_count)
                        if variant
                        else None
                    ),
                    "samples": sample_count,
                }
            )

    with (output_dir / "scenario_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(scenario_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(scenario_rows)
    with (output_dir / "offset_records.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(offset_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(offset_rows)

    offset_summary: dict[str, Any] = {}
    for name in (*methods, "ours"):
        per_scenario: dict[str, Any] = {}
        for spec in specs:
            subset = [
                row
                for row in offset_rows
                if row["method"] == name and row["scenario"] == spec.key
            ]
            if not subset:
                continue
            errors = [row["error"] for row in subset]
            per_scenario[spec.key] = {
                "samples": len(subset),
                "mean_epe": sum(errors) / len(errors),
                "pck1": sum(1 for e in errors if e <= 1.0) / len(errors),
                "pck2": sum(1 for e in errors if e <= 2.0) / len(errors),
                "gate_rate": sum(
                    1 for row in subset if row["gate"]
                )
                / len(subset),
            }
        offset_summary[name] = per_scenario

    # Operational guard for the lesson in main record 85.4: before any
    # new conclusion is read, the tool must reproduce a number that is
    # already known. The first version of this tool silently measured
    # the q-gated path as "raw" and produced a favourable verdict; this
    # check would have caught it immediately.
    expected_raw = protocol.get("expected_raw_f1")
    if expected_raw:
        tolerance = float(protocol.get("expected_raw_f1_tolerance", 1e-4))
        measured = {
            str(row["scenario"]): float(row["f1"])
            for row in scenario_rows
            if row["method"] == "raw"
        }
        drift = {
            key: measured[key] - float(value)
            for key, value in expected_raw.items()
            if key in measured
        }
        missing = [key for key in expected_raw if key not in measured]
        worst = max(
            (abs(value) for value in drift.values()),
            default=0.0,
        )
        if missing or worst > tolerance:
            raise RuntimeError(
                "Raw baseline does not reproduce the recorded P7-3 "
                f"values (worst drift {worst:.6f} > {tolerance}); "
                f"missing={missing}; drift={drift}. The instrument is "
                "measuring something other than the frozen detector on "
                "the unwarped pair -- fix it before reading any "
                "comparison."
            )
        print(f"raw baseline reproduces recorded values (worst {worst:.2e})")

    reference = protocol["reference_operating_point"]
    decision = baseline_decision(
        scenario_rows,
        reference_shift_mean_gain=float(reference["shift_mean_gain"]),
        reference_clean_drop=float(reference["clean_drop"]),
        margin=float(reference["margin"]),
        clean_drop_max=float(reference["clean_drop_max"]),
    )
    decision["offset_accuracy"] = offset_summary
    decision["seconds_per_image"] = {
        name: (sum(values) / len(values) if values else None)
        for name, values in timing.items()
    }
    decision["timing_caveat"] = (
        "Classical estimators run on CPU per image; the learned "
        "localizer runs batched on GPU. These numbers are indicative of "
        "deployment cost as configured, NOT a controlled speed "
        "comparison."
    )
    save_json(output_dir / "decision.json", decision)
    printable = {
        key: value
        for key, value in decision.items()
        if key not in {"offset_accuracy"}
    }
    print("===== CLASSICAL BASELINE COMPARISON =====")
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print("--- offset accuracy (mean EPE / PCK@1) ---")
    for name, table in offset_summary.items():
        cells = " ".join(
            f"{key.split(':')[0][:5]}:{entry['mean_epe']:.2f}/"
            f"{entry['pck1']:.2f}"
            for key, entry in table.items()
        )
        print(f"{name:8s} {cells}")
    print(f"Saved classical baseline comparison to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
