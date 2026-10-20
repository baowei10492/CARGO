from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.registration_controls import phase_correlation_offset
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.evaluate_safe_alignment_transfer import (
    load_safe_classifier_bundle,
)
from tools.train_evaluate_safe_alignment_classifier import (
    _classifier_probability,
)
from tools.train_evaluate_task_benefit_router import (
    _forward_experts,
    _loader,
    _valid_mask,
)


def perturbation_specs(protocol: dict[str, Any]) -> list[CorruptionSpec]:
    seed = int(protocol["seed"])
    specs = [CorruptionSpec(name="clean", severity=0, seed=seed)]
    for entry in protocol["perturbations"]:
        name = str(entry["name"])
        mode = str(entry.get("mode", "t2"))
        for severity in entry["severities"]:
            specs.append(
                CorruptionSpec(
                    name=name,
                    severity=int(severity),
                    mode=mode,
                    seed=seed,
                )
            )
    return specs


def safety_decision(
    rows: list[dict[str, Any]],
    *,
    primary_method: str,
    harm_tolerance: float,
    clean_key: str = "clean:s0:common",
) -> dict[str, Any]:
    """Apply the predeclared no-harm gate on out-of-scope perturbations.

    Rotation cannot be corrected by any single translation, so it lies
    outside the stated applicability condition. The claim under test is
    NOT that the method fixes it -- it structurally cannot -- but that
    the risk-controlled router recognises it cannot help and therefore
    does no net harm. Sub-pixel translations test the other edge: they
    are below the 2 px decoding grid, so the router should likewise
    decline rather than over-correct.

    A failure here is a real limitation and is reported as one; it would
    not contradict the applicability condition, it would sharpen it.
    """

    by_key = {
        (str(row["scenario"]), str(row["method"])): float(row["f1"])
        for row in rows
    }
    scenarios = sorted(
        {str(row["scenario"]) for row in rows if row["scenario"] != clean_key}
    )
    families: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        family = scenario.split(":")[0]
        raw = by_key[(scenario, "raw")]
        primary = by_key[(scenario, primary_method)]
        entry = {
            "scenario": scenario,
            "raw_f1": raw,
            "primary_f1": primary,
            "delta": primary - raw,
            "no_harm": bool(primary - raw >= -float(harm_tolerance)),
        }
        families.setdefault(family, []).append(entry)

    family_results = {
        family: {
            "entries": entries,
            "worst_delta": min(item["delta"] for item in entries),
            "all_no_harm": all(item["no_harm"] for item in entries),
            "best_delta": max(item["delta"] for item in entries),
        }
        for family, entries in families.items()
    }
    clean_delta = (
        by_key[(clean_key, primary_method)] - by_key[(clean_key, "raw")]
    )
    passed = all(
        result["all_no_harm"] for result in family_results.values()
    ) and clean_delta >= -float(harm_tolerance)
    return {
        "status": "completed",
        "primary_method": primary_method,
        "harm_tolerance": float(harm_tolerance),
        "clean_delta": clean_delta,
        "families": family_results,
        "no_harm_gate_passed": passed,
        "verdict": (
            "router_degrades_gracefully_outside_scope"
            if passed
            else "router_does_net_harm_documented_limitation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the deployed router on perturbations it cannot "
            "represent: sub-pixel translations (below the 2 px decoding "
            "grid) and rotations (not a translation at all)."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registration-classifier", required=True)
    parser.add_argument("--task-classifier", required=True)
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

    classifiers = {
        "safe_veto_registration": load_safe_classifier_bundle(
            Path(args.registration_classifier).expanduser().resolve()
        ),
        "safe_veto_task": load_safe_classifier_bundle(
            Path(args.task_classifier).expanduser().resolve()
        ),
    }
    for name, bundle in classifiers.items():
        print(
            f"{name}: tau={float(bundle['probability_threshold']):.6f} "
            f"features={len(bundle['feature_names'])}"
        )

    data = config["data"]
    threshold = float(protocol["threshold"])
    max_offset = float(protocol["maximum_offset_pixels"])
    use_phase = bool(protocol.get("include_phase_baseline", True))
    base_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=str(protocol["validation_split"]),
        image_size=None,
        normalization=str(data.get("normalization", "imagenet")),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode="none",
    )

    methods = [
        "raw",
        "always_aligned",
        "q_gate",
        "safe_veto_registration",
        "safe_veto_task",
    ]
    if use_phase:
        methods += ["phase_subpixel_always", "phase_subpixel_gated"]

    scenario_rows: list[dict[str, Any]] = []
    specs = perturbation_specs(protocol)
    for index, spec in enumerate(specs):
        print(f"[perturbation {index + 1}/{len(specs)}] {spec.key}")
        dataset = CorruptedPairDataset(
            base_dataset,
            spec,
            normalization=str(data.get("normalization", "imagenet")),
        )
        loader = _loader(
            dataset=dataset,
            batch_size=int(protocol["batch_size"]),
            shuffle=False,
            num_workers=int(protocol["num_workers"]),
            seed=int(protocol["seed"]) + index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        accumulators = {
            name: BinaryChangeMetrics(threshold=threshold)
            for name in methods
        }
        selected = {name: 0 for name in methods}
        sample_count = 0

        for batch in loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            valid = _valid_mask(
                batch,
                mask,
                border=int(protocol["evaluation_border_pixels"]),
                device=device,
            )
            registration, raw_logits, aligned_logits, features = (
                _forward_experts(
                    model,
                    t1,
                    t2,
                    amp=bool(protocol.get("amp", True)),
                    maximum_offset_pixels=max_offset,
                    threshold=threshold,
                    valid=valid,
                )
            )
            q_gate = registration["alignment_sample_gate"] >= 0.5
            gates = {
                "always_aligned": torch.ones_like(q_gate),
                "q_gate": q_gate,
            }
            for name in (
                "safe_veto_registration",
                "safe_veto_task",
            ):
                bundle = classifiers[name]
                probability = _classifier_probability(bundle, features)
                gates[name] = q_gate & (
                    probability
                    >= float(bundle["probability_threshold"])
                )

            logits_by_method = {"raw": raw_logits}
            for name, gate in gates.items():
                logits_by_method[name] = torch.where(
                    gate[:, None, None, None],
                    aligned_logits,
                    raw_logits,
                )

            if use_phase:
                offsets = []
                for sample_index in range(t1.shape[0]):
                    result = phase_correlation_offset(
                        batch["image_t1"][sample_index],
                        batch["image_t2"][sample_index],
                        normalization=str(
                            data.get("normalization", "imagenet")
                        ),
                        max_abs_shift=int(max_offset),
                        subpixel=True,
                    )
                    offsets.append((result.dx, result.dy))
                offset_tensor = torch.tensor(
                    offsets,
                    dtype=torch.float32,
                    device=device,
                )
                phase_gate = offset_tensor.abs().amax(dim=1) >= float(
                    protocol["gate_pixels"]
                )
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(protocol.get("amp", True))
                    and device.type == "cuda",
                ):
                    forced = model.forward_with_forced_offset(
                        t1,
                        t2,
                        offset_tensor.to(t2.dtype),
                    )
                phase_aligned = forced["aligned_change_logits"].float()
                logits_by_method["phase_subpixel_always"] = phase_aligned
                logits_by_method["phase_subpixel_gated"] = torch.where(
                    phase_gate[:, None, None, None],
                    phase_aligned,
                    raw_logits,
                )
                gates["phase_subpixel_always"] = torch.ones_like(
                    phase_gate
                )
                gates["phase_subpixel_gated"] = phase_gate

            for name, logits in logits_by_method.items():
                accumulators[name].update(
                    logits,
                    mask,
                    from_logits=True,
                    valid_mask=valid,
                )
                if name in gates:
                    selected[name] += int(gates[name].sum())
            sample_count += int(t1.shape[0])

        for name in methods:
            values = accumulators[name].compute()
            scenario_rows.append(
                {
                    "scenario": spec.key,
                    "corruption": spec.name,
                    "severity": spec.severity,
                    "value": spec.value,
                    "method": name,
                    **values,
                    "aligned_selection_rate": (
                        selected[name] / max(1, sample_count)
                        if name != "raw"
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

    decision = safety_decision(
        scenario_rows,
        primary_method=str(protocol["primary_method"]),
        harm_tolerance=float(protocol["harm_tolerance"]),
    )
    decision["selection_rates"] = {
        f"{row['scenario']}:{row['method']}": row["aligned_selection_rate"]
        for row in scenario_rows
        if row["aligned_selection_rate"] is not None
    }
    save_json(output_dir / "decision.json", decision)
    printable = {
        key: value
        for key, value in decision.items()
        if key != "selection_rates"
    }
    print("===== PERTURBATION ROBUSTNESS =====")
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print("--- F1 by scenario and method ---")
    scenarios = []
    for row in scenario_rows:
        if row["scenario"] not in scenarios:
            scenarios.append(row["scenario"])
    for scenario in scenarios:
        cells = " ".join(
            f"{row['method'][:14]}={row['f1']:.4f}"
            for row in scenario_rows
            if row["scenario"] == scenario
        )
        print(f"{scenario:26s} {cells}")
    print(f"Saved perturbation robustness to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
