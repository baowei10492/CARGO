from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.dual_path import (
    SafeAlignmentClassifier,
    select_aligned_by_task_loss,
)
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.robustness import CorruptedPairDataset
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader
from tools.train_evaluate_task_benefit_router import (
    _forward_experts,
    _scenario_specs,
    _valid_mask,
    _write_csv,
)
from tools.train_evaluate_safe_alignment_classifier import (
    _classifier_probability,
    _sha256_of_file,
)


DETECTOR_PREFIXES = ("backbone.", "decoder.")


def merge_detector_state(
    task_state: dict[str, torch.Tensor],
    detector_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Overwrite the frozen expert (backbone/decoder) with another detector.

    Every detector-prefixed key of the task-aware state must be replaced
    exactly once from the plain detector state; registration.* keys are
    kept untouched. Any mismatch raises instead of silently mixing
    experts from two datasets.
    """

    expected = {
        key for key in task_state if key.startswith(DETECTOR_PREFIXES)
    }
    provided = {
        key
        for key in detector_state
        if key.startswith(DETECTOR_PREFIXES)
    }
    if expected != provided:
        missing = sorted(expected - provided)[:5]
        extra = sorted(provided - expected)[:5]
        raise RuntimeError(
            "Detector state does not match task-aware detector keys "
            f"(missing={missing}, extra={extra})"
        )
    merged = dict(task_state)
    for key in expected:
        merged[key] = detector_state[key]
    return merged, len(expected)


def load_safe_classifier_bundle(
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hidden_dims = tuple(int(x) for x in checkpoint["hidden_dims"])
    model = SafeAlignmentClassifier(
        input_dim=len(checkpoint["feature_names"]),
        hidden_dims=(hidden_dims[0], hidden_dims[1]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {**checkpoint, "model": model}


def conditional_pck_rates(
    predicted_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
) -> dict[str, float]:
    error = torch.linalg.vector_norm(
        predicted_offsets.float() - target_offsets.float(),
        dim=1,
    )
    total = max(1, int(error.numel()))
    return {
        "mean_epe_pixels": float(error.mean()) if error.numel() else 0.0,
        "pck1": float((error <= 1.0).sum()) / total,
        "pck2": float((error <= 2.0).sum()) / total,
        "pck4": float((error <= 4.0).sum()) / total,
    }


def unlabeled_gate_decision(
    diagnostics: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    specs = _scenario_specs(protocol)
    shift8_key = next(
        spec.key
        for spec in specs
        if spec.name == "shift" and int(spec.value) == 8
    )
    by_key = {str(row["scenario"]): row for row in diagnostics}
    value = float(by_key[shift8_key]["pck1"])
    minimum = float(
        protocol["unlabeled_gate"]["shift8_conditional_pck1_min"]
    )
    return {
        "shift8_conditional_pck1": {
            "value": value,
            "minimum": minimum,
            "passed": value >= minimum,
        },
    }


@torch.no_grad()
def run_unlabeled_diagnostics(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Registration-only pass: PCK against injected offsets + q rates.

    Uses only synthetic shift ground truth; never touches change masks.
    """

    seed = int(protocol["seed"])
    data = config["data"]
    rows: list[dict[str, Any]] = []
    specs = _scenario_specs(protocol)
    for scenario_index, spec in enumerate(specs):
        print(
            f"[unlabeled {scenario_index + 1}/{len(specs)}] {spec.key}"
        )
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
            seed=seed + scenario_index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        predicted: list[torch.Tensor] = []
        target: list[torch.Tensor] = []
        q_count = 0
        sample_count = 0
        for batch in loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(protocol.get("amp", True))
                and device.type == "cuda",
            ):
                registration = model.estimate_registration(t1, t2)
            predicted.append(
                registration["conditional_hard_offset_pixels"]
                .float()
                .cpu()
            )
            target.append(
                torch.stack(
                    (
                        torch.as_tensor(batch["shift_dx"]),
                        torch.as_tensor(batch["shift_dy"]),
                    ),
                    dim=1,
                ).float()
            )
            q_count += int(
                (registration["alignment_sample_gate"] >= 0.5).sum()
            )
            sample_count += int(t1.shape[0])
        rates = conditional_pck_rates(
            torch.cat(predicted, dim=0),
            torch.cat(target, dim=0),
        )
        rows.append(
            {
                "scenario": spec.key,
                "shift_pixels": spec.value,
                "samples": sample_count,
                "q_trigger_rate": q_count / max(1, sample_count),
                **rates,
            }
        )
        print(
            f"  pck1={rates['pck1']:.4f} pck2={rates['pck2']:.4f} "
            f"epe={rates['mean_epe_pixels']:.3f} "
            f"q_rate={q_count / max(1, sample_count):.4f}"
        )
    return rows


@torch.no_grad()
def evaluate_transfer_validation(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    classifiers: dict[str, dict[str, Any]],
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(protocol["seed"])
    data = config["data"]
    threshold = float(protocol["threshold"])
    loss_config = config["loss"]
    bce_weight = float(loss_config.get("bce_weight", 1.0))
    dice_weight = float(loss_config.get("dice_weight", 1.0))
    configured_pos_weight = loss_config.get("pos_weight")
    pos_weight = (
        float(configured_pos_weight)
        if configured_pos_weight is not None
        else None
    )
    variants = (
        "raw",
        "always_aligned",
        "q_gate",
        "safe_veto_registration",
        "safe_veto_task",
        "loss_oracle",
    )
    scenario_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    specs = _scenario_specs(protocol)
    for scenario_index, spec in enumerate(specs):
        print(
            f"[transfer val {scenario_index + 1}/{len(specs)}] {spec.key}"
        )
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
            seed=seed + scenario_index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        accumulators = {
            name: BinaryChangeMetrics(threshold=threshold)
            for name in variants
        }
        loss_sums = {name: 0.0 for name in variants}
        selection_counts = {name: 0 for name in variants}
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
                    maximum_offset_pixels=float(
                        protocol["maximum_offset_pixels"]
                    ),
                    threshold=threshold,
                    valid=valid,
                )
            )
            oracle_logits, oracle_gate, raw_loss, aligned_loss = (
                select_aligned_by_task_loss(
                    raw_logits,
                    aligned_logits,
                    mask,
                    valid,
                    bce_weight=bce_weight,
                    dice_weight=dice_weight,
                    pos_weight=pos_weight,
                )
            )
            q_gate = registration["alignment_sample_gate"] >= 0.5
            probabilities: dict[str, torch.Tensor] = {}
            gates: dict[str, torch.Tensor] = {
                "q_gate": q_gate,
                "loss_oracle": oracle_gate,
            }
            for variant_name, bundle_name in (
                (
                    "safe_veto_registration",
                    "safe_alignment_registration",
                ),
                ("safe_veto_task", "safe_alignment_task"),
            ):
                bundle = classifiers[bundle_name]
                probability = _classifier_probability(bundle, features)
                probabilities[variant_name] = probability
                gates[variant_name] = q_gate & (
                    probability
                    >= float(bundle["probability_threshold"])
                )
            logits_by_variant = {
                "raw": raw_logits,
                "always_aligned": aligned_logits,
                **{
                    name: torch.where(
                        gate[:, None, None, None],
                        aligned_logits,
                        raw_logits,
                    )
                    for name, gate in gates.items()
                },
            }
            loss_by_variant = {
                "raw": raw_loss,
                "always_aligned": aligned_loss,
                **{
                    name: torch.where(gate, aligned_loss, raw_loss)
                    for name, gate in gates.items()
                },
            }
            for name in variants:
                accumulators[name].update(
                    logits_by_variant[name],
                    mask,
                    from_logits=True,
                    valid_mask=valid,
                )
                loss_sums[name] += float(loss_by_variant[name].sum())
                if name in gates:
                    selection_counts[name] += int(gates[name].sum())
            target = torch.stack(
                (
                    torch.as_tensor(batch["shift_dx"]),
                    torch.as_tensor(batch["shift_dy"]),
                ),
                dim=1,
            ).to(features)
            conditional_offset = registration[
                "conditional_hard_offset_pixels"
            ]
            epe = torch.linalg.vector_norm(
                conditional_offset - target,
                dim=1,
            )
            sample_count += int(t1.shape[0])
            for index, sample_id in enumerate(batch["id"]):
                sample_rows.append(
                    {
                        "id": str(sample_id),
                        "scenario": spec.key,
                        "raw_task_loss": float(raw_loss[index]),
                        "aligned_task_loss": float(
                            aligned_loss[index]
                        ),
                        "aligned_loss_advantage": float(
                            raw_loss[index] - aligned_loss[index]
                        ),
                        "conditional_epe_pixels": float(epe[index]),
                        "q_gate": bool(q_gate[index]),
                        "safe_registration_probability": float(
                            probabilities["safe_veto_registration"][
                                index
                            ]
                        ),
                        "safe_veto_registration_gate": bool(
                            gates["safe_veto_registration"][index]
                        ),
                        "safe_task_probability": float(
                            probabilities["safe_veto_task"][index]
                        ),
                        "safe_veto_task_gate": bool(
                            gates["safe_veto_task"][index]
                        ),
                        "oracle_gate": bool(oracle_gate[index]),
                    }
                )
        for name in variants:
            values = accumulators[name].compute()
            scenario_rows.append(
                {
                    "scenario": spec.key,
                    "corruption": spec.name,
                    "severity": spec.severity,
                    "shift_pixels": spec.value,
                    "method": name,
                    **values,
                    "mean_task_loss": (
                        loss_sums[name] / max(1, sample_count)
                    ),
                    "samples": sample_count,
                    "aligned_selection_rate": (
                        selection_counts[name] / max(1, sample_count)
                        if name in gates
                        else None
                    ),
                }
            )
        values_by_method = {
            row["method"]: row
            for row in scenario_rows
            if row["scenario"] == spec.key
        }
        print(
            " ".join(
                f"{name}={float(values_by_method[name]['f1']):.6f}"
                for name in variants
            )
        )
    _write_csv(
        output_dir / "transfer_scenario_metrics.csv",
        scenario_rows,
    )
    _write_csv(
        output_dir / "transfer_sample_decisions.csv",
        sample_rows,
    )
    return scenario_rows, sample_rows


def transfer_decision(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    primary = str(
        protocol.get("primary_method", "safe_veto_registration")
    )
    by_key = {
        (str(row["scenario"]), str(row["method"])): row for row in rows
    }
    specs = _scenario_specs(protocol)
    clean_key = "clean:s0:common"
    shift_key = {
        int(spec.value): spec.key
        for spec in specs
        if spec.name == "shift"
    }

    def f1(scenario: str, method: str) -> float:
        return float(by_key[(scenario, method)]["f1"])

    clean_drop = f1(clean_key, "raw") - f1(clean_key, primary)
    shift2_gain = f1(shift_key[2], primary) - f1(shift_key[2], "raw")
    shift4_oracle_gain = f1(shift_key[4], "loss_oracle") - f1(
        shift_key[4], "raw"
    )
    shift8_oracle_gain = f1(shift_key[8], "loss_oracle") - f1(
        shift_key[8], "raw"
    )
    shift4_gain = f1(shift_key[4], primary) - f1(shift_key[4], "raw")
    shift8_gain = f1(shift_key[8], primary) - f1(shift_key[8], "raw")
    shift4_recovery = (
        shift4_gain / shift4_oracle_gain
        if shift4_oracle_gain > 1e-12
        else 0.0
    )
    shift8_recovery = (
        shift8_gain / shift8_oracle_gain
        if shift8_oracle_gain > 1e-12
        else 0.0
    )
    gate_config = protocol["gates"]
    gates = {
        "clean_primary_f1_drop": {
            "value": clean_drop,
            "maximum": float(gate_config["clean_f1_drop_max"]),
        },
        "shift8_transferred_oracle_gain": {
            "value": shift8_oracle_gain,
            "minimum": float(gate_config["shift8_oracle_gain_min"]),
        },
        "shift8_oracle_gain_recovery": {
            "value": shift8_recovery,
            "minimum": float(
                gate_config["shift8_oracle_recovery_min"]
            ),
        },
        "shift4_oracle_gain_recovery": {
            "value": shift4_recovery,
            "minimum": float(
                gate_config["shift4_oracle_recovery_min"]
            ),
        },
        "shift2_primary_f1_gain": {
            "value": shift2_gain,
            "minimum": float(gate_config["shift2_f1_gain_min"]),
        },
    }
    for gate in gates.values():
        value = float(gate["value"])
        if "maximum" in gate:
            gate["passed"] = value <= float(gate["maximum"])
        else:
            gate["passed"] = value >= float(gate["minimum"])
    passed = all(bool(gate["passed"]) for gate in gates.values())
    selection_rates = {
        key: (
            float(by_key[(key, primary)]["aligned_selection_rate"])
            if by_key[(key, primary)]["aligned_selection_rate"]
            is not None
            else None
        )
        for key in [clean_key, *shift_key.values()]
    }
    return {
        "status": "passed" if passed else "failed",
        "primary_method": primary,
        "transfer_without_target_dataset_training": True,
        "validation_used_once_for_fixed_protocol": True,
        "test_split_untouched": True,
        "gates": gates,
        "aligned_selection_rate_reported_not_gated": selection_rates,
        "next_action": (
            "final_freeze_then_efficiency_and_paper_baselines"
            if passed
            else "diagnose_transfer_gap_then_dataset_specific_or_p7_5"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage C: transfer the frozen LEVIR-trained registration "
            "stack and safe-alignment veto classifiers to S2Looking "
            "with the S2Looking P3b detector as the frozen experts. "
            "Runs a label-free localization gate first; the labeled "
            "validation protocol executes once only if that gate "
            "passes."
        )
    )
    parser.add_argument("--p7-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--classifier-registration", required=True)
    parser.add_argument("--classifier-task", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p7_stage_c_s2looking_transfer.yaml",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    p7_path = Path(args.p7_checkpoint).expanduser().resolve()
    detector_path = Path(
        args.detector_checkpoint
    ).expanduser().resolve()
    classifier_paths = {
        "safe_alignment_registration": Path(
            args.classifier_registration
        ).expanduser().resolve(),
        "safe_alignment_task": Path(
            args.classifier_task
        ).expanduser().resolve(),
    }
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Stage C protocol must be a mapping")
    seed_everything(int(protocol["seed"]))
    device = select_device(args.device)

    sources = {
        "p7_checkpoint": str(p7_path),
        "p7_checkpoint_sha256": _sha256_of_file(p7_path),
        "detector_checkpoint": str(detector_path),
        "detector_checkpoint_sha256": _sha256_of_file(detector_path),
        **{
            f"{name}_checkpoint": str(path)
            for name, path in classifier_paths.items()
        },
        **{
            f"{name}_sha256": _sha256_of_file(path)
            for name, path in classifier_paths.items()
        },
    }
    for key, value in sources.items():
        print(f"{key}={value}")

    p7_checkpoint = torch.load(
        p7_path,
        map_location="cpu",
        weights_only=False,
    )
    detector_checkpoint = torch.load(
        detector_path,
        map_location="cpu",
        weights_only=False,
    )
    config = copy.deepcopy(detector_checkpoint["config"])
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    model_config = copy.deepcopy(p7_checkpoint["config"]["model"])
    model_config["registration_only"] = False
    model_config["freeze_registration"] = True
    model_config["offset_decoding"] = "hard"
    model_config["alignment_gate"] = "threshold_nonzero"
    config["model"] = model_config

    model = build_model(config)
    merged_state, replaced = merge_detector_state(
        p7_checkpoint["model"],
        detector_checkpoint["model"],
    )
    model.load_state_dict(merged_state, strict=True)
    model = model.to(device)
    model.eval()
    print(f"hybrid_detector_parameters_replaced={replaced}")

    classifiers = {
        name: load_safe_classifier_bundle(path)
        for name, path in classifier_paths.items()
    }
    for name, bundle in classifiers.items():
        print(
            f"{name}: tau={float(bundle['probability_threshold']):.6f} "
            f"features={len(bundle['feature_names'])}"
        )

    validation_dataset = PairedChangeDataset(
        dataset_root=config["data"]["dataset_root"],
        manifest_path=config["data"]["manifest"],
        split=str(protocol["validation_split"]),
        image_size=None,
        normalization=config["data"].get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(config["data"].get("mask_threshold", 127)),
        cache_mode=str(config["data"].get("cache_mode", "none")),
    )

    diagnostics = run_unlabeled_diagnostics(
        model,
        validation_dataset,
        protocol=protocol,
        config=config,
        device=device,
    )
    _write_csv(output_dir / "unlabeled_diagnostics.csv", diagnostics)
    unlabeled_gates = unlabeled_gate_decision(
        diagnostics,
        protocol=protocol,
    )
    save_json(
        output_dir / "unlabeled_gate.json",
        {"gates": unlabeled_gates, "diagnostics": diagnostics},
    )
    unlabeled_passed = all(
        bool(gate["passed"]) for gate in unlabeled_gates.values()
    )
    if not unlabeled_passed:
        decision = {
            "status": "stopped_at_unlabeled_gate",
            "primary_method": str(
                protocol.get(
                    "primary_method",
                    "safe_veto_registration",
                )
            ),
            "unlabeled_gates": unlabeled_gates,
            "labeled_validation_executed": False,
            "test_split_untouched": True,
            "next_action": (
                "dataset_specific_localizer_branch_new_protocol"
            ),
        }
        results = {
            "status": "completed",
            **sources,
            "protocol": str(protocol_path),
            "decision": decision,
            "unlabeled_diagnostics": diagnostics,
        }
        save_json(output_dir / "results.json", results)
        save_json(output_dir / "decision.json", decision)
        (output_dir / "protocol_resolved.yaml").write_text(
            yaml.safe_dump(
                protocol,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        print("===== P7 STAGE C DECISION =====")
        print(json.dumps(decision, indent=2, ensure_ascii=False))
        print(
            "Unlabeled localization gate failed; labeled validation "
            "was not executed."
        )
        return

    validation_rows, _ = evaluate_transfer_validation(
        model,
        validation_dataset,
        classifiers,
        protocol=protocol,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    decision = transfer_decision(validation_rows, protocol=protocol)
    decision["unlabeled_gates"] = unlabeled_gates
    decision["labeled_validation_executed"] = True
    results = {
        "status": "completed",
        **sources,
        "protocol": str(protocol_path),
        "decision": decision,
        "unlabeled_diagnostics": diagnostics,
        "validation_results": validation_rows,
    }
    save_json(output_dir / "results.json", results)
    save_json(output_dir / "decision.json", decision)
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("===== P7 STAGE C DECISION =====")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Saved Stage C transfer experiment to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
