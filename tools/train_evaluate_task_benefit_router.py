from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from rs_cd.datasets import PairedChangeDataset
from rs_cd.dual_path import (
    REGISTRATION_ROUTER_FEATURE_NAMES,
    ROUTER_FEATURE_NAMES,
    TaskBenefitRouter,
    dual_path_router_features,
    per_sample_binary_change_loss,
    select_aligned_by_task_loss,
)
from rs_cd.geometry import interior_valid_mask
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


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


def _stable_calibration_membership(
    sample_id: str,
    *,
    seed: int,
    fraction: float,
) -> bool:
    digest = hashlib.sha256(
        f"{seed}|router-calibration|{sample_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64)
    return value < fraction


def _scenario_specs(
    protocol: dict[str, Any],
) -> list[CorruptionSpec]:
    seed = int(protocol["seed"])
    return [
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


def _valid_mask(
    batch: dict[str, Any],
    mask: torch.Tensor,
    *,
    border: int,
    device: torch.device,
) -> torch.Tensor:
    valid = batch.get("valid_mask")
    if isinstance(valid, torch.Tensor):
        valid = valid.to(device, non_blocking=True)
    else:
        valid = torch.ones_like(mask)
    return valid * interior_valid_mask(
        batch_size=int(mask.shape[0]),
        height=int(mask.shape[-2]),
        width=int(mask.shape[-1]),
        border=border,
        device=device,
    )


def _forward_experts(
    model: torch.nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    *,
    amp: bool,
    maximum_offset_pixels: float,
    threshold: float,
    valid: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    estimate = getattr(model, "estimate_registration", None)
    forced_forward = getattr(model, "forward_with_forced_offset", None)
    if not callable(estimate) or not callable(forced_forward):
        raise TypeError("Model does not expose P7 dual-path inference")
    with torch.autocast(
        device_type=t1.device.type,
        enabled=amp and t1.device.type == "cuda",
    ):
        registration = estimate(t1, t2)
        conditional_offset = registration[
            "conditional_hard_offset_pixels"
        ]
        forced = forced_forward(t1, t2, conditional_offset)
    raw_logits = forced["raw_change_logits"].float()
    aligned_logits = forced["aligned_change_logits"].float()
    features = dual_path_router_features(
        raw_logits,
        aligned_logits,
        valid,
        registration,
        maximum_offset_pixels=maximum_offset_pixels,
        threshold=threshold,
    )
    return registration, raw_logits, aligned_logits, features


@torch.no_grad()
def _collect_training_features(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    seed = int(protocol["seed"])
    data = config["data"]
    loss_config = config["loss"]
    bce_weight = float(loss_config.get("bce_weight", 1.0))
    dice_weight = float(loss_config.get("dice_weight", 1.0))
    configured_pos_weight = loss_config.get("pos_weight")
    pos_weight = (
        float(configured_pos_weight)
        if configured_pos_weight is not None
        else None
    )
    calibration_fraction = float(
        protocol["partition"]["calibration_fraction"]
    )
    feature_batches: list[torch.Tensor] = []
    benefit_batches: list[torch.Tensor] = []
    clean_batches: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []

    specs = _scenario_specs(protocol)
    for scenario_index, spec in enumerate(specs):
        print(
            f"[train features {scenario_index + 1}/{len(specs)}] "
            f"{spec.key}"
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
                    threshold=float(protocol["threshold"]),
                    valid=valid,
                )
            )
            raw_loss = per_sample_binary_change_loss(
                raw_logits,
                mask,
                valid,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
            )
            aligned_loss = per_sample_binary_change_loss(
                aligned_logits,
                mask,
                valid,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
            )
            benefit = raw_loss - aligned_loss
            feature_batches.append(features.cpu())
            benefit_batches.append(benefit.cpu())
            clean_batches.append(
                torch.full(
                    (int(t1.shape[0]),),
                    spec.name == "clean",
                    dtype=torch.bool,
                )
            )
            for index, sample_id in enumerate(batch["id"]):
                is_calibration = _stable_calibration_membership(
                    str(sample_id),
                    seed=seed,
                    fraction=calibration_fraction,
                )
                row: dict[str, Any] = {
                    "id": str(sample_id),
                    "scenario": spec.key,
                    "partition": (
                        "calibration" if is_calibration else "fit"
                    ),
                    "raw_task_loss": float(raw_loss[index]),
                    "aligned_task_loss": float(aligned_loss[index]),
                    "aligned_loss_advantage": float(benefit[index]),
                    "aligned_is_beneficial": bool(benefit[index] > 0),
                    "existing_q_selected_aligned": bool(
                        registration["alignment_sample_gate"][index] >= 0.5
                    ),
                }
                for feature_index, name in enumerate(ROUTER_FEATURE_NAMES):
                    row[name] = float(features[index, feature_index])
                rows.append(row)

    features = torch.cat(feature_batches, dim=0)
    benefit = torch.cat(benefit_batches, dim=0)
    clean = torch.cat(clean_batches, dim=0)
    if len(rows) != int(features.shape[0]):
        raise RuntimeError("Training feature metadata is inconsistent")
    return features, benefit, clean, rows


def _train_router(
    name: str,
    features: torch.Tensor,
    benefit: torch.Tensor,
    clean: torch.Tensor,
    metadata: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    *,
    protocol: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    seed = int(protocol["seed"])
    seed_everything(seed)
    all_feature_indices = {
        name: index for index, name in enumerate(ROUTER_FEATURE_NAMES)
    }
    indices = torch.tensor(
        [all_feature_indices[item] for item in feature_names],
        dtype=torch.long,
    )
    selected_features = features[:, indices].float()
    calibration_mask = torch.tensor(
        [row["partition"] == "calibration" for row in metadata],
        dtype=torch.bool,
    )
    fit_mask = ~calibration_mask
    if not bool(fit_mask.any()) or not bool(calibration_mask.any()):
        raise RuntimeError("Fit/calibration partition is empty")
    mean = selected_features[fit_mask].mean(dim=0)
    standard_deviation = selected_features[fit_mask].std(
        dim=0,
        unbiased=False,
    ).clamp_min(1e-6)
    normalized = (selected_features - mean) / standard_deviation

    router_config = protocol["router"]
    hidden_dims = tuple(int(x) for x in router_config["hidden_dims"])
    model = TaskBenefitRouter(
        input_dim=len(feature_names),
        hidden_dims=(hidden_dims[0], hidden_dims[1]),
        dropout=float(router_config["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(router_config["learning_rate"]),
        weight_decay=float(router_config["weight_decay"]),
    )
    fit_features = normalized[fit_mask]
    fit_benefit = benefit[fit_mask].float()
    fit_clean = clean[fit_mask]
    magnitude_weight = 1.0 + (
        fit_benefit.abs()
        / float(router_config["magnitude_weight_scale"])
    ).clamp(max=float(router_config["magnitude_weight_cap"]))
    sample_weight = magnitude_weight * torch.where(
        fit_clean,
        torch.full_like(
            fit_benefit,
            float(router_config["clean_sample_weight"]),
        ),
        torch.ones_like(fit_benefit),
    )
    dataset = TensorDataset(
        fit_features,
        fit_benefit,
        sample_weight,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(router_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(router_config["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_target, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            per_sample = F.smooth_l1_loss(
                prediction,
                batch_target,
                beta=float(router_config["huber_beta"]),
                reduction="none",
            )
            loss = (per_sample * batch_weight).sum() / (
                batch_weight.sum().clamp_min(1.0)
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * int(batch_target.shape[0])
            sample_count += int(batch_target.shape[0])
        if epoch == 1 or epoch % 25 == 0 or epoch == int(
            router_config["epochs"]
        ):
            model.eval()
            with torch.no_grad():
                calibration_prediction = model(
                    normalized[calibration_mask]
                )
                calibration_rmse = float(
                    torch.sqrt(
                        (
                            calibration_prediction
                            - benefit[calibration_mask]
                        )
                        .square()
                        .mean()
                    )
                )
            record = {
                "epoch": epoch,
                "fit_loss": loss_sum / max(1, sample_count),
                "calibration_rmse": calibration_rmse,
            }
            history.append(record)
            print(
                f"router={name} epoch={epoch:03d} "
                f"fit_loss={record['fit_loss']:.6f} "
                f"cal_rmse={calibration_rmse:.6f}"
            )

    model.eval()
    with torch.no_grad():
        scores = model(normalized)
    clean_calibration = calibration_mask & clean
    if not bool(clean_calibration.any()):
        raise RuntimeError("Calibration partition has no clean samples")
    threshold = max(
        float(router_config["threshold_floor"]),
        float(scores[clean_calibration].max())
        + float(router_config["threshold_epsilon"]),
    )
    selected = scores > threshold
    for index, row in enumerate(metadata):
        row[f"{name}_score"] = float(scores[index])
        row[f"{name}_selected_aligned"] = bool(selected[index])

    def partition_summary(mask: torch.Tensor) -> dict[str, Any]:
        prediction = scores[mask]
        target = benefit[mask]
        chosen = selected[mask]
        clean_subset = clean[mask]
        return {
            "samples": int(mask.sum()),
            "rmse": float(
                torch.sqrt((prediction - target).square().mean())
            ),
            "selection_rate": float(chosen.float().mean()),
            "clean_selection_rate": (
                float(chosen[clean_subset].float().mean())
                if bool(clean_subset.any())
                else None
            ),
            "shift_selection_rate": (
                float(chosen[~clean_subset].float().mean())
                if bool((~clean_subset).any())
                else None
            ),
            "mean_realized_task_loss_gain": float(
                torch.where(chosen, target, torch.zeros_like(target)).mean()
            ),
            "harmful_selection_rate": float(
                (chosen & (target <= 0)).float().mean()
            ),
        }

    summary = {
        "name": name,
        "feature_names": list(feature_names),
        "threshold": threshold,
        "threshold_rule": (
            "max score on clean training-calibration samples plus epsilon, "
            "lower-bounded by zero"
        ),
        "fit": partition_summary(fit_mask),
        "calibration": partition_summary(calibration_mask),
        "history": history,
    }
    checkpoint = {
        "name": name,
        "feature_names": list(feature_names),
        "feature_indices": indices,
        "feature_mean": mean,
        "feature_std": standard_deviation,
        "hidden_dims": hidden_dims,
        "dropout": float(router_config["dropout"]),
        "threshold": threshold,
        "model_state": model.state_dict(),
        "summary": summary,
    }
    torch.save(checkpoint, output_dir / f"{name}.pth")
    save_json(output_dir / f"{name}_summary.json", summary)
    return {"model": model, **checkpoint}


def _router_score(
    bundle: dict[str, Any],
    features: torch.Tensor,
) -> torch.Tensor:
    indices = bundle["feature_indices"].to(features.device)
    mean = bundle["feature_mean"].to(features.device)
    standard_deviation = bundle["feature_std"].to(features.device)
    selected = features[:, indices]
    normalized = (selected - mean) / standard_deviation
    model = bundle["model"].to(features.device)
    model.eval()
    return model(normalized)


@torch.no_grad()
def _evaluate_validation(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    routers: dict[str, dict[str, Any]],
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
        "registration_router",
        "task_router",
        "loss_oracle",
    )
    scenario_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    specs = _scenario_specs(protocol)

    for scenario_index, spec in enumerate(specs):
        print(
            f"[validation {scenario_index + 1}/{len(specs)}] {spec.key}"
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
            gates: dict[str, torch.Tensor] = {
                "q_gate": q_gate,
                "loss_oracle": oracle_gate,
            }
            scores: dict[str, torch.Tensor] = {}
            for router_name, bundle in routers.items():
                score = _router_score(bundle, features)
                scores[router_name] = score
                gates[router_name] = score > float(bundle["threshold"])
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
                        "aligned_task_loss": float(aligned_loss[index]),
                        "aligned_loss_advantage": float(
                            raw_loss[index] - aligned_loss[index]
                        ),
                        "conditional_epe_pixels": float(epe[index]),
                        "q_gate": bool(q_gate[index]),
                        "registration_router_score": float(
                            scores["registration_router"][index]
                        ),
                        "registration_router_gate": bool(
                            gates["registration_router"][index]
                        ),
                        "task_router_score": float(
                            scores["task_router"][index]
                        ),
                        "task_router_gate": bool(
                            gates["task_router"][index]
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
    _write_csv(output_dir / "validation_scenario_metrics.csv", scenario_rows)
    _write_csv(output_dir / "validation_sample_decisions.csv", sample_rows)
    return scenario_rows, sample_rows


def _decision(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    by_key = {
        (str(row["scenario"]), str(row["method"])): row for row in rows
    }
    specs = _scenario_specs(protocol)
    clean_key = "clean:s0:common"
    shift_keys = [spec.key for spec in specs if spec.name == "shift"]
    shift8_key = next(
        spec.key
        for spec in specs
        if spec.name == "shift" and int(spec.value) == 8
    )
    clean_raw = float(by_key[(clean_key, "raw")]["f1"])
    clean_router = float(by_key[(clean_key, "task_router")]["f1"])
    shift_router_gains = [
        float(by_key[(key, "task_router")]["f1"])
        - float(by_key[(key, "raw")]["f1"])
        for key in shift_keys
    ]
    shift8_raw = float(by_key[(shift8_key, "raw")]["f1"])
    shift8_router = float(by_key[(shift8_key, "task_router")]["f1"])
    shift8_oracle = float(by_key[(shift8_key, "loss_oracle")]["f1"])
    shift8_gain = shift8_router - shift8_raw
    oracle_gain = shift8_oracle - shift8_raw
    recovery = shift8_gain / oracle_gain if oracle_gain > 1e-12 else 0.0
    gate_config = protocol["gates"]
    gates = {
        "clean_task_router_f1_drop": {
            "value": clean_raw - clean_router,
            "maximum": float(gate_config["clean_f1_drop_max"]),
        },
        "shift_mean_task_router_f1_gain": {
            "value": sum(shift_router_gains)
            / max(1, len(shift_router_gains)),
            "minimum": float(gate_config["shift_mean_f1_gain_min"]),
        },
        "shift8_task_router_f1_gain": {
            "value": shift8_gain,
            "minimum": float(gate_config["shift8_f1_gain_min"]),
        },
        "shift8_oracle_gain_recovery": {
            "value": recovery,
            "minimum": float(
                gate_config["shift8_oracle_recovery_min"]
            ),
        },
        "shift8_task_router_aligned_selection_rate": {
            "value": float(
                by_key[(shift8_key, "task_router")][
                    "aligned_selection_rate"
                ]
            ),
            "minimum": float(
                gate_config["shift8_aligned_selection_rate_min"]
            ),
        },
    }
    for gate in gates.values():
        value = float(gate["value"])
        if "maximum" in gate:
            gate["passed"] = value <= float(gate["maximum"])
        else:
            gate["passed"] = value >= float(gate["minimum"])
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "status": "passed" if passed else "failed",
        "validation_used_once_for_fixed_protocol": True,
        "test_split_untouched": True,
        "gates": gates,
        "next_action": (
            "external_s2looking_validation_and_router_ablation"
            if passed
            else "diagnose_router_generalization_without_val_tuning"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train risk-controlled task-benefit routers on the training "
            "split and evaluate the fixed protocol once on validation."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p7_task_benefit_router.yaml",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Task-benefit router protocol must be a mapping")
    calibration_fraction = float(
        protocol["partition"]["calibration_fraction"]
    )
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")

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
    config["model"]["registration_only"] = False
    config["model"]["freeze_registration"] = True
    config["model"]["offset_decoding"] = "hard"
    config["model"]["alignment_gate"] = "threshold_nonzero"
    seed_everything(int(protocol["seed"]))
    data = config["data"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    train_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=str(protocol["train_split"]),
        image_size=None,
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    features, benefit, clean, metadata = _collect_training_features(
        model,
        train_dataset,
        protocol=protocol,
        config=config,
        device=device,
    )
    routers = {
        "registration_router": _train_router(
            "registration_router",
            features,
            benefit,
            clean,
            metadata,
            REGISTRATION_ROUTER_FEATURE_NAMES,
            protocol=protocol,
            output_dir=output_dir,
        ),
        "task_router": _train_router(
            "task_router",
            features,
            benefit,
            clean,
            metadata,
            ROUTER_FEATURE_NAMES,
            protocol=protocol,
            output_dir=output_dir,
        ),
    }
    _write_csv(output_dir / "training_router_features.csv", metadata)

    validation_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=str(protocol["validation_split"]),
        image_size=None,
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    validation_rows, _ = _evaluate_validation(
        model,
        validation_dataset,
        routers,
        protocol=protocol,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    decision = _decision(validation_rows, protocol=protocol)
    results = {
        "status": "completed",
        "source_checkpoint": str(checkpoint_path),
        "protocol": str(protocol_path.resolve()),
        "feature_names": list(ROUTER_FEATURE_NAMES),
        "router_summaries": {
            name: bundle["summary"] for name, bundle in routers.items()
        },
        "decision": decision,
        "validation_results": validation_rows,
    }
    save_json(output_dir / "results.json", results)
    save_json(output_dir / "decision.json", decision)
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("===== P7 TASK-BENEFIT ROUTER DECISION =====")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Saved task-benefit router experiment to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
