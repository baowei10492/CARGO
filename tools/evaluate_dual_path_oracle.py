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

from rs_cd.datasets import PairedChangeDataset
from rs_cd.dual_path import (
    per_sample_binary_change_loss,
    select_aligned_by_task_loss,
)
from rs_cd.geometry import interior_valid_mask
from rs_cd.metrics import BinaryChangeMetrics
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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the label-aware upper bound of a safe Raw/Aligned "
            "dual-path router. Oracle results are diagnostic only."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p7_dual_path_task_oracle.yaml",
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
        raise ValueError("Dual-path Oracle protocol must be a mapping")

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

    seed = int(protocol["seed"])
    seed_everything(seed)
    split = str(args.split or protocol.get("split", "val"))
    data = config["data"]
    base_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=split,
        image_size=data.get("eval_crop_size"),
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    estimate_registration = getattr(model, "estimate_registration", None)
    forced_forward = getattr(model, "forward_with_forced_offset", None)
    if not callable(estimate_registration) or not callable(forced_forward):
        raise TypeError(
            "Checkpoint must provide P7 registration and forced-offset paths"
        )

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
    variants = ("raw", "always_aligned", "q_gate", "loss_oracle")
    scenario_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for scenario_index, spec in enumerate(_scenario_specs(protocol)):
        print(f"[{scenario_index + 1}/4] scenario={spec.key}")
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
        sample_count = 0
        q_aligned_count = 0
        oracle_aligned_count = 0
        route_agreement_count = 0

        for batch in loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            registration = estimate_registration(t1, t2)
            conditional_offset = registration[
                "conditional_hard_offset_pixels"
            ]
            if not isinstance(conditional_offset, torch.Tensor):
                raise TypeError("Conditional hard offset is not a tensor")
            forced = forced_forward(t1, t2, conditional_offset)
            raw_logits = forced["raw_change_logits"]
            aligned_logits = forced["aligned_change_logits"]
            if not isinstance(raw_logits, torch.Tensor):
                raise TypeError("Raw logits are not a tensor")
            if not isinstance(aligned_logits, torch.Tensor):
                raise TypeError("Aligned logits are not a tensor")

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
            q_logits = torch.where(
                q_gate[:, None, None, None],
                aligned_logits,
                raw_logits,
            )
            q_loss = per_sample_binary_change_loss(
                q_logits,
                mask,
                valid,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                pos_weight=pos_weight,
            )
            oracle_loss = torch.minimum(raw_loss, aligned_loss)
            logits_by_variant = {
                "raw": raw_logits,
                "always_aligned": aligned_logits,
                "q_gate": q_logits,
                "loss_oracle": oracle_logits,
            }
            losses_by_variant = {
                "raw": raw_loss,
                "always_aligned": aligned_loss,
                "q_gate": q_loss,
                "loss_oracle": oracle_loss,
            }
            for name in variants:
                accumulators[name].update(
                    logits_by_variant[name],
                    mask,
                    from_logits=True,
                    valid_mask=valid,
                )
                loss_sums[name] += float(
                    losses_by_variant[name].sum()
                )

            target = torch.stack(
                (
                    torch.as_tensor(batch["shift_dx"]),
                    torch.as_tensor(batch["shift_dy"]),
                ),
                dim=1,
            ).to(conditional_offset)
            epe = torch.linalg.vector_norm(
                conditional_offset - target,
                dim=1,
            )
            q_aligned_count += int(q_gate.sum())
            oracle_aligned_count += int(oracle_gate.sum())
            route_agreement_count += int((q_gate == oracle_gate).sum())
            sample_count += int(t1.shape[0])
            for item_index, sample_id in enumerate(batch["id"]):
                sample_rows.append(
                    {
                        "scenario": spec.key,
                        "id": str(sample_id),
                        "target_dx_pixels": float(target[item_index, 0]),
                        "target_dy_pixels": float(target[item_index, 1]),
                        "conditional_dx_pixels": float(
                            conditional_offset[item_index, 0]
                        ),
                        "conditional_dy_pixels": float(
                            conditional_offset[item_index, 1]
                        ),
                        "conditional_epe_pixels": float(epe[item_index]),
                        "raw_task_loss": float(raw_loss[item_index]),
                        "aligned_task_loss": float(
                            aligned_loss[item_index]
                        ),
                        "aligned_loss_advantage": float(
                            raw_loss[item_index] - aligned_loss[item_index]
                        ),
                        "q_selected_aligned": bool(q_gate[item_index]),
                        "oracle_selected_aligned": bool(
                            oracle_gate[item_index]
                        ),
                        "q_oracle_agree": bool(
                            q_gate[item_index] == oracle_gate[item_index]
                        ),
                        "shift_probability": float(
                            registration[
                                "registration_shift_probability"
                            ][item_index]
                        ),
                        "zero_probability": float(
                            registration[
                                "registration_zero_probability"
                            ][item_index]
                        ),
                    }
                )

        scenario_metrics: dict[str, dict[str, float | int]] = {}
        for name in variants:
            values = accumulators[name].compute()
            scenario_metrics[name] = values
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
                        q_aligned_count / max(1, sample_count)
                        if name == "q_gate"
                        else (
                            oracle_aligned_count / max(1, sample_count)
                            if name == "loss_oracle"
                            else None
                        )
                    ),
                    "q_oracle_agreement_rate": (
                        route_agreement_count / max(1, sample_count)
                        if name in {"q_gate", "loss_oracle"}
                        else None
                    ),
                }
            )
        print(
            " ".join(
                f"{name}={float(values['f1']):.6f}"
                for name, values in scenario_metrics.items()
            )
        )
        print(
            f"q_aligned={q_aligned_count / max(1, sample_count):.3f} "
            "oracle_aligned="
            f"{oracle_aligned_count / max(1, sample_count):.3f} "
            "agreement="
            f"{route_agreement_count / max(1, sample_count):.3f}"
        )

    by_key = {
        (str(row["scenario"]), str(row["method"])): row
        for row in scenario_rows
    }
    clean_key = "clean:s0:common"
    shift_keys = [
        spec.key
        for spec in _scenario_specs(protocol)
        if spec.name == "shift"
    ]
    shift8_key = next(
        spec.key
        for spec in _scenario_specs(protocol)
        if spec.name == "shift" and int(spec.value) == 8
    )
    clean_raw = float(by_key[(clean_key, "raw")]["f1"])
    clean_oracle = float(by_key[(clean_key, "loss_oracle")]["f1"])
    shift_oracle_gains = [
        float(by_key[(key, "loss_oracle")]["f1"])
        - float(by_key[(key, "raw")]["f1"])
        for key in shift_keys
    ]
    shift8_oracle_gain = (
        float(by_key[(shift8_key, "loss_oracle")]["f1"])
        - float(by_key[(shift8_key, "raw")]["f1"])
    )
    shift8_aligned_rate = float(
        by_key[(shift8_key, "loss_oracle")]["aligned_selection_rate"]
    )
    gates = {
        "clean_oracle_f1_drop": {
            "value": clean_raw - clean_oracle,
            "maximum": float(protocol["gates"]["clean_f1_drop_max"]),
        },
        "shift_mean_oracle_f1_gain": {
            "value": sum(shift_oracle_gains)
            / max(1, len(shift_oracle_gains)),
            "minimum": float(
                protocol["gates"]["shift_mean_f1_gain_min"]
            ),
        },
        "shift8_oracle_f1_gain": {
            "value": shift8_oracle_gain,
            "minimum": float(
                protocol["gates"]["shift8_f1_gain_min"]
            ),
        },
        "shift8_oracle_aligned_selection_rate": {
            "value": shift8_aligned_rate,
            "minimum": float(
                protocol["gates"]["shift8_aligned_selection_rate_min"]
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
    next_action = (
        "train_task_benefit_router_on_training_data"
        if passed
        else "stop_router_and_revisit_alignment_experts"
    )
    decision = {
        "status": "passed" if passed else "failed",
        "diagnostic_only": True,
        "labels_used_for_oracle_routing": True,
        "test_split_untouched": split != "test",
        "gates": gates,
        "next_action": next_action,
    }
    results = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": str(protocol_path),
        "protocol_sha256": _protocol_hash(protocol),
        "split": split,
        "loss": {
            "bce_weight": bce_weight,
            "dice_weight": dice_weight,
            "pos_weight": pos_weight,
        },
        "coordinate_convention": (
            "reference(x,y) corresponds to target(x+dx,y+dy)"
        ),
        "methods": {
            "raw": "Frozen P3b on the corrupted pair.",
            "always_aligned": (
                "Frozen P3b after the conditional non-zero hard offset."
            ),
            "q_gate": (
                "Existing P7C q_shift gate chooses Raw or Aligned."
            ),
            "loss_oracle": (
                "Per-image target BCE+Dice chooses Raw or Aligned; "
                "diagnostic upper bound only."
            ),
        },
        "decision": decision,
        "results": scenario_rows,
    }
    save_json(output_dir / "results.json", results)
    save_json(output_dir / "decision.json", decision)
    _write_csv(output_dir / "scenario_metrics.csv", scenario_rows)
    _write_csv(output_dir / "sample_decisions.csv", sample_rows)
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print("===== DUAL-PATH ORACLE DECISION =====")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Saved dual-path Oracle audit to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
