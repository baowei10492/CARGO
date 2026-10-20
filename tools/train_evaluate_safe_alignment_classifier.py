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
    ROUTER_FEATURE_NAMES,
    SAFE_REGISTRATION_FEATURE_NAMES,
    SAFE_TASK_FEATURE_NAMES,
    TILE_CONSISTENCY_FEATURE_NAMES,
    tile_consistency_features,
    SafeAlignmentClassifier,
    TaskBenefitRouter,
    select_aligned_by_task_loss,
)
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.robustness import CorruptedPairDataset
from rs_cd.runtime import (
    build_model,
    save_json,
    seed_everything,
    select_device,
)
from tools.train import _loader
from tools.train_evaluate_task_benefit_router import (
    _forward_experts,
    _scenario_specs,
    _valid_mask,
    _write_csv,
)


SPLIT_NAMES = ("fit", "early_stop", "risk_calibration")


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_split_assignment(
    sample_id: str,
    *,
    split_seed: int,
    fit_fraction: float,
    early_stop_fraction: float,
) -> str:
    """Assign one of three splits from a stable per-ID hash.

    All scenario variants of the same original image ID receive the same
    split, so clean/Shift-2/Shift-4/Shift-8 never leak across fit,
    early-stop, and risk-calibration.
    """

    digest = hashlib.sha256(
        f"{split_seed}|safe-alignment-split|{sample_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64)
    if value < fit_fraction:
        return "fit"
    if value < fit_fraction + early_stop_fraction:
        return "early_stop"
    return "risk_calibration"


def _yes(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _tile_feature_config(
    protocol: dict[str, Any],
) -> dict[str, Any] | None:
    """Parse the optional tile-consistency block of a P7-5 protocol.

    Absent or disabled -> None, and every code path below reduces to the
    frozen P7-3 behaviour byte for byte (a test asserts the universe is
    then exactly ROUTER_FEATURE_NAMES).
    """

    block = protocol.get("tile_features")
    if not block or not bool(block.get("enabled", False)):
        return None
    requested = tuple(str(name) for name in block["names"])
    unknown = [
        name
        for name in requested
        if name not in TILE_CONSISTENCY_FEATURE_NAMES
    ]
    if unknown:
        raise ValueError(
            f"Unknown tile-consistency features: {unknown}; allowed: "
            f"{list(TILE_CONSISTENCY_FEATURE_NAMES)}"
        )
    # Canonical order, independent of the yaml listing order.
    names = tuple(
        name
        for name in TILE_CONSISTENCY_FEATURE_NAMES
        if name in requested
    )
    if not names:
        raise ValueError("tile_features.names must not be empty")
    return {
        "names": names,
        "tile_size": int(block["tile_size"]),
    }


def _feature_universe(
    tile_config: dict[str, Any] | None,
) -> tuple[str, ...]:
    if tile_config is None:
        return ROUTER_FEATURE_NAMES
    return (*ROUTER_FEATURE_NAMES, *tile_config["names"])


def _join_tile_features(
    rows: list[dict[str, Any]],
    tile_csv_path: Path,
    names: tuple[str, ...],
) -> int:
    """Attach training-side tile features by (id, scenario) join.

    Every training row must find a partner; a silent partial join would
    train on rows whose tile columns default to something arbitrary.
    """

    with tile_csv_path.open("r", encoding="utf-8", newline="") as handle:
        tile_rows = list(csv.DictReader(handle))
    if not tile_rows:
        raise RuntimeError(f"Tile feature CSV is empty: {tile_csv_path}")
    missing_columns = [
        name for name in names if name not in tile_rows[0]
    ]
    if missing_columns:
        raise RuntimeError(
            f"Tile feature CSV lacks columns: {missing_columns}"
        )
    table: dict[tuple[str, str], dict[str, float]] = {}
    for raw in tile_rows:
        key = (str(raw["id"]), str(raw["scenario"]))
        if key in table:
            raise RuntimeError(f"Duplicate tile feature row: {key}")
        table[key] = {name: float(raw[name]) for name in names}
    unmatched = 0
    for row in rows:
        entry = table.get((str(row["id"]), str(row["scenario"])))
        if entry is None:
            unmatched += 1
            continue
        row.update(entry)
    if unmatched:
        raise RuntimeError(
            f"{unmatched} training rows lack tile features; the tile "
            "records must cover every (id, scenario) pair"
        )
    return len(rows)


def _load_feature_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise RuntimeError(f"Feature CSV is empty: {csv_path}")
    missing = [
        name for name in ROUTER_FEATURE_NAMES if name not in raw_rows[0]
    ]
    if missing:
        raise RuntimeError(
            f"Feature CSV is missing router features: {missing}"
        )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row: dict[str, Any] = {
            "id": str(raw["id"]),
            "scenario": str(raw["scenario"]),
            "raw_task_loss": float(raw["raw_task_loss"]),
            "aligned_task_loss": float(raw["aligned_task_loss"]),
            "aligned_loss_advantage": float(
                raw["aligned_loss_advantage"]
            ),
            "existing_q_selected_aligned": _yes(
                raw["existing_q_selected_aligned"]
            ),
        }
        for name in ROUTER_FEATURE_NAMES:
            row[name] = float(raw[name])
        rows.append(row)
    return rows


def _prepare_training_tensors(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    feature_universe: tuple[str, ...] = ROUTER_FEATURE_NAMES,
) -> dict[str, Any]:
    partition = protocol["partition"]
    fit_fraction = float(partition["fit_fraction"])
    early_stop_fraction = float(partition["early_stop_fraction"])
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")
    if not 0.0 < early_stop_fraction < 1.0 - fit_fraction:
        raise ValueError(
            "early_stop_fraction must leave a non-empty risk split"
        )
    split_seed = int(protocol["split_seed"])
    label_config = protocol["label"]
    benefit_margin = float(label_config["benefit_margin"])
    weight_config = protocol["weights"]
    magnitude_scale = float(weight_config["magnitude_scale"])
    magnitude_floor = float(weight_config["magnitude_floor"])
    magnitude_cap = float(weight_config["magnitude_cap"])
    unsafe_weight = float(weight_config["unsafe_weight"])
    clean_unsafe_extra = float(
        weight_config["clean_unsafe_extra_weight"]
    )

    features = torch.tensor(
        [
            [row[name] for name in feature_universe]
            for row in rows
        ],
        dtype=torch.float32,
    )
    benefit = torch.tensor(
        [row["aligned_loss_advantage"] for row in rows],
        dtype=torch.float32,
    )
    clean = torch.tensor(
        [row["scenario"].startswith("clean") for row in rows],
        dtype=torch.bool,
    )
    q_gate = torch.tensor(
        [row["existing_q_selected_aligned"] for row in rows],
        dtype=torch.bool,
    )
    safe = benefit > benefit_margin
    weight = (benefit.abs() / magnitude_scale).clamp(
        min=magnitude_floor,
        max=magnitude_cap,
    )
    weight = torch.where(safe, weight, weight * unsafe_weight)
    weight = torch.where(
        (~safe) & clean,
        weight * clean_unsafe_extra,
        weight,
    )
    splits = [
        _stable_split_assignment(
            row["id"],
            split_seed=split_seed,
            fit_fraction=fit_fraction,
            early_stop_fraction=early_stop_fraction,
        )
        for row in rows
    ]
    split_masks = {
        name: torch.tensor(
            [split == name for split in splits],
            dtype=torch.bool,
        )
        for name in SPLIT_NAMES
    }
    for name in SPLIT_NAMES:
        if not bool(split_masks[name].any()):
            raise RuntimeError(f"Split '{name}' is empty")
    return {
        "features": features,
        "benefit": benefit,
        "clean": clean,
        "q_gate": q_gate,
        "safe": safe,
        "weight": weight,
        "splits": splits,
        "split_masks": split_masks,
    }


def _weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def _q_veto_audit(
    *,
    scenarios: list[str],
    probability: torch.Tensor,
    tau: float,
    mask: torch.Tensor,
    benefit: torch.Tensor,
    q_gate: torch.Tensor,
    scenario_labels: list[str],
) -> dict[str, Any]:
    audit: dict[str, Any] = {"scenarios": {}}
    total_q_benefit = 0.0
    total_veto_benefit = 0.0
    total_selected = 0
    total_harmful = 0
    for scenario in scenarios:
        scenario_mask = torch.tensor(
            [label == scenario for label in scenario_labels],
            dtype=torch.bool,
        )
        rows = mask & scenario_mask
        q_rows = rows & q_gate
        veto_rows = q_rows & (probability >= tau)
        q_benefit = float(benefit[q_rows].sum())
        veto_benefit = float(benefit[veto_rows].sum())
        harmful = int((benefit[veto_rows] <= 0).sum())
        entry = {
            "q_selected": int(q_rows.sum()),
            "q_harmful": int((benefit[q_rows] <= 0).sum()),
            "q_benefit": q_benefit,
            "veto_selected": int(veto_rows.sum()),
            "veto_harmful": harmful,
            "veto_benefit": veto_benefit,
            "retained_benefit_ratio": (
                veto_benefit / q_benefit
                if abs(q_benefit) > 1e-12
                else None
            ),
        }
        audit["scenarios"][scenario] = entry
        if not scenario.startswith("clean"):
            total_q_benefit += q_benefit
            total_veto_benefit += veto_benefit
        total_selected += entry["veto_selected"]
        total_harmful += harmful
    audit["shift_q_benefit"] = total_q_benefit
    audit["shift_veto_benefit"] = total_veto_benefit
    audit["shift_retained_benefit_ratio"] = (
        total_veto_benefit / total_q_benefit
        if abs(total_q_benefit) > 1e-12
        else None
    )
    audit["harmful_selection_rate"] = (
        total_harmful / total_selected if total_selected else 0.0
    )
    return audit


def train_safe_alignment_classifier(
    name: str,
    rows: list[dict[str, Any]],
    prepared: dict[str, Any],
    feature_names: tuple[str, ...],
    *,
    protocol: dict[str, Any],
    output_dir: Path | None,
    feature_universe: tuple[str, ...] = ROUTER_FEATURE_NAMES,
) -> dict[str, Any]:
    seed = int(protocol["seed"])
    seed_everything(seed)
    all_feature_indices = {
        feature: index
        for index, feature in enumerate(feature_universe)
    }
    indices = torch.tensor(
        [all_feature_indices[item] for item in feature_names],
        dtype=torch.long,
    )
    selected = prepared["features"][:, indices].float()
    split_masks = prepared["split_masks"]
    fit_mask = split_masks["fit"]
    early_stop_mask = split_masks["early_stop"]
    risk_mask = split_masks["risk_calibration"]
    mean = selected[fit_mask].mean(dim=0)
    standard_deviation = selected[fit_mask].std(
        dim=0,
        unbiased=False,
    ).clamp_min(1e-6)
    normalized = (selected - mean) / standard_deviation

    safe = prepared["safe"].float()
    weight = prepared["weight"].float()
    classifier_config = protocol["classifier"]
    hidden_dims = tuple(
        int(x) for x in classifier_config["hidden_dims"]
    )
    model = SafeAlignmentClassifier(
        input_dim=len(feature_names),
        hidden_dims=(hidden_dims[0], hidden_dims[1]),
        dropout=float(classifier_config["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(classifier_config["learning_rate"]),
        weight_decay=float(classifier_config["weight_decay"]),
    )
    dataset = TensorDataset(
        normalized[fit_mask],
        safe[fit_mask],
        weight[fit_mask],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(classifier_config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    max_epochs = int(classifier_config["max_epochs"])
    patience = int(classifier_config["early_stop_patience"])
    best_metric = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch_features, batch_target, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_features)
            loss = _weighted_bce(
                prediction,
                batch_target,
                batch_weight,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * int(
                batch_target.shape[0]
            )
            sample_count += int(batch_target.shape[0])
        model.eval()
        with torch.no_grad():
            early_stop_metric = float(
                _weighted_bce(
                    model(normalized[early_stop_mask]),
                    safe[early_stop_mask],
                    weight[early_stop_mask],
                )
            )
        record = {
            "epoch": epoch,
            "fit_loss": loss_sum / max(1, sample_count),
            "early_stop_loss": early_stop_metric,
        }
        history.append(record)
        if early_stop_metric < best_metric - 1e-9:
            best_metric = early_stop_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == max_epochs:
            print(
                f"classifier={name} epoch={epoch:03d} "
                f"fit_loss={record['fit_loss']:.6f} "
                f"early_stop_loss={early_stop_metric:.6f} "
                f"best_epoch={best_epoch:03d}"
            )
        if epochs_without_improvement >= patience:
            print(
                f"classifier={name} early stop at epoch={epoch:03d} "
                f"(best_epoch={best_epoch:03d})"
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(normalized))

    risk_config = protocol["risk"]
    probability_floor = float(risk_config["probability_floor"])
    epsilon = float(risk_config["threshold_epsilon"])
    clean = prepared["clean"]
    unsafe = ~prepared["safe"]
    clean_unsafe_risk = risk_mask & clean & unsafe
    threshold_fallback_used = False
    if bool(clean_unsafe_risk.any()):
        reference = probability[clean_unsafe_risk]
    else:
        threshold_fallback_used = True
        reference = probability[risk_mask & clean]
    if reference.numel() == 0:
        raise RuntimeError(
            "Risk-calibration split has no clean reference samples"
        )
    tau = max(probability_floor, float(reference.max()) + epsilon)

    scenario_labels = [row["scenario"] for row in rows]
    scenario_order: list[str] = []
    for label in scenario_labels:
        if label not in scenario_order:
            scenario_order.append(label)
    benefit = prepared["benefit"]
    q_gate = prepared["q_gate"]
    grid_points = int(risk_config["coverage_grid_points"])
    coverage_rows: list[dict[str, Any]] = []
    for grid_index in range(grid_points):
        grid_tau = grid_index / max(1, grid_points - 1)
        for scenario in scenario_order:
            scenario_mask = torch.tensor(
                [label == scenario for label in scenario_labels],
                dtype=torch.bool,
            )
            rows_mask = risk_mask & scenario_mask
            q_rows = rows_mask & q_gate
            accepted = q_rows & (probability >= grid_tau)
            coverage_rows.append(
                {
                    "tau": grid_tau,
                    "scenario": scenario,
                    "q_candidates": int(q_rows.sum()),
                    "q_benefit": float(benefit[q_rows].sum()),
                    "accepted": int(accepted.sum()),
                    "accepted_harmful": int(
                        (benefit[accepted] <= 0).sum()
                    ),
                    "accepted_benefit": float(
                        benefit[accepted].sum()
                    ),
                    "clean_probability_pass": int(
                        (
                            rows_mask
                            & clean
                            & (probability >= grid_tau)
                        ).sum()
                    ),
                }
            )
    risk_audit = _q_veto_audit(
        scenarios=scenario_order,
        probability=probability,
        tau=tau,
        mask=risk_mask,
        benefit=benefit,
        q_gate=q_gate,
        scenario_labels=scenario_labels,
    )

    def split_summary(mask: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            split_loss = float(
                _weighted_bce(
                    model(normalized[mask]),
                    safe[mask],
                    weight[mask],
                )
            )
        predicted_safe = probability[mask] >= 0.5
        actual_safe = prepared["safe"][mask]
        return {
            "samples": int(mask.sum()),
            "weighted_bce": split_loss,
            "safe_rate": float(actual_safe.float().mean()),
            "accuracy_at_half": float(
                (predicted_safe == actual_safe).float().mean()
            ),
        }

    summary = {
        "name": name,
        "feature_names": list(feature_names),
        "best_epoch": best_epoch,
        "trained_epochs": len(history),
        "max_epochs": max_epochs,
        "best_early_stop_loss": best_metric,
        "probability_threshold": tau,
        "threshold_rule": (
            "max predicted safe-probability among clean unsafe "
            "risk-calibration samples plus epsilon, lower-bounded by "
            "the probability floor"
        ),
        "threshold_fallback_used": threshold_fallback_used,
        "fit": split_summary(prepared["split_masks"]["fit"]),
        "early_stop": split_summary(early_stop_mask),
        "risk_calibration": split_summary(risk_mask),
        "risk_calibration_q_veto_audit": risk_audit,
        "history_tail": history[-5:],
    }
    checkpoint = {
        "name": name,
        "feature_names": list(feature_names),
        "feature_indices": indices,
        "feature_mean": mean,
        "feature_std": standard_deviation,
        "hidden_dims": hidden_dims,
        "dropout": float(classifier_config["dropout"]),
        "probability_threshold": tau,
        "best_epoch": best_epoch,
        "model_state": best_state,
        "summary": summary,
    }
    if output_dir is not None:
        torch.save(checkpoint, output_dir / f"{name}.pth")
        save_json(output_dir / f"{name}_summary.json", summary)
        _write_csv(
            output_dir / f"{name}_risk_coverage.csv",
            coverage_rows,
        )
    print(
        f"classifier={name} tau={tau:.6f} "
        f"risk_shift_retained="
        f"{risk_audit['shift_retained_benefit_ratio']} "
        f"risk_harmful_rate="
        f"{risk_audit['harmful_selection_rate']:.4f}"
    )
    return {
        "model": model,
        "probability": probability,
        "coverage_rows": coverage_rows,
        **checkpoint,
    }


def _classifier_probability(
    bundle: dict[str, Any],
    features: torch.Tensor,
) -> torch.Tensor:
    indices = bundle["feature_indices"].to(features.device)
    mean = bundle["feature_mean"].to(features.device)
    standard_deviation = bundle["feature_std"].to(features.device)
    normalized = (features[:, indices] - mean) / standard_deviation
    model = bundle["model"].to(features.device)
    model.eval()
    return torch.sigmoid(model(normalized))


def _load_task_router_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hidden_dims = tuple(int(x) for x in checkpoint["hidden_dims"])
    model = TaskBenefitRouter(
        input_dim=len(checkpoint["feature_names"]),
        hidden_dims=(hidden_dims[0], hidden_dims[1]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {**checkpoint, "model": model.to(device)}


def _task_router_score(
    bundle: dict[str, Any],
    features: torch.Tensor,
) -> torch.Tensor:
    indices = bundle["feature_indices"].to(features.device)
    mean = bundle["feature_mean"].to(features.device)
    standard_deviation = bundle["feature_std"].to(features.device)
    normalized = (features[:, indices] - mean) / standard_deviation
    model = bundle["model"].to(features.device)
    model.eval()
    return model(normalized)


@torch.no_grad()
@torch.no_grad()
def _batch_tile_features(
    model: torch.nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    registration: dict[str, torch.Tensor],
    *,
    tile_size: int,
    names: tuple[str, ...],
    amp: bool,
) -> torch.Tensor:
    """Compute tile-consistency features for one validation batch.

    Mirrors the training-side measurement exactly: per-tile conditional
    hard offsets from ``estimate_registration`` on non-overlapping
    tiles, reduced against the whole-image conditional estimate (the
    same convention as tools.diagnose_registration_hygiene's records,
    which is what the screening AUROCs were computed on).
    """

    from tools.diagnose_registration_hygiene import _split_tiles

    estimate = getattr(model, "estimate_registration", None)
    if not callable(estimate):
        raise TypeError("Model does not expose estimate_registration")
    tiles_t1 = _split_tiles(t1, tile_size)
    tiles_t2 = _split_tiles(t2, tile_size)
    per_tile: list[torch.Tensor] = []
    with torch.autocast(
        device_type=t1.device.type,
        enabled=amp and t1.device.type == "cuda",
    ):
        for first, second in zip(tiles_t1, tiles_t2):
            outputs = estimate(
                first.contiguous(),
                second.contiguous(),
            )
            per_tile.append(
                outputs["conditional_hard_offset_pixels"].float()
            )
    whole = registration["conditional_hard_offset_pixels"].float()
    values: list[list[float]] = []
    for index in range(whole.shape[0]):
        tile_offsets = [
            (float(offsets[index, 0]), float(offsets[index, 1]))
            for offsets in per_tile
        ]
        reduced = tile_consistency_features(
            tile_offsets,
            (float(whole[index, 0]), float(whole[index, 1])),
        )
        values.append([reduced[name] for name in names])
    return torch.tensor(
        values,
        dtype=torch.float32,
        device=t1.device,
    )


def _evaluate_validation(
    model: torch.nn.Module,
    base_dataset: PairedChangeDataset,
    classifiers: dict[str, dict[str, Any]],
    task_router_bundle: dict[str, Any] | None,
    *,
    protocol: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    tile_config: dict[str, Any] | None = None,
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
    variants = [
        "raw",
        "always_aligned",
        "q_gate",
        "safe_veto_registration",
        "safe_veto_task",
    ]
    if task_router_bundle is not None:
        variants.append("p72_task_router")
    variants.append("loss_oracle")
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
            if tile_config is not None:
                tile_tensor = _batch_tile_features(
                    model,
                    t1,
                    t2,
                    registration,
                    tile_size=int(tile_config["tile_size"]),
                    names=tuple(tile_config["names"]),
                    amp=bool(protocol.get("amp", True)),
                )
                features = torch.cat([features, tile_tensor], dim=1)
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
                ("safe_veto_registration", "safe_alignment_registration"),
                ("safe_veto_task", "safe_alignment_task"),
            ):
                bundle = classifiers[bundle_name]
                probability = _classifier_probability(bundle, features)
                probabilities[variant_name] = probability
                gates[variant_name] = q_gate & (
                    probability
                    >= float(bundle["probability_threshold"])
                )
            task_router_scores = None
            if task_router_bundle is not None:
                task_router_scores = _task_router_score(
                    task_router_bundle,
                    features,
                )
                gates["p72_task_router"] = task_router_scores > float(
                    task_router_bundle["threshold"]
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
                sample_row = {
                    "id": str(sample_id),
                    "scenario": spec.key,
                    "raw_task_loss": float(raw_loss[index]),
                    "aligned_task_loss": float(aligned_loss[index]),
                    "aligned_loss_advantage": float(
                        raw_loss[index] - aligned_loss[index]
                    ),
                    "conditional_epe_pixels": float(epe[index]),
                    "q_gate": bool(q_gate[index]),
                    "safe_registration_probability": float(
                        probabilities["safe_veto_registration"][index]
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
                if task_router_scores is not None:
                    sample_row["p72_task_router_score"] = float(
                        task_router_scores[index]
                    )
                    sample_row["p72_task_router_gate"] = bool(
                        gates["p72_task_router"][index]
                    )
                sample_rows.append(sample_row)
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
        output_dir / "validation_scenario_metrics.csv",
        scenario_rows,
    )
    _write_csv(
        output_dir / "validation_sample_decisions.csv",
        sample_rows,
    )
    return scenario_rows, sample_rows


def _primary_gains(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
) -> dict[str, float]:
    """Shared gain arithmetic for a scenario-metrics table."""

    primary = str(protocol.get("primary_method", "safe_veto_task"))
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
    shift_gains = [
        float(by_key[(key, primary)]["f1"])
        - float(by_key[(key, "raw")]["f1"])
        for key in shift_keys
    ]
    return {
        "clean_drop": float(by_key[(clean_key, "raw")]["f1"])
        - float(by_key[(clean_key, primary)]["f1"]),
        "shift_mean_gain": sum(shift_gains) / max(1, len(shift_gains)),
        "shift8_gain": float(by_key[(shift8_key, primary)]["f1"])
        - float(by_key[(shift8_key, "raw")]["f1"]),
    }


def _decision(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    primary = str(protocol.get("primary_method", "safe_veto_task"))
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
    clean_primary = float(by_key[(clean_key, primary)]["f1"])
    shift_gains = [
        float(by_key[(key, primary)]["f1"])
        - float(by_key[(key, "raw")]["f1"])
        for key in shift_keys
    ]
    shift8_raw = float(by_key[(shift8_key, "raw")]["f1"])
    shift8_primary = float(by_key[(shift8_key, primary)]["f1"])
    shift8_oracle = float(by_key[(shift8_key, "loss_oracle")]["f1"])
    shift8_gain = shift8_primary - shift8_raw
    oracle_gain = shift8_oracle - shift8_raw
    recovery = shift8_gain / oracle_gain if oracle_gain > 1e-12 else 0.0
    gate_config = protocol["gates"]
    gates = {
        "clean_primary_f1_drop": {
            "value": clean_raw - clean_primary,
            "maximum": float(gate_config["clean_f1_drop_max"]),
        },
        "shift_mean_primary_f1_gain": {
            "value": sum(shift_gains) / max(1, len(shift_gains)),
            "minimum": float(gate_config["shift_mean_f1_gain_min"]),
        },
        "shift8_primary_f1_gain": {
            "value": shift8_gain,
            "minimum": float(gate_config["shift8_f1_gain_min"]),
        },
        "shift8_oracle_gain_recovery": {
            "value": recovery,
            "minimum": float(
                gate_config["shift8_oracle_recovery_min"]
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

    improvement = None
    improvement_passed = None
    improvement_config = protocol.get("improvement_gates")
    if improvement_config is not None:
        if baseline_rows is None:
            raise ValueError(
                "Protocol declares improvement_gates but no baseline "
                "scenario metrics were provided"
            )
        baseline = _primary_gains(baseline_rows, protocol=protocol)
        current = _primary_gains(rows, protocol=protocol)
        improvement = {
            "baseline_primary_gains": baseline,
            "current_primary_gains": current,
            "shift8_f1_gain_improvement": {
                "value": current["shift8_gain"]
                - baseline["shift8_gain"],
                "minimum": float(
                    improvement_config["shift8_gain_improvement_min"]
                ),
            },
            "shift_mean_f1_gain_delta": {
                "value": current["shift_mean_gain"]
                - baseline["shift_mean_gain"],
                "minimum": -float(
                    improvement_config["shift_mean_gain_drop_max"]
                ),
            },
        }
        for key in (
            "shift8_f1_gain_improvement",
            "shift_mean_f1_gain_delta",
        ):
            gate = improvement[key]
            gate["passed"] = float(gate["value"]) >= float(
                gate["minimum"]
            )
        improvement_passed = all(
            bool(improvement[key]["passed"])
            for key in (
                "shift8_f1_gain_improvement",
                "shift_mean_f1_gain_delta",
            )
        )

    selection_rates = {
        key: (
            float(by_key[(key, primary)]["aligned_selection_rate"])
            if by_key[(key, primary)]["aligned_selection_rate"]
            is not None
            else None
        )
        for key in [clean_key, *shift_keys]
    }
    result: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "primary_method": primary,
        "validation_used_once_for_fixed_protocol": True,
        "test_split_untouched": True,
        "gates": gates,
        "aligned_selection_rate_reported_not_gated": selection_rates,
    }
    if improvement is not None:
        result["improvement_gates"] = improvement
        result["improvement_passed"] = improvement_passed
        # "Not improved" splits into two very different outcomes and the
        # label must say which: a genuinely flat expansion, versus one
        # that clears the absolute gates while REGRESSING against the
        # paired baseline. Both keep P7-3, so the routing is identical --
        # but a decision file that says "neutral" when the deltas are
        # negative is the same label-versus-content defect recorded as
        # F7 in main record 73.3.
        regressed = improvement is not None and (
            float(improvement["shift8_f1_gain_improvement"]["value"]) < 0.0
            or float(improvement["shift_mean_f1_gain_delta"]["value"]) < 0.0
        )
        result["regressed_against_baseline"] = bool(regressed)
        if passed and improvement_passed:
            result["next_action"] = "p7_5_phase_b_three_seeds"
        elif passed and regressed:
            result["next_action"] = (
                "p7_5_expansion_regressed_keep_p7_3_as_final"
            )
        elif passed:
            result["next_action"] = (
                "p7_5_expansion_neutral_keep_p7_3_as_final"
            )
        else:
            result["next_action"] = (
                "p7_5_expansion_harmful_discard_keep_p7_3"
            )
    else:
        result["next_action"] = (
            "freeze_structure_then_multi_seed_and_s2looking_transfer"
            if passed
            else "diagnose_safe_classifier_generalization_without_val_tuning"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train risk-controlled safe-alignment veto classifiers on "
            "frozen P7-2 training features and evaluate the fixed "
            "protocol once on LEVIR validation."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument(
        "--tile-features-csv",
        default=None,
        help=(
            "CSV with per-(id, scenario) tile-consistency features "
            "(the P7-5 screening's joined_training_rows.csv). Required "
            "when the protocol enables tile_features."
        ),
    )
    parser.add_argument(
        "--baseline-scenario-metrics",
        default=None,
        help=(
            "validation_scenario_metrics.csv of the same-seed P7-3 "
            "baseline run. Required when the protocol declares "
            "improvement_gates."
        ),
    )
    parser.add_argument(
        "--protocol",
        default="configs/p7_safe_alignment_classifier.yaml",
    )
    parser.add_argument("--task-router-checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    features_csv = Path(args.features_csv).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError(
            "Safe-alignment classifier protocol must be a mapping"
        )
    if args.seed is not None:
        protocol["seed"] = int(args.seed)

    features_sha256 = _sha256_of_file(features_csv)
    print(f"features_csv={features_csv}")
    print(f"features_csv_sha256={features_sha256}")

    tile_config = _tile_feature_config(protocol)
    feature_universe = _feature_universe(tile_config)
    rows = _load_feature_rows(features_csv)
    if tile_config is not None:
        if args.tile_features_csv is None:
            raise ValueError(
                "Protocol enables tile_features but no "
                "--tile-features-csv was provided"
            )
        tile_csv = Path(args.tile_features_csv).expanduser().resolve()
        tile_sha256 = _sha256_of_file(tile_csv)
        print(f"tile_features_csv={tile_csv}")
        print(f"tile_features_csv_sha256={tile_sha256}")
        _join_tile_features(rows, tile_csv, tile_config["names"])
    prepared = _prepare_training_tensors(
        rows,
        protocol=protocol,
        feature_universe=feature_universe,
    )
    split_rows = [
        {
            "id": row["id"],
            "scenario": row["scenario"],
            "split": prepared["splits"][index],
            "safe_label": bool(prepared["safe"][index]),
            "sample_weight": float(prepared["weight"][index]),
            "aligned_loss_advantage": float(
                prepared["benefit"][index]
            ),
            "existing_q_selected_aligned": bool(
                prepared["q_gate"][index]
            ),
        }
        for index, row in enumerate(rows)
    ]

    registration_set = SAFE_REGISTRATION_FEATURE_NAMES
    task_set = SAFE_TASK_FEATURE_NAMES
    if tile_config is not None:
        registration_set = (
            *SAFE_REGISTRATION_FEATURE_NAMES,
            *tile_config["names"],
        )
        task_set = (*SAFE_TASK_FEATURE_NAMES, *tile_config["names"])
    classifiers = {
        "safe_alignment_registration": train_safe_alignment_classifier(
            "safe_alignment_registration",
            rows,
            prepared,
            registration_set,
            protocol=protocol,
            output_dir=output_dir,
            feature_universe=feature_universe,
        ),
        "safe_alignment_task": train_safe_alignment_classifier(
            "safe_alignment_task",
            rows,
            prepared,
            task_set,
            protocol=protocol,
            output_dir=output_dir,
            feature_universe=feature_universe,
        ),
    }
    for name, bundle in classifiers.items():
        for index in range(len(split_rows)):
            split_rows[index][f"{name}_probability"] = float(
                bundle["probability"][index]
            )
    _write_csv(
        output_dir / "training_split_assignments.csv",
        split_rows,
    )

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

    task_router_bundle = None
    if args.task_router_checkpoint is not None:
        task_router_bundle = _load_task_router_bundle(
            Path(args.task_router_checkpoint).expanduser().resolve(),
            device,
        )

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
        classifiers,
        task_router_bundle,
        protocol=protocol,
        config=config,
        device=device,
        output_dir=output_dir,
        tile_config=tile_config,
    )
    baseline_rows = None
    if args.baseline_scenario_metrics is not None:
        baseline_path = (
            Path(args.baseline_scenario_metrics).expanduser().resolve()
        )
        with baseline_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            baseline_rows = list(csv.DictReader(handle))
        print(f"baseline_scenario_metrics={baseline_path}")
    decision = _decision(
        validation_rows,
        protocol=protocol,
        baseline_rows=baseline_rows,
    )
    results = {
        "status": "completed",
        "source_checkpoint": str(checkpoint_path),
        "features_csv": str(features_csv),
        "features_csv_sha256": features_sha256,
        "task_router_checkpoint": (
            str(args.task_router_checkpoint)
            if args.task_router_checkpoint is not None
            else None
        ),
        "protocol": str(protocol_path),
        "classifier_summaries": {
            name: bundle["summary"]
            for name, bundle in classifiers.items()
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
    print("===== P7-3 SAFE-ALIGNMENT CLASSIFIER DECISION =====")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(
        "Saved safe-alignment classifier experiment to "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
