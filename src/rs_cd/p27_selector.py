"""P27 selector step (docs/P27_SELECTOR_AMENDMENT.md).

Label-free features per crop and level, the per-crop record shared by
training and validation, the P7-3 sample weights, threshold calibration and
the path rules. The per-crop detector batch is built exactly as the F1 step
builds it, so the validation run can reproduce the F1 table.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from rs_cd.affine_f1 import DETECTOR_PATHS, add_counts, confusion_counts, path_images, scoring_mask
from rs_cd.affine_registration import CLEAN_LEVEL, AffineRanges, compose, mean_displacement, perturb, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.dual_path import ROUTER_FEATURE_NAMES, SafeAlignmentClassifier, dual_path_router_features, per_sample_binary_change_loss
from rs_cd.localizer_training import TrainedLocalizer, step1_rows
from rs_cd.metrics.binary import BinaryChangeMetrics

REGISTRATION_FEATURES = (
    "misaligned_probability",
    "estimate_displacement_px",
    "log_scale_magnitude",
    "centre_shift_px",
    "forward_backward_px",
    "aligned_valid_fraction",
)
TASK_FEATURES = tuple(ROUTER_FEATURE_NAMES[12:])
FEATURE_NAMES = REGISTRATION_FEATURES + TASK_FEATURES
COUNT_KEYS = ("tp", "fp", "fn", "tn")
RULES = ("raw", "oracle", "estimated", "gated", "selector", "selector_veto", "loss_oracle")
_REGISTRATION_KEYS = (
    "registration_shift_probability", "registration_zero_probability", "registration_confidence",
    "registration_margin", "registration_local_margin", "registration_reliability",
)


def scenario(level: int) -> str:
    return "clean" if level == CLEAN_LEVEL else f"level{level}"


def matrix_of(row: dict[str, Any]) -> np.ndarray:
    return np.array([[row["a11"], row["a12"], row["a13"]], [row["a21"], row["a22"], row["a23"]]], dtype=np.float64)


def task_features(raw_logits: torch.Tensor, aligned_logits: torch.Tensor, valid: torch.Tensor,
                  threshold: float) -> torch.Tensor:
    """P7's 13 raw / aligned disagreement columns, computed by the P7 function itself."""

    batch = raw_logits.shape[0]
    registration = {key: raw_logits.new_zeros(batch) for key in _REGISTRATION_KEYS}
    registration["conditional_hard_offset_pixels"] = raw_logits.new_zeros(batch, 2)
    features = dual_path_router_features(raw_logits, aligned_logits, valid, registration,
                                         maximum_offset_pixels=1.0, threshold=threshold)
    return features[:, 12:]


def registration_features(matrix: np.ndarray, back: np.ndarray, probability: float, points: np.ndarray,
                          size: int) -> dict[str, float]:
    centre = np.array([[(size - 1) / 2.0, (size - 1) / 2.0]])
    return {
        "misaligned_probability": float(probability),
        "estimate_displacement_px": mean_displacement(matrix, points),
        "log_scale_magnitude": abs(math.log(max(float(matrix[0, 0]), 1e-6))),
        "centre_shift_px": mean_displacement(matrix, centre),
        # T1 -> input T2 -> T1 on the pair the detector will see.
        "forward_backward_px": mean_displacement(compose(back, matrix), points),
        "aligned_valid_fraction": float(scoring_mask([matrix], size).mean()),
    }


def selector_split(sample_id: str, *, holdout: bool, seed: int, early_stop_fraction: float) -> str:
    if holdout:
        return "risk_calibration"
    digest = hashlib.sha256(f"{seed}|p27-selector-split|{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64)
    return "early_stop" if value < early_stop_fraction else "fit"


@torch.no_grad()
def crop_records(
    localizer: TrainedLocalizer,
    detector: torch.nn.Module,
    *,
    threshold: float,
    loss_weights: dict[str, float],
    dataset: str,
    item: dict[str, Any],
    t1: np.ndarray,
    t2: np.ndarray,
    mask: np.ndarray,
    mask_threshold: int,
    ranges: AffineRanges,
    levels: Sequence[int],
    seed: int,
    points: np.ndarray,
    tolerance: float,
    trigger_probability: float,
    device: torch.device,
    geometry: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(localizer rows, one record per level) for one crop.

    ``geometry`` (P30, rs_cd.p30_geometry.BorderFreeGeometry) makes the perturbed and the
    corrected T2 from the whole tile; without it the crop is perturbed as in P27.
    """

    size = t1.shape[0]
    truth_mask = mask > int(mask_threshold)
    change_fraction = float(truth_mask.mean())
    rows = step1_rows(
        localizer, dataset=dataset, crop_id=item["crop_id"], t1=t1, t2=t2, change_fraction=change_fraction,
        ranges=ranges, levels=levels, seed=seed, points=points, tolerance=tolerance,
        trigger_probability=trigger_probability,
        make_target=None if geometry is None else geometry.perturb,
    )
    truths = [perturbation_matrix(float(r["angle"]), float(r["scale"]), float(r["tx"]), float(r["ty"]),
                                  width=size, height=size) for r in rows]
    estimates = [matrix_of(r) for r in rows]
    valid = scoring_mask([m for r, m in zip(rows, truths) if int(r["level"]) != CLEAN_LEVEL], size)
    if geometry is None:
        perturbed = [t2 if int(r["level"]) == CLEAN_LEVEL else perturb(t2, m) for r, m in zip(rows, truths)]
        views_of = path_images
    else:
        perturbed = [t2 if int(r["level"]) == CLEAN_LEVEL else geometry.perturb(m) for r, m in zip(rows, truths)]
        views_of = geometry.path_images
    back = localizer.estimate_many(perturbed, [t1] * len(perturbed))

    # Same batch, in the same order, as the F1 step.
    images = []
    for image, truth, estimate in zip(perturbed, truths, estimates):
        images.extend(normalize(view) for view in views_of(image, truth, estimate).values())
    t2_batch = torch.stack(images).to(device)
    t1_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(device)
    logits = detector(t1_batch, t2_batch)["change_logits"]
    target = torch.from_numpy(truth_mask.astype(np.float32))[None, None].to(device)
    keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
    counts = confusion_counts(logits, target, keep, threshold)
    losses = per_sample_binary_change_loss(
        logits.float(), target.expand(logits.shape[0], -1, -1, -1), keep, **loss_weights
    ).cpu().tolist()

    step = len(DETECTOR_PATHS)
    raw_at = [k * step + DETECTOR_PATHS.index("raw") for k in range(len(rows))]
    estimated_at = [k * step + DETECTOR_PATHS.index("estimated") for k in range(len(rows))]
    inside = np.stack([scoring_mask([estimate], size) for estimate in estimates]).astype(np.float32)
    task = task_features(logits[raw_at].float(), logits[estimated_at].float(),
                         torch.from_numpy(inside)[:, None].to(device), threshold).cpu().tolist()
    records = []
    for k, row in enumerate(rows):
        level = int(row["level"])
        record: dict[str, Any] = {
            "dataset": dataset, "tile_id": item["sample_id"], "crop_id": item["crop_id"], "level": level,
            "scenario": scenario(level), "change_fraction": change_fraction, "trigger": bool(row["trigger"]),
            "localizer_success": row["success"], "valid_pixels": int(valid.sum()),
        }
        record.update(registration_features(estimates[k], back[k].matrix, row["misaligned_probability"], points, size))
        record.update(zip(TASK_FEATURES, task[k]))
        record["raw_loss"] = losses[raw_at[k]]
        record["estimated_loss"] = losses[estimated_at[k]]
        record["advantage"] = record["raw_loss"] - record["estimated_loss"]
        for j, path in enumerate(DETECTOR_PATHS):
            record.update({f"{path}_{key}": int(value) for key, value in zip(COUNT_KEYS, counts[k * step + j])})
        if geometry is not None:
            record["estimated_missing_px"] = geometry.missing_pixels(truths[k])
        records.append(record)
    return rows, records


# ------------------------------------------------------------- training
def sample_weights(advantage: torch.Tensor, clean: torch.Tensor, spec: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """(safe, weight) with the P7-3 formulas (tools.train_evaluate_safe_alignment_classifier)."""

    weights = spec["weights"]
    safe = advantage > float(spec["label"]["benefit_margin"])
    weight = (advantage.abs() / float(weights["magnitude_scale"])).clamp(
        min=float(weights["magnitude_floor"]), max=float(weights["magnitude_cap"]))
    weight = torch.where(safe, weight, weight * float(weights["unsafe_weight"]))
    weight = torch.where((~safe) & clean, weight * float(weights["clean_unsafe_extra_weight"]), weight)
    return safe, weight


def feature_matrix(records: Sequence[dict[str, Any]], names: Sequence[str] = FEATURE_NAMES) -> torch.Tensor:
    unknown = [name for name in names if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown selector features {unknown}")
    return torch.tensor([[float(record[name]) for name in names] for record in records], dtype=torch.float32)


@torch.no_grad()
def selector_probability(bundle: dict[str, Any], features: torch.Tensor) -> torch.Tensor:
    model = SafeAlignmentClassifier(len(bundle["feature_names"]), tuple(bundle["hidden_dims"]), float(bundle["dropout"]))
    model.load_state_dict(bundle["model_state"])
    model.eval()
    if features.shape[1] != len(bundle["feature_names"]):
        raise ValueError("feature matrix does not match the selector bundle")
    return torch.sigmoid(model((features - bundle["feature_mean"]) / bundle["feature_std"]))


# ------------------------------------------------------------ decisions
def choose(rule: str, record: dict[str, Any], probability: float, tau: float) -> str:
    if rule in DETECTOR_PATHS:
        return rule
    if rule == "gated":
        use = bool(record["trigger"])
    elif rule == "selector":
        use = probability >= tau
    elif rule == "selector_veto":
        use = bool(record["trigger"]) and probability >= tau
    elif rule == "loss_oracle":
        use = float(record["estimated_loss"]) < float(record["raw_loss"])
    else:
        raise ValueError(f"unknown rule {rule}")
    return "estimated" if use else "raw"


def pooled_f1(pairs: Iterable[tuple[dict[str, Any], str]]) -> float:
    """F1 over (record, chosen path) pairs, with BinaryChangeMetrics' formula."""

    metrics = BinaryChangeMetrics()
    for record, path in pairs:
        add_counts(metrics, [record[f"{path}_{key}"] for key in COUNT_KEYS])
    return float(metrics.compute()["f1"])


def f1_table(records: Sequence[dict[str, Any]], probabilities: Sequence[float], tau: float,
             rules: Sequence[str] = RULES) -> dict[str, dict[str, float]]:
    paired = list(zip(records, probabilities))
    table = {}
    for level in sorted({int(record["level"]) for record in records}):
        members = [(r, p) for r, p in paired if int(r["level"]) == level]
        table[str(level)] = {
            rule: pooled_f1((r, choose(rule, r, p, tau)) for r, p in members) for rule in rules
        }
    return table


def selection_rates(records: Sequence[dict[str, Any]], probabilities: Sequence[float], tau: float) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    for level in sorted({int(record["level"]) for record in records}):
        members = [(r, p) for r, p in zip(records, probabilities) if int(r["level"]) == level]
        rates[str(level)] = {
            rule: sum(choose(rule, r, p, tau) == "estimated" for r, p in members) / len(members)
            for rule in ("gated", "selector", "selector_veto", "loss_oracle")
        }
    return rates


def calibrate_tau(records: Sequence[dict[str, Any]], probabilities: Sequence[float], *, target: float,
                  start: float, stop: float, step: float) -> tuple[float, list[dict[str, Any]]]:
    """Smallest grid tau whose clean F1 drop is <= target; inf (never correct) if none."""

    clean = [(r, p) for r, p in zip(records, probabilities) if int(r["level"]) == CLEAN_LEVEL]
    if not clean:
        raise ValueError("no clean calibration rows")
    raw_f1 = pooled_f1((r, "raw") for r, _ in clean)
    count = int(round((stop - start) / step))
    chosen, table = math.inf, []
    for index in range(count + 1):
        tau = round(start + index * step, 10)
        decided = [(r, choose("selector", r, p, tau)) for r, p in clean]
        drop = raw_f1 - pooled_f1(decided)
        selected = sum(path == "estimated" for _, path in decided) / len(decided)
        table.append({"tau": tau, "clean_drop": drop, "clean_selected": selected})
        if math.isinf(chosen) and drop <= target:
            chosen = tau
    return chosen, table
