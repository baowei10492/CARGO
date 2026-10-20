from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ConfusionCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        *,
        true_positive: int,
        false_positive: int,
        false_negative: int,
        true_negative: int,
    ) -> None:
        self.true_positive += int(true_positive)
        self.false_positive += int(false_positive)
        self.false_negative += int(false_negative)
        self.true_negative += int(true_negative)

    @property
    def valid_pixels(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    def compute(self) -> dict[str, float | int]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        change_f1 = (
            2.0 * precision * recall / (precision + recall + eps)
        )
        change_iou = tp / (tp + fp + fn + eps)
        background_precision = tn / (tn + fn + eps)
        background_recall = tn / (tn + fp + eps)
        background_f1 = (
            2.0
            * background_precision
            * background_recall
            / (background_precision + background_recall + eps)
        )
        background_iou = tn / (tn + fp + fn + eps)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "valid_pixels": self.valid_pixels,
            "change_precision": precision,
            "change_recall": recall,
            "change_f1": change_f1,
            "change_iou": change_iou,
            "background_f1": background_f1,
            "background_iou": background_iou,
            "mean_f1_author_compatibility": (
                change_f1 + background_f1
            )
            / 2.0,
            "mean_iou_author_compatibility": (
                change_iou + background_iou
            )
            / 2.0,
            "overall_accuracy": (tp + tn)
            / max(1, self.valid_pixels),
        }


def nonoverlapping_tiles(
    height: int,
    width: int,
    tile_size: int,
) -> list[tuple[int, int, int, int]]:
    if min(height, width, tile_size) < 1:
        raise ValueError("Image and tile dimensions must be positive")
    if height % tile_size or width % tile_size:
        raise ValueError(
            f"{height}x{width} is not divisible by tile size {tile_size}"
        )
    return [
        (top, left, top + tile_size, left + tile_size)
        for top in range(0, height, tile_size)
        for left in range(0, width, tile_size)
    ]


def stable_inference_seed(
    base_seed: int,
    sample_id: str,
    tile_start: int,
) -> int:
    payload = (
        f"{base_seed}|diffregcd-common-random|{sample_id}|{tile_start}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "little",
    ) % (2**31)


def support_digest(
    rows: Iterable[tuple[str, bytes]],
) -> str:
    digest = hashlib.sha256()
    for sample_id, packed_mask in rows:
        encoded_id = sample_id.encode("utf-8")
        digest.update(len(encoded_id).to_bytes(4, "little"))
        digest.update(encoded_id)
        digest.update(len(packed_mask).to_bytes(8, "little"))
        digest.update(packed_mask)
    return digest.hexdigest()


def same_support_decision(
    support: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not support:
        raise ValueError("Support table is empty")
    digests = {
        str(value["sha256"]) for value in support.values()
    }
    pixel_counts = {
        int(value["valid_pixels"]) for value in support.values()
    }
    passed = len(digests) == 1 and len(pixel_counts) == 1
    return {
        "passed": passed,
        "unique_support_sha256": sorted(digests),
        "unique_valid_pixel_counts": sorted(pixel_counts),
    }


def clean_reproduction_audit(
    scenario_rows: list[dict[str, Any]],
    author_reference: dict[str, Any],
) -> dict[str, Any]:
    clean = [
        row
        for row in scenario_rows
        if row["scenario"] == "clean:s0:common"
    ]
    expected_paths = {"official_unmodified", "intended_warped"}
    available_paths = {str(row["method"]) for row in clean}
    if available_paths != expected_paths:
        raise ValueError(
            "Clean audit requires official_unmodified and intended_warped"
        )
    reference = {
        "change_f1": float(author_reference["change_f1"]),
        "mean_f1_author_compatibility": float(
            author_reference["mean_f1"]
        ),
        "mean_iou_author_compatibility": float(
            author_reference["mean_iou"]
        ),
        "overall_accuracy": float(
            author_reference["overall_accuracy"]
        ),
    }
    return {
        "status": "manual_review_required",
        "reason": str(
            author_reference["evaluation_distribution_note"]
        ),
        "author_reference": reference,
        "paths": {
            str(row["method"]): {
                metric: {
                    "observed": float(row[metric]),
                    "author_reference": target,
                    "delta": float(row[metric]) - target,
                }
                for metric, target in reference.items()
            }
            for row in clean
        },
        "used_for_path_selection": False,
    }


def official_path_freeze_decision(
    validation_result: dict[str, Any],
    *,
    metric_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Decide whether the public author inference path may be frozen.

    The decision uses only clean reproduction and engineering provenance.
    Shift performance is deliberately excluded to prevent selecting the
    external implementation variant that happens to favour our benchmark.
    """
    if metric_tolerance <= 0:
        raise ValueError("metric_tolerance must be positive")
    decision = validation_result.get("decision", {})
    support_gate = validation_result.get("support_gate", {})
    clean_audit = validation_result.get(
        "clean_reproduction_audit",
        {},
    )
    official = clean_audit.get("paths", {}).get(
        "official_unmodified",
        {},
    )
    required_metrics = (
        "change_f1",
        "mean_f1_author_compatibility",
        "mean_iou_author_compatibility",
        "overall_accuracy",
    )
    metric_gates: dict[str, dict[str, Any]] = {}
    for metric in required_metrics:
        record = official.get(metric)
        if not isinstance(record, dict) or "delta" not in record:
            metric_gates[metric] = {
                "passed": False,
                "reason": "missing_clean_reproduction_metric",
            }
            continue
        delta = float(record["delta"])
        metric_gates[metric] = {
            "delta": delta,
            "absolute_delta": abs(delta),
            "maximum_absolute_delta": metric_tolerance,
            "passed": abs(delta) <= metric_tolerance,
        }

    engineering_passed = bool(
        decision.get("engineering_gates_passed", False)
    )
    same_support_passed = bool(support_gate.get("passed", False))
    clean_passed = all(
        bool(record["passed"]) for record in metric_gates.values()
    )
    passed = engineering_passed and same_support_passed and clean_passed
    return {
        "status": "passed" if passed else "failed",
        "selected_external_path": (
            "official_unmodified" if passed else None
        ),
        "selection_basis": (
            "public_author_test_wiring_and_clean_reproduction_only"
        ),
        "shift_metrics_used_for_path_selection": False,
        "engineering_gates_passed": engineering_passed,
        "same_support_passed": same_support_passed,
        "clean_metric_gates": metric_gates,
        "test_authorized": passed,
        "next_action": (
            "freeze_and_run_one_shot_external_test"
            if passed
            else "do_not_consume_external_test"
        ),
    }


def external_test_summary(
    scenario_rows: list[dict[str, Any]],
    *,
    method: str = "official_unmodified",
) -> dict[str, Any]:
    table = {
        (str(row["scenario"]), str(row["method"])): row
        for row in scenario_rows
    }
    scenarios = (
        "clean:s0:common",
        "shift:s1:t2",
        "shift:s2:t2",
        "shift:s3:t2",
    )
    missing = [
        (scenario, method)
        for scenario in scenarios
        if (scenario, method) not in table
    ]
    if missing:
        raise ValueError(
            f"External test rows lack required entries: {missing}"
        )
    return {
        "status": "completed",
        "reporting_only_no_selection": True,
        "external_method": "DiffRegCD",
        "inference_path": method,
        "test_result_used_for_model_selection": False,
        "scenario_change_f1": {
            scenario: float(table[(scenario, method)]["change_f1"])
            for scenario in scenarios
        },
    }
