from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from rs_cd.dual_path import SAFE_REGISTRATION_FEATURE_NAMES
from rs_cd.runtime import save_json
from tools.diagnose_registration_hygiene import WHOLE_IMAGE_TILE_INDEX
from tools.train_evaluate_task_benefit_router import _write_csv


# Candidate features derived from independent tile estimates. Every one is
# computable at deployment: it uses only the estimator's own outputs, no
# change label and no knowledge of the injected offset.
TILE_FEATURE_NAMES = (
    "tile_max_pairwise_distance",
    "tile_dispersion_radius",
    "tile_agreement_within_2px",
    "tile_agreement_within_4px",
    "tile_mean_offset_magnitude",
    "tile_magnitude_spread",
)


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney U AUROC with tie handling. None if a class is empty."""

    count = len(scores)
    if count != len(labels) or count == 0:
        raise ValueError("scores and labels must be equal, non-empty")
    positives = sum(labels)
    negatives = count - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(count), key=lambda index: scores[index])
    ranks = [0.0] * count
    position = 0
    while position < count:
        end = position
        while (
            end + 1 < count
            and scores[order[end + 1]] == scores[order[position]]
        ):
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def directional_auc(
    scores: list[float],
    labels: list[bool],
) -> tuple[float | None, int]:
    """Return (|AUROC - 0.5| + 0.5, direction) so sign does not matter.

    A monotone-but-inverted feature is equally usable by a downstream
    classifier, so screening uses the direction-free magnitude and
    records which way it points (+1 higher-is-safe, -1 higher-is-unsafe).
    """

    raw = roc_auc(scores, labels)
    if raw is None:
        return None, 0
    if raw >= 0.5:
        return raw, 1
    return 1.0 - raw, -1


def _distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tile_features_by_key(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Reduce raw tile records into one candidate feature row per sample."""

    grouped: dict[
        tuple[str, str], dict[int, tuple[float, float]]
    ] = defaultdict(dict)
    for record in records:
        key = (str(record["id"]), str(record["scenario"]))
        grouped[key][int(record["tile_index"])] = (
            float(record["est_dx"]),
            float(record["est_dy"]),
        )
    features: dict[tuple[str, str], dict[str, float]] = {}
    for key, entries in grouped.items():
        tiles = [
            offset
            for index, offset in entries.items()
            if index != WHOLE_IMAGE_TILE_INDEX
        ]
        if len(tiles) < 2:
            continue
        whole = entries.get(WHOLE_IMAGE_TILE_INDEX)
        widest = max(
            _distance(tiles[first], tiles[second])
            for first in range(len(tiles))
            for second in range(first + 1, len(tiles))
        )
        centre = (
            sum(tile[0] for tile in tiles) / len(tiles),
            sum(tile[1] for tile in tiles) / len(tiles),
        )
        radius = sum(
            _distance(tile, centre) for tile in tiles
        ) / len(tiles)
        magnitudes = [_distance(tile, (0.0, 0.0)) for tile in tiles]
        reference = whole if whole is not None else centre
        features[key] = {
            "tile_max_pairwise_distance": widest,
            "tile_dispersion_radius": radius,
            "tile_agreement_within_2px": sum(
                1 for tile in tiles if _distance(tile, reference) <= 2.0
            )
            / len(tiles),
            "tile_agreement_within_4px": sum(
                1 for tile in tiles if _distance(tile, reference) <= 4.0
            )
            / len(tiles),
            "tile_mean_offset_magnitude": sum(magnitudes)
            / len(magnitudes),
            "tile_magnitude_spread": max(magnitudes) - min(magnitudes),
        }
    return features


def screen_features(
    tile_features: dict[tuple[str, str], dict[str, float]],
    training_rows: list[dict[str, Any]],
    *,
    benefit_margin: float,
    minimum_auc: float,
) -> dict[str, Any]:
    """Apply the preregistered inclusion rule on training data only."""

    joined: list[dict[str, Any]] = []
    missing = 0
    for row in training_rows:
        key = (str(row["id"]), str(row["scenario"]))
        candidate = tile_features.get(key)
        if candidate is None:
            missing += 1
            continue
        entry: dict[str, Any] = {
            "id": key[0],
            "scenario": key[1],
            "aligned_loss_advantage": float(
                row["aligned_loss_advantage"]
            ),
            "safe_label": float(row["aligned_loss_advantage"])
            > benefit_margin,
        }
        for name in SAFE_REGISTRATION_FEATURE_NAMES:
            entry[name] = float(row[name])
        entry.update(candidate)
        joined.append(entry)
    if not joined:
        raise RuntimeError("Join produced no rows; check ids/scenarios")

    labels = [bool(row["safe_label"]) for row in joined]
    scenarios = sorted({str(row["scenario"]) for row in joined})

    def auc_table(name: str) -> dict[str, Any]:
        pooled, direction = directional_auc(
            [float(row[name]) for row in joined],
            labels,
        )
        per_scenario: dict[str, float | None] = {}
        for scenario in scenarios:
            subset = [
                row for row in joined if row["scenario"] == scenario
            ]
            value, _ = directional_auc(
                [float(row[name]) for row in subset],
                [bool(row["safe_label"]) for row in subset],
            )
            per_scenario[scenario] = value
        return {
            "pooled_directional_auc": pooled,
            "direction": direction,
            "per_scenario_directional_auc": per_scenario,
        }

    incumbents = {
        name: auc_table(name)
        for name in SAFE_REGISTRATION_FEATURE_NAMES
    }
    candidates = {name: auc_table(name) for name in TILE_FEATURE_NAMES}
    incumbent_values = sorted(
        table["pooled_directional_auc"]
        for table in incumbents.values()
        if table["pooled_directional_auc"] is not None
    )
    middle = len(incumbent_values) // 2
    incumbent_median = (
        incumbent_values[middle]
        if len(incumbent_values) % 2
        else 0.5
        * (incumbent_values[middle - 1] + incumbent_values[middle])
    )
    bar = max(float(minimum_auc), incumbent_median)
    accepted = sorted(
        name
        for name, table in candidates.items()
        if table["pooled_directional_auc"] is not None
        and table["pooled_directional_auc"] >= bar
    )
    return {
        "status": "completed",
        "rows_joined": len(joined),
        "rows_unmatched": missing,
        "scenarios": scenarios,
        "safe_rate": sum(labels) / len(labels),
        "benefit_margin": float(benefit_margin),
        "inclusion_rule": (
            "pooled directional AUROC on the LEVIR training split must "
            "reach max(minimum_auc, median pooled directional AUROC of "
            "the ten incumbent registration features)"
        ),
        "minimum_auc": float(minimum_auc),
        "incumbent_median_auc": incumbent_median,
        "inclusion_bar": bar,
        "incumbent_features": incumbents,
        "candidate_features": candidates,
        "accepted_candidates": accepted,
        "next_action": (
            "p7_5_add_accepted_tile_features_new_protocol"
            if accepted
            else "p7_5_drop_tile_consistency_use_other_evidence"
        ),
        "joined_rows_written": True,
    }, joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen tile-consistency candidate features for P7-5 using "
            "the LEVIR training split only. Joins tile offset records "
            "with the frozen P7-2 training feature table and compares "
            "each candidate's directional AUROC against the incumbent "
            "registration features. Touches no validation data."
        )
    )
    parser.add_argument("--tile-records", required=True)
    parser.add_argument("--training-features", required=True)
    parser.add_argument(
        "--benefit-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument("--minimum-auc", type=float, default=0.60)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with Path(args.tile_records).expanduser().open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        tile_records = list(csv.DictReader(handle))
    with Path(args.training_features).expanduser().open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        training_rows = list(csv.DictReader(handle))
    print(f"tile_records={len(tile_records)}")
    print(f"training_rows={len(training_rows)}")

    decision, joined = screen_features(
        tile_features_by_key(tile_records),
        training_rows,
        benefit_margin=float(args.benefit_margin),
        minimum_auc=float(args.minimum_auc),
    )
    _write_csv(output_dir / "joined_training_rows.csv", joined)
    save_json(output_dir / "decision.json", decision)
    print("===== P7-5 TILE FEATURE SCREENING =====")
    printable = {
        key: value
        for key, value in decision.items()
        if key not in {"incumbent_features", "candidate_features"}
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print("--- candidate pooled directional AUROC ---")
    for name, table in decision["candidate_features"].items():
        value = table["pooled_directional_auc"]
        arrow = "higher=safe" if table["direction"] > 0 else "higher=unsafe"
        print(f"{name:34s} {value:.4f}  ({arrow})")
    print("--- incumbent pooled directional AUROC ---")
    for name, table in decision["incumbent_features"].items():
        print(f"{name:34s} {table['pooled_directional_auc']:.4f}")
    print(f"Saved screening to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
