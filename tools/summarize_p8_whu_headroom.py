from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


CLEAN_KEY = "clean:s0:common"


def _next_action(dataset: str, passed: bool) -> str:
    """Return the preregistered route without changing historical labels."""

    normalized = dataset.strip().lower().replace("_", "-")
    if normalized in {"whu", "whu-cd"}:
        return (
            "p8_continue_chain_dual_path_oracle_then_p7_1b"
            if passed
            else "p8_stop_whu_joins_domain_boundary_cases"
        )
    if normalized in {"dsifn", "dsifn-cd"}:
        return (
            "p15_proceed_to_gate2_full_chain"
            if passed
            else "p15_stop_dsifn_input_level_not_recoverable"
        )
    return (
        "proceed_to_full_chain"
        if passed
        else "stop_input_level_not_recoverable"
    )


def _f1_by_scenario(rows: list[dict[str, str]]) -> dict[str, float]:
    table: dict[str, float] = {}
    for row in rows:
        key = str(row["scenario"])
        if key in table:
            raise ValueError(f"Duplicate scenario in results: {key}")
        table[key] = float(row["f1"])
    if CLEAN_KEY not in table:
        raise ValueError("Results are missing the clean scenario")
    return table


def _valid_pixels_by_scenario(
    rows: list[dict[str, str]],
) -> dict[str, int]:
    table: dict[str, int] = {}
    for row in rows:
        key = str(row["scenario"])
        if "valid_pixels" not in row or row["valid_pixels"] in ("", None):
            raise ValueError(
                "Headroom comparison requires valid_pixels for every "
                f"scenario; missing in {key}"
            )
        table[key] = int(row["valid_pixels"])
    return table


def headroom_decision(
    raw_rows: list[dict[str, str]],
    oracle_rows: list[dict[str, str]],
    *,
    shift8_headroom_min: float,
    clean_agreement_tolerance: float = 1e-6,
    dataset: str = "WHU-CD",
) -> dict[str, Any]:
    """Apply the preregistered decisive gate for a new dataset.

    The input-level oracle inverse shift needs only the detector and the
    known injected offset, so this quantity is immune to every
    registration-side instrument problem (clamping pseudo-hits, forced
    choice, tile size, threshold transfer). It answers the only question
    that matters before investing in the routing chain: does this dataset
    contain recoverable misalignment at all?
    """

    raw = _f1_by_scenario(raw_rows)
    oracle = _f1_by_scenario(oracle_rows)
    raw_support = _valid_pixels_by_scenario(raw_rows)
    oracle_support = _valid_pixels_by_scenario(oracle_rows)
    if raw_support != oracle_support:
        mismatched = {
            key: {
                "raw_valid_pixels": raw_support.get(key),
                "oracle_valid_pixels": oracle_support.get(key),
            }
            for key in sorted(set(raw_support) | set(oracle_support))
            if raw_support.get(key) != oracle_support.get(key)
        }
        raise ValueError(
            "Raw and oracle runs use different valid-pixel support; "
            f"F1 headroom is not paired: {mismatched}"
        )
    clean_delta = abs(raw[CLEAN_KEY] - oracle[CLEAN_KEY])
    if clean_delta > clean_agreement_tolerance:
        raise ValueError(
            "Clean F1 differs between the raw and oracle runs "
            f"({raw[CLEAN_KEY]} vs {oracle[CLEAN_KEY]}); the oracle flag "
            "must not affect the clean scenario"
        )
    shift_keys = sorted(key for key in raw if key.startswith("shift:"))
    if not shift_keys:
        raise ValueError("Results contain no shift scenarios")
    missing = [key for key in shift_keys if key not in oracle]
    if missing:
        raise ValueError(f"Oracle results lack scenarios: {missing}")

    headroom = {
        key: oracle[key] - raw[key] for key in shift_keys
    }
    shift8_key = next(
        (key for key in shift_keys if key.endswith("s3:t2")),
        None,
    )
    if shift8_key is None:
        raise ValueError("Could not find the severity-3 shift scenario")
    value = headroom[shift8_key]
    gate = {
        "shift8_input_oracle_headroom": {
            "value": value,
            "minimum": float(shift8_headroom_min),
            "passed": value >= float(shift8_headroom_min),
        }
    }
    passed = bool(gate["shift8_input_oracle_headroom"]["passed"])
    return {
        "status": "completed",
        "dataset": dataset,
        "clean_f1": raw[CLEAN_KEY],
        "raw_f1": {key: raw[key] for key in shift_keys},
        "oracle_f1": {key: oracle[key] for key in shift_keys},
        "input_oracle_headroom": headroom,
        "paired_valid_pixels": {
            key: raw_support[key] for key in sorted(raw_support)
        },
        "gates": gate,
        "decisive_gate_passed": passed,
        "next_action": _next_action(dataset, passed),
        "interpretation_constraints": [
            "Input-oracle headroom tests recoverability of an injected "
            "global translation; it is not sensitive to native "
            "misregistration.",
            "Passing this gate neither proves nor disproves that native "
            "misregistration is spatially varying. Gate 0 clean-q and "
            "Gate 2 end-to-end results resolve that ambiguity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the input-level oracle headroom on a new dataset "
            "and apply the preregistered decisive gate."
        )
    )
    parser.add_argument("--raw", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--train-summary", default=None)
    parser.add_argument(
        "--dataset",
        default="WHU-CD",
        help=(
            "Dataset label used only for reporting and preregistered "
            "next-action naming. The numerical gate is unchanged."
        ),
    )
    parser.add_argument(
        "--shift8-headroom-min",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw_path = Path(args.raw).expanduser().resolve()
    oracle_path = Path(args.oracle).expanduser().resolve()
    if raw_path == oracle_path:
        raise ValueError("--raw and --oracle must be distinct result files")

    def read(path: str) -> list[dict[str, str]]:
        with Path(path).expanduser().open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            return list(csv.DictReader(handle))

    decision = headroom_decision(
        read(str(raw_path)),
        read(str(oracle_path)),
        shift8_headroom_min=float(args.shift8_headroom_min),
        dataset=str(args.dataset),
    )
    decision["raw_results"] = str(raw_path)
    decision["oracle_results"] = str(oracle_path)
    if args.train_summary:
        summary = json.loads(
            Path(args.train_summary)
            .expanduser()
            .read_text(encoding="utf-8")
        )
        decision["detector_best_epoch"] = summary.get("best_epoch")
        decision["detector_best_val_f1"] = summary.get("best_f1")
    save_json(args.output, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
