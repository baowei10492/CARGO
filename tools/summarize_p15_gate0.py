from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


"""Route the P15 Gate 0 label-free screening decision.

Two hygiene runs feed this tool: the candidate dataset and a
geometry-matched LEVIR control. The control is mandatory, because the
dispersion statistic is not comparable across image and tile geometry.
Earlier records contain values from a different statistic/configuration
and must not be substituted for the control generated in the same run.
Main record 75.2 established the general rule that an anomaly statistic
must first be run on a known-good matched control before its value means
anything.

Clean-scenario dispersion is never used: the conditional localizer
cannot emit a zero offset, so on clean pairs it measures forced-choice
noise and is anti-correlated with the real misalignment (main record 65).
Only the clean *q trigger rate* is read from the clean scenario.
"""


CLEAN_SCENARIO = "clean:s0:common"
DISPERSION_KEY = "median_max_pairwise_tile_distance"
Q_RATE_KEY = "whole_image_q_trigger_rate"


def _injected_scenarios(summary: dict[str, Any]) -> dict[str, Any]:
    scenarios = summary["scenarios"]
    injected = {
        key: value
        for key, value in scenarios.items()
        if key != CLEAN_SCENARIO
    }
    if not injected:
        raise ValueError(
            "Gate 0 needs at least one injected-shift scenario; clean "
            "dispersion is not interpretable (main record 65)"
        )
    for key, value in injected.items():
        if bool(value.get("forced_choice_regime", False)):
            raise ValueError(
                f"Scenario {key} is in the forced-choice regime; its "
                "dispersion must not be used as evidence"
            )
    return injected


def mean_injected_dispersion(summary: dict[str, Any]) -> float:
    injected = _injected_scenarios(summary)
    values = [float(item[DISPERSION_KEY]) for item in injected.values()]
    return sum(values) / len(values)


def clean_q_trigger_rate(summary: dict[str, Any]) -> float:
    scenarios = summary["scenarios"]
    if CLEAN_SCENARIO not in scenarios:
        raise ValueError(
            "Gate 0 needs the clean scenario for the q trigger rate; "
            "run the hygiene diagnostic with severity 0 included"
        )
    record = scenarios[CLEAN_SCENARIO]
    if Q_RATE_KEY not in record:
        raise ValueError(
            "Clean scenario has no whole-image q trigger rate"
        )
    return float(record[Q_RATE_KEY])


def geometry_matches(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    def effective_size(summary: dict[str, Any]) -> int | None:
        crop = summary.get("center_crop")
        return int(crop) if crop is not None else None

    tiles_match = int(candidate["tile_size"]) == int(control["tile_size"])
    candidate_tiles = candidate["scenarios"]
    control_tiles = control["scenarios"]
    counts_candidate = sorted(
        {
            tuple(item.get("tiles_per_image", []))
            for item in candidate_tiles.values()
        }
    )
    counts_control = sorted(
        {
            tuple(item.get("tiles_per_image", []))
            for item in control_tiles.values()
        }
    )
    return {
        "passed": tiles_match and counts_candidate == counts_control,
        "candidate_tile_size": int(candidate["tile_size"]),
        "control_tile_size": int(control["tile_size"]),
        "candidate_center_crop": effective_size(candidate),
        "control_center_crop": effective_size(control),
        "candidate_tiles_per_image": [
            list(item) for item in counts_candidate
        ],
        "control_tiles_per_image": [
            list(item) for item in counts_control
        ],
    }


def gate0_decision(
    candidate: dict[str, Any],
    control: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["gate0"]
    geometry = geometry_matches(candidate, control)

    candidate_d = mean_injected_dispersion(candidate)
    control_d = mean_injected_dispersion(control)
    excess_d = candidate_d - control_d
    candidate_q = clean_q_trigger_rate(candidate)
    control_q = clean_q_trigger_rate(control)

    dispersion_green = excess_d <= float(
        thresholds["dispersion_excess_green_max"]
    )
    dispersion_red = excess_d >= float(
        thresholds["dispersion_excess_red_min"]
    )
    q_green = candidate_q <= float(thresholds["clean_q_green_max"])
    q_red = candidate_q >= float(thresholds["clean_q_red_min"])

    if dispersion_green and q_green:
        branch = "green"
        action = "proceed_to_gate1"
    elif dispersion_red and q_red:
        branch = "red"
        action = "stop_loss_record_as_spatially_varying_field"
    else:
        branch = "yellow"
        action = "proceed_to_gate1"

    return {
        "status": "completed",
        "dataset": config.get("dataset"),
        "geometry_match": geometry,
        "signals": {
            "mean_injected_dispersion": {
                "candidate": candidate_d,
                "control": control_d,
                "excess": excess_d,
                "green_max": float(
                    thresholds["dispersion_excess_green_max"]
                ),
                "red_min": float(
                    thresholds["dispersion_excess_red_min"]
                ),
                "green": dispersion_green,
                "red": dispersion_red,
            },
            "clean_q_trigger_rate": {
                "candidate": candidate_q,
                "control": control_q,
                "green_max": float(thresholds["clean_q_green_max"]),
                "red_min": float(thresholds["clean_q_red_min"]),
                "green": q_green,
                "red": q_red,
            },
        },
        "branch": branch,
        "action": action,
        "labels_read": False,
        "test_split_touched": False,
        "notes": [
            "Dispersion is comparable only against the geometry-matched "
            "control produced in the same run; absolute values move with "
            "image size, tile size, and statistic definition.",
            "Clean-scenario dispersion is excluded by protocol; only "
            "the clean q trigger rate is read from it (main record 65).",
            "Ambiguous evidence routes to the decisive criterion rather "
            "than to stop-loss, which is what prevented the wrong WHU "
            "stop in main record 71.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the preregistered P15 Gate 0 routing to a candidate "
            "dataset and its geometry-matched LEVIR control."
        )
    )
    parser.add_argument("--candidate-hygiene", required=True)
    parser.add_argument("--control-hygiene", required=True)
    parser.add_argument(
        "--config", default="configs/p15_dsifn_screening.yaml"
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    candidate = json.loads(
        Path(args.candidate_hygiene).read_text(encoding="utf-8")
    )
    control = json.loads(
        Path(args.control_hygiene).read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )
    decision = gate0_decision(candidate, control, config=config)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    if not decision["geometry_match"]["passed"]:
        raise SystemExit(
            "Gate 0 geometry control mismatch: the candidate and control "
            "runs are not comparable"
        )


if __name__ == "__main__":
    main()
