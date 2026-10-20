from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


"""Report how much of the available benefit a router actually captured.

The preregistered gates are stated as absolute F1 gains, and main record
120.4 showed that those numbers encode the loss scale of the dataset they
were calibrated on: where an injected shift costs little, the absolute
bar can sit above the oracle ceiling and no router can clear it.

This tool separates the two questions. The verdict against the frozen
absolute gates is reported unchanged, because that is what was
preregistered. Alongside it, the captured fraction of the oracle gain is
reported as a descriptive quantity, so that "the method is ineffective
here" can be distinguished from "this dataset has little to recover".
The capture rate defines no gate and changes no verdict.
"""


SHIFT_SCENARIOS = ("shift:s1:t2", "shift:s2:t2", "shift:s3:t2")

# Measured on the LEVIR-CD validation split, for context only. Derived
# from the recorded P7-3 gains over the dual-path oracle ceiling of that
# same split: Shift-8 0.1396/0.2141 and 0.1786/0.2141, mean 0.0821/0.1125
# and 0.0984/0.1125.
LEVIR_REFERENCE = {
    "seed3407": {"shift8_capture": 0.652, "mean_capture": 0.730},
    "seed1234": {"shift8_capture": 0.834, "mean_capture": 0.875},
}


def _f1(rows: list[dict[str, Any]], scenario: str, method: str) -> float:
    for row in rows:
        if str(row["scenario"]) == scenario and str(row["method"]) == method:
            return float(row["f1"])
    raise KeyError(f"Missing {scenario}/{method} in scenario metrics")


def capture_summary(
    rows: list[dict[str, Any]],
    *,
    primary_method: str,
    oracle_method: str = "loss_oracle",
    raw_method: str = "raw",
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for scenario in SHIFT_SCENARIOS:
        raw = _f1(rows, scenario, raw_method)
        primary = _f1(rows, scenario, primary_method)
        oracle = _f1(rows, scenario, oracle_method)
        available = oracle - raw
        captured = primary - raw
        scenarios[scenario] = {
            "raw_f1": raw,
            "primary_f1": primary,
            "oracle_f1": oracle,
            "available_gain": available,
            "captured_gain": captured,
            "capture_rate": (
                captured / available if available > 0 else None
            ),
        }
    mean_available = sum(
        item["available_gain"] for item in scenarios.values()
    ) / len(scenarios)
    mean_captured = sum(
        item["captured_gain"] for item in scenarios.values()
    ) / len(scenarios)
    return {
        "per_scenario": scenarios,
        "mean_available_gain": mean_available,
        "mean_captured_gain": mean_captured,
        "mean_capture_rate": (
            mean_captured / mean_available if mean_available > 0 else None
        ),
        "shift8_capture_rate": scenarios["shift:s3:t2"]["capture_rate"],
    }


def ceiling_analysis(
    capture: dict[str, Any],
    *,
    gates: dict[str, float],
) -> dict[str, Any]:
    """State which absolute gates were unreachable before the run."""
    mean_available = capture["mean_available_gain"]
    shift8_available = capture["per_scenario"]["shift:s3:t2"][
        "available_gain"
    ]
    mean_required = gates["shift_mean_f1_gain_min"]
    shift8_required = gates["shift8_f1_gain_min"]
    return {
        "mean_gain": {
            "required": mean_required,
            "oracle_ceiling": mean_available,
            "required_capture_rate": (
                mean_required / mean_available
                if mean_available > 0
                else None
            ),
            "reachable": mean_available >= mean_required,
        },
        "shift8_gain": {
            "required": shift8_required,
            "oracle_ceiling": shift8_available,
            "required_capture_rate": (
                shift8_required / shift8_available
                if shift8_available > 0
                else None
            ),
            "reachable": shift8_available >= shift8_required,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Separate router effectiveness from dataset loss scale by "
            "reporting the captured fraction of the oracle gain."
        )
    )
    parser.add_argument("--scenario-metrics", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    import csv

    import yaml

    with Path(args.scenario_metrics).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    decision = json.loads(
        Path(args.decision).read_text(encoding="utf-8")
    )
    protocol = yaml.safe_load(
        Path(args.protocol).read_text(encoding="utf-8")
    )
    primary = str(protocol["primary_method"])
    capture = capture_summary(rows, primary_method=primary)
    report = {
        "status": "completed",
        "dataset": args.dataset,
        "primary_method": primary,
        "preregistered_verdict": decision.get("status", decision.get(
            "decision", "see decision.json"
        )),
        "gates_from_protocol": protocol["gates"],
        "ceiling_analysis": ceiling_analysis(
            capture, gates=protocol["gates"]
        ),
        **capture,
        "levir_reference_capture": LEVIR_REFERENCE,
        "capture_rate_is_descriptive_only": True,
        "note": (
            "The preregistered verdict stands as recorded. The capture "
            "rate is reported so that an unreachable absolute gate is "
            "not misread as an ineffective method, and it must not be "
            "used to re-judge this or any completed run."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"{'scenario':>14s} {'raw':>8s} {'primary':>8s} {'oracle':>8s} "
          f"{'avail':>8s} {'capt':>8s} {'rate':>7s}")
    for scenario, item in capture["per_scenario"].items():
        rate = item["capture_rate"]
        print(
            f"{scenario:>14s} {item['raw_f1']:8.4f} "
            f"{item['primary_f1']:8.4f} {item['oracle_f1']:8.4f} "
            f"{item['available_gain']:8.4f} {item['captured_gain']:8.4f} "
            f"{'n/a' if rate is None else f'{rate:6.1%}'}"
        )
    mean_rate = capture["mean_capture_rate"]
    print(
        f"\nmean capture rate: "
        f"{'n/a' if mean_rate is None else f'{mean_rate:.1%}'}"
        f"   (LEVIR seed-3407 73.0%, seed-1234 87.5%)"
    )
    for name, item in report["ceiling_analysis"].items():
        print(
            f"{name}: required {item['required']:.3f} vs ceiling "
            f"{item['oracle_ceiling']:.4f} -> reachable "
            f"{item['reachable']}"
        )


if __name__ == "__main__":
    main()
