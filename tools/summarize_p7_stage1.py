from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


def _gate(
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, float | bool]:
    if (minimum is None) == (maximum is None):
        raise ValueError("Specify exactly one of minimum or maximum")
    passed = value >= minimum if minimum is not None else value <= maximum
    result: dict[str, float | bool] = {
        "value": float(value),
        "passed": bool(passed),
    }
    if minimum is not None:
        result["minimum"] = float(minimum)
    if maximum is not None:
        result["maximum"] = float(maximum)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply predeclared P7-1 estimator gates."
    )
    parser.add_argument("--registration-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.registration_results).expanduser().resolve()
    data: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    summary = data["summary_by_severity"]
    clean = summary["0"]
    shift4 = summary["2"]
    shift8 = summary["3"]
    gates = {
        "clean_hard_within_1px": _gate(
            float(clean["hard_within_1px_rate"]),
            minimum=0.95,
        ),
        "shift4_mean_hard_epe": _gate(
            float(shift4["mean_hard_epe_pixels"]),
            maximum=1.0,
        ),
        "shift4_hard_pck1": _gate(
            float(shift4["hard_within_1px_rate"]),
            minimum=0.80,
        ),
        "shift4_hard_pck2": _gate(
            float(shift4["hard_within_2px_rate"]),
            minimum=0.95,
        ),
        "shift8_mean_hard_epe": _gate(
            float(shift8["mean_hard_epe_pixels"]),
            maximum=1.0,
        ),
        "shift8_hard_pck1": _gate(
            float(shift8["hard_within_1px_rate"]),
            minimum=0.80,
        ),
        "shift8_hard_pck2": _gate(
            float(shift8["hard_within_2px_rate"]),
            minimum=0.95,
        ),
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    output = {
        "status": "completed",
        "registration_results": str(source),
        "estimator_passed": passed,
        "gates": gates,
        "next_action": (
            "evaluate_input_alignment"
            if passed
            else "diagnose_estimator_before_segmentation_tuning"
        ),
    }
    save_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
