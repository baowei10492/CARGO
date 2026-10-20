from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the predeclared P7 Phase-0 decisions."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    geometry = _load(root / "geometry.json")
    feature = {
        "levir": _load(root / "levir_feature_oracle" / "results.json"),
        "s2looking": _load(root / "s2_feature_oracle" / "results.json"),
    }
    controls = {
        "levir": _load(root / "levir_phase_control" / "results.json"),
        "s2looking": _load(root / "s2_phase_control" / "results.json"),
    }
    geometry_passed = geometry.get("status") == "passed"
    feature_passed = {
        name: value.get("status") == "passed"
        for name, value in feature.items()
    }
    phase_solves = {
        name: bool(
            value.get("classical_decision", {})
            .get("phase", {})
            .get("solves_synthetic_global_translation", False)
        )
        for name, value in controls.items()
    }
    if not geometry_passed:
        next_action = "fix_geometry_or_coordinate_implementation"
    elif not all(feature_passed.values()):
        next_action = "fix_feature_warp_or_fusion_before_new_matcher"
    elif all(phase_solves.values()):
        next_action = (
            "skip_learned_global_translation_and_move_to_local_real_warps"
        )
    else:
        next_action = "begin_p7_1_robust_global_from_local_matcher"
    result = {
        "status": "completed",
        "run_root": str(root.resolve()),
        "geometry_passed": geometry_passed,
        "feature_path_passed": feature_passed,
        "phase_correlation_solves_global_translation": phase_solves,
        "next_action": next_action,
        "feature_path_gates": {
            name: value["gates"] for name, value in feature.items()
        },
        "phase_decisions": {
            name: value["classical_decision"]["phase"]
            for name, value in controls.items()
        },
    }
    save_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
