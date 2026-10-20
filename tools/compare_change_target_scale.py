from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    for name, report in (("candidate", candidate), ("reference", reference)):
        if report.get("split") != "val":
            raise ValueError(f"{name} report is not validation-only")
        if report.get("test_split_touched"):
            raise ValueError(f"{name} report touched test")

    candidate_diameter = float(
        candidate["components"]["area_weighted_equivalent_diameter"][
            "median"
        ]
    )
    reference_diameter = float(
        reference["components"]["area_weighted_equivalent_diameter"][
            "median"
        ]
    )
    rows = {}
    for pixels in ("2", "4", "8"):
        candidate_iou = float(
            candidate["mask_self_iou_on_paired_support"][pixels]
        )
        reference_iou = float(
            reference["mask_self_iou_on_paired_support"][pixels]
        )
        candidate_boundary = float(
            candidate["boundary_band_fraction_of_change"][pixels]
        )
        reference_boundary = float(
            reference["boundary_band_fraction_of_change"][pixels]
        )
        rows[pixels] = {
            "candidate_self_iou": candidate_iou,
            "reference_self_iou": reference_iou,
            "self_iou_difference": candidate_iou - reference_iou,
            "candidate_boundary_fraction": candidate_boundary,
            "reference_boundary_fraction": reference_boundary,
            "boundary_fraction_difference": (
                candidate_boundary - reference_boundary
            ),
        }
    supports_large_target_hypothesis = (
        rows["8"]["self_iou_difference"] > 0
        and rows["8"]["boundary_fraction_difference"] < 0
        and candidate_diameter > reference_diameter
    )
    return {
        "status": "completed",
        "diagnostic_only_no_selection": True,
        "candidate": candidate["dataset"],
        "reference": reference["dataset"],
        "test_split_touched": False,
        "change_fraction": {
            "candidate": candidate["change_fraction"],
            "reference": reference["change_fraction"],
            "ratio": float(candidate["change_fraction"])
            / max(float(reference["change_fraction"]), 1e-12),
        },
        "area_weighted_median_equivalent_diameter_pixels": {
            "candidate": candidate_diameter,
            "reference": reference_diameter,
            "ratio": candidate_diameter / max(reference_diameter, 1e-12),
        },
        "shift_sensitivity": rows,
        "supports_larger_target_lower_shift_sensitivity_hypothesis": (
            supports_large_target_hypothesis
        ),
        "interpretation_guard": (
            "This is a descriptive validation-mask morphology audit. "
            "It can support or weaken the target-scale explanation but "
            "cannot establish causality or replace detector robustness "
            "measurements."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    result = compare(candidate, reference)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
