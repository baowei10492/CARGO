from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rs_cd.runtime import save_json


PRIMARY_SCENARIO = "shift:s2:t2"
CONTAMINATION_PRONE_SCENARIO = "shift:s3:t2"


def _conditional_pck(summary: dict[str, Any], scenario: str) -> dict[str, float]:
    entry = summary["per_shift"][scenario]
    return {
        "pck1": float(entry["raw_pck1"]),
        "pck2": float(entry["raw_pck2"]),
        "boundary_fraction": float(entry.get("boundary_fraction", -1.0)),
    }


def matched_image_size_decision(
    reference_native: dict[str, Any],
    reference_matched: dict[str, Any],
    target: dict[str, Any],
    *,
    exonerate_min: float,
    convict_max: float,
    retention_min: float,
) -> dict[str, Any]:
    """Decide whether image size confounds a cross-dataset verdict.

    The WHU localizer stage-A verdict (all conditional gates failed) was
    measured on 512 px tiles, while every LEVIR number it is compared
    against was measured on 1024 px images. A global offset estimator has
    four times less support to vote with on a 512 px image, so the
    comparison confounds "this dataset is outside the applicability
    domain" with "this measurement configuration is too small". This is
    the F4 family (instrument mismatch) applied to the one structural
    difference between the two datasets.

    The control holds everything else fixed -- same checkpoint, same
    tool, same protocol, same radius -- and varies only image size on the
    reference dataset. Shift-4 is the primary scenario because its
    boundary fraction is the lowest of the three, so it is the least
    contaminated by clamping pseudo-hits; Shift-8 is reported but is not
    decisive.

    This control CANNOT reopen the WHU gates. Whatever it returns, the
    stage-A verdict stays failed. It decides only what that failure may
    be attributed to in writing.
    """

    native_image_size = reference_native.get("evaluation_image_size")
    matched_image_size = reference_matched.get("evaluation_image_size")
    if native_image_size is None or matched_image_size is None:
        raise ValueError(
            "Both reference summaries must record evaluation_image_size"
        )
    if native_image_size == matched_image_size:
        raise ValueError(
            "Reference native and matched controls have the same "
            "evaluation_image_size; image size was not varied"
        )

    native = _conditional_pck(reference_native, PRIMARY_SCENARIO)
    matched = _conditional_pck(reference_matched, PRIMARY_SCENARIO)
    subject = _conditional_pck(target, PRIMARY_SCENARIO)
    if native["pck1"] <= 0.0:
        raise ValueError(
            "Reference native conditional PCK@1 is zero; the reference "
            "measurement is unusable as a baseline"
        )
    retention = matched["pck1"] / native["pck1"]
    exonerated = (
        matched["pck1"] >= float(exonerate_min)
        and retention >= float(retention_min)
    )
    convicted = matched["pck1"] <= float(convict_max)
    if exonerated and convicted:
        raise ValueError(
            "Exonerate and convict bands overlap; check the thresholds"
        )
    if exonerated:
        verdict = "image_size_not_the_cause"
        attribution = "dataset_property_second_negative_case_supported"
    elif convicted:
        verdict = "image_size_is_the_dominant_confound"
        attribution = "whu_verdict_limited_to_512px_configuration"
    else:
        verdict = "ambiguous_partial_confound"
        attribution = "neither_explanation_may_be_claimed_alone"
    return {
        "status": "completed",
        "primary_scenario": PRIMARY_SCENARIO,
        "reference_native": native,
        "reference_matched": matched,
        "target": subject,
        "reference_native_image_size": native_image_size,
        "reference_matched_image_size": matched_image_size,
        "target_image_size": target.get("evaluation_image_size"),
        "retention_ratio": retention,
        "bands": {
            "exonerate_min_pck1": float(exonerate_min),
            "retention_min": float(retention_min),
            "convict_max_pck1": float(convict_max),
        },
        "verdict": verdict,
        "attribution": attribution,
        "shift8_reported_not_decisive": {
            "reference_native": _conditional_pck(
                reference_native,
                CONTAMINATION_PRONE_SCENARIO,
            ),
            "reference_matched": _conditional_pck(
                reference_matched,
                CONTAMINATION_PRONE_SCENARIO,
            ),
            "target": _conditional_pck(
                target,
                CONTAMINATION_PRONE_SCENARIO,
            ),
        },
        "gates_reopened": False,
        "note": (
            "Attribution control only. The localizer stage-A verdict on "
            "the target dataset remains failed regardless of this "
            "result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether evaluation image size confounds the "
            "cross-dataset localizer comparison, by varying only image "
            "size on the reference dataset."
        )
    )
    parser.add_argument("--reference-native", required=True)
    parser.add_argument("--reference-matched", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--exonerate-min", type=float, default=0.60)
    parser.add_argument("--convict-max", type=float, default=0.30)
    parser.add_argument("--retention-min", type=float, default=0.80)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def read(path: str) -> dict[str, Any]:
        return json.loads(
            Path(path).expanduser().read_text(encoding="utf-8")
        )

    decision = matched_image_size_decision(
        read(args.reference_native),
        read(args.reference_matched),
        read(args.target),
        exonerate_min=float(args.exonerate_min),
        convict_max=float(args.convict_max),
        retention_min=float(args.retention_min),
    )
    decision["sources"] = {
        "reference_native": str(args.reference_native),
        "reference_matched": str(args.reference_matched),
        "target": str(args.target),
    }
    save_json(args.output, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
