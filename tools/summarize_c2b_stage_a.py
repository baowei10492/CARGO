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


def boundary_contamination_annotation(
    offset_summary: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Annotate whether Shift-8 PCK is distorted by the search boundary.

    This is an ANNOTATION, never a gate.

    CORRECTED 2026-07-26 (main record 75.2). The first version compared
    the estimate's raw boundary rate against a fixed threshold. That was
    wrong: ``boundary_fraction`` counts estimates whose Chebyshev norm
    reaches the search limit, and at a severity whose INJECTED offset
    already sits on that limit a perfect estimator saturates it. The
    reference dataset proves it -- LEVIR Shift-8 measures
    boundary_fraction 0.984 with conditional PCK@1 0.922 and mean EPE
    0.58. The old rule would have flagged the best measurement in the
    project as contaminated.

    The interpretable quantity is the EXCESS over the ground truth's own
    boundary rate:

    * excess >= +threshold -> estimates pile onto the boundary that the
      truth does not reach: clamping, PCK may be inflated;
    * excess <= -threshold -> estimates fail to reach a boundary the
      truth does reach: undershoot, the estimator is not tracking the
      injection (this is WHU's case, not clamping);
    * otherwise -> the estimate's boundary behaviour matches the truth's.

    Summaries produced before ``gt_boundary_fraction`` existed cannot be
    annotated at all, and say so rather than guessing.
    """

    shift8 = offset_summary["per_shift"]["shift:s3:t2"]
    value = float(shift8.get("boundary_fraction", -1.0))
    truth = shift8.get("gt_boundary_fraction")
    if value < 0.0:
        return {
            "boundary_fraction": value,
            "threshold": float(threshold),
            "available": False,
            "interpretation": "boundary_fraction_not_reported",
        }
    if truth is None:
        return {
            "boundary_fraction": value,
            "threshold": float(threshold),
            "available": False,
            "interpretation": (
                "gt_boundary_fraction_missing_annotation_not_computable"
            ),
        }
    truth = float(truth)
    excess = float(shift8.get("excess_boundary_fraction", value - truth))
    if excess >= float(threshold):
        interpretation = "shift8_pck_contains_clamping_pseudo_hits"
        clamped, undershooting = True, False
    elif excess <= -float(threshold):
        interpretation = "shift8_estimates_undershoot_the_true_magnitude"
        clamped, undershooting = False, True
    else:
        interpretation = "shift8_boundary_behaviour_matches_truth"
        clamped, undershooting = False, False
    return {
        "boundary_fraction": value,
        "gt_boundary_fraction": truth,
        "excess_boundary_fraction": excess,
        "threshold": float(threshold),
        "available": True,
        "contaminated": clamped,
        "undershooting": undershooting,
        "interpretation": interpretation,
    }


def c2b_stage_a_decision(
    registration_summary: dict[str, Any],
    offset_summary: dict[str, Any],
    *,
    boundary_contamination_threshold: float = 0.30,
) -> dict[str, Any]:
    """Apply the predeclared C2b stage-A gates.

    Conditional gates measure task-coordinate localization quality on
    S2Looking val (against injected offsets). The clean gate measures
    whether the decoupled q head keeps clean pairs silent; if only the
    clean gate fails, the precedented next step is P7-1C-style
    full-resolution q recalibration, not abandoning the branch.
    """

    clean = registration_summary["0"]
    shift4 = offset_summary["per_shift"]["shift:s2:t2"]
    shift8 = offset_summary["per_shift"]["shift:s3:t2"]
    conditional_gates = {
        "shift4_conditional_pck1": _gate(
            float(shift4["raw_pck1"]),
            minimum=0.70,
        ),
        "shift4_conditional_pck2": _gate(
            float(shift4["raw_pck2"]),
            minimum=0.85,
        ),
        "shift8_conditional_pck1": _gate(
            float(shift8["raw_pck1"]),
            minimum=0.70,
        ),
        "shift8_conditional_pck2": _gate(
            float(shift8["raw_pck2"]),
            minimum=0.85,
        ),
    }
    clean_gates = {
        "clean_gated_hard_within_1px": _gate(
            float(clean["hard_within_1px_rate"]),
            minimum=0.90,
        ),
    }
    conditional_passed = all(
        bool(gate["passed"]) for gate in conditional_gates.values()
    )
    clean_passed = all(
        bool(gate["passed"]) for gate in clean_gates.values()
    )
    # Dataset-agnostic branch names. The original strings were written
    # when this tool only ever adjudicated S2Looking, so a WHU run came
    # out labelled "s2looking_scope_review_stop_loss" -- correct routing,
    # wrong dataset in the label, which is exactly the kind of thing that
    # gets misread months later. The three branches and their conditions
    # are unchanged; only the strings and the added dataset field are new.
    if conditional_passed and clean_passed:
        next_action = "localizer_stage_a_passed_continue_chain"
        legacy = "c2b_step3_feature_collection_and_p7_3_replay"
    elif conditional_passed:
        next_action = "localizer_stage_a_fullres_q_calibration"
        legacy = "c2b_step2_fullres_q_calibration"
    else:
        next_action = "localizer_stage_a_stop_loss_scope_review"
        legacy = "s2looking_scope_review_stop_loss"
    return {
        "status": "completed",
        "estimator_conditional_passed": conditional_passed,
        "clean_gate_passed": clean_passed,
        "conditional_gates": conditional_gates,
        "clean_gates": clean_gates,
        "clean_q_trigger_rate_reported": float(
            offset_summary["q_trigger_rate"]["clean:s0:common"]
        ),
        "shift8_boundary_fraction_reported": float(
            offset_summary["per_shift"]["shift:s3:t2"].get(
                "boundary_fraction",
                -1.0,
            )
        ),
        "shift8_boundary_contamination": boundary_contamination_annotation(
            offset_summary,
            threshold=boundary_contamination_threshold,
        ),
        "next_action": next_action,
        "next_action_legacy_label": legacy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply predeclared C2b stage-A estimator gates."
    )
    parser.add_argument("--registration-results", required=True)
    parser.add_argument("--offset-summary", required=True)
    parser.add_argument(
        "--dataset-label",
        default="unspecified",
        help=(
            "Which dataset this decision adjudicates. Recorded in the "
            "decision file so a stop-loss verdict can never be read as "
            "belonging to a different dataset."
        ),
    )
    parser.add_argument(
        "--boundary-contamination-threshold",
        type=float,
        default=0.30,
        help=(
            "Shift-8 boundary fraction at or above which the Shift-8 "
            "PCK numbers are annotated as containing clamping "
            "pseudo-hits. Annotation only: it never changes a gate."
        ),
    )
    parser.add_argument(
        "--transfer-clean-q-reference",
        type=float,
        default=None,
        help=(
            "Clean q trigger rate measured on this dataset with the "
            "LEVIR-trained localizer. Supplying it emits the "
            "screening-ambiguity resolution block: a native-trained "
            "localizer that drops the clean q rate to LEVIR levels "
            "attributes the screening red to a descriptor domain gap "
            "rather than to a spatially varying intrinsic field."
        ),
    )
    parser.add_argument(
        "--native-clean-q-max",
        type=float,
        default=0.20,
        help=(
            "Clean q trigger rate at or below which the native "
            "localizer counts as LEVIR-like for the ambiguity "
            "resolution above (LEVIR itself measured 0.0625)."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    registration = json.loads(
        Path(args.registration_results)
        .expanduser()
        .read_text(encoding="utf-8")
    )
    offsets = json.loads(
        Path(args.offset_summary)
        .expanduser()
        .read_text(encoding="utf-8")
    )
    decision = c2b_stage_a_decision(
        registration["summary_by_severity"],
        offsets,
        boundary_contamination_threshold=float(
            args.boundary_contamination_threshold
        ),
    )
    decision["dataset"] = str(args.dataset_label)
    decision["registration_results"] = str(args.registration_results)
    decision["offset_summary"] = str(args.offset_summary)
    offset_image_size = offsets.get("evaluation_image_size")
    if offset_image_size is not None:
        decision["evaluation_image_size"] = offset_image_size
    if args.transfer_clean_q_reference is not None:
        native = float(decision["clean_q_trigger_rate_reported"])
        decision["screening_ambiguity_resolution"] = {
            "transfer_localizer_clean_q": float(
                args.transfer_clean_q_reference
            ),
            "native_localizer_clean_q": native,
            "native_clean_q_max": float(args.native_clean_q_max),
            "attribution": (
                "descriptor_domain_gap"
                if native <= float(args.native_clean_q_max)
                else "intrinsic_field_or_unresolved"
            ),
        }
    save_json(args.output, decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
