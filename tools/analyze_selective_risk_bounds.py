"""Distribution-free bounds on the selective router's risk at the frozen point.

The deployment threshold is reported as a point estimate: at the frozen
tau the router adopts alignment on some fraction of validation samples
and a small number of those adoptions are harmful. A point estimate on
64 samples per scenario invites the obvious objection that the safety
claim is an artifact of the sample size. This tool answers it without
touching the model, the threshold, or the test split: it reads the
frozen validation artifacts and reports exact Clopper-Pearson bounds.

Two quantities matter and they are not the same:

* the clean adoption rate, which the asymmetric design exists to hold at
  zero, and where the interesting statement is an upper bound given zero
  observed adoptions;
* the harmful-adoption rate conditional on adopting, which is the risk a
  deployment would actually carry.

Independence. Within one scenario the samples are 64 distinct image
pairs and a binomial model is reasonable. Across scenarios they are the
same 64 base pairs under different injected displacements, so a pooled
binomial over all four is not justified. The pooled row below is
reported for completeness and flagged, not used as the headline.

This is an analysis of already-frozen artifacts. It selects nothing,
tunes nothing, and reads no test label.

Usage
-----
    python -m tools.analyze_selective_risk_bounds \
        --selective paper/data/sources/p7_task_selective.json \
        --output paper/data/selective_risk_bounds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.stats import beta


def clopper_pearson_upper(successes: int, trials: int, *, confidence: float) -> float:
    """One-sided exact upper bound on a binomial rate."""
    if trials == 0:
        return float("nan")
    if successes >= trials:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def clopper_pearson_lower(successes: int, trials: int, *, confidence: float) -> float:
    if trials == 0:
        return float("nan")
    if successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selective", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    report = json.loads(Path(args.selective).read_text(encoding="utf-8"))
    if not report.get("reporting_only_no_gate"):
        raise SystemExit(
            "refusing to run: the source is not marked reporting-only, so it "
            "may be a gated artifact rather than a frozen validation report"
        )

    rows: list[dict[str, Any]] = []
    pooled_samples = pooled_accepted = pooled_harmful = 0
    for scenario, record in report["per_scenario"].items():
        point = record["deployed_operating_point"]
        samples = int(record["samples"])
        accepted = int(point["accepted"])
        harmful = int(point["harmful_accepted"])
        pooled_samples += samples
        pooled_accepted += accepted
        pooled_harmful += harmful
        rows.append(
            {
                "scenario": scenario,
                "samples": samples,
                "accepted": accepted,
                "adoption_rate": accepted / samples,
                "adoption_rate_upper": clopper_pearson_upper(
                    accepted, samples, confidence=args.confidence
                ),
                "harmful_accepted": harmful,
                "harmful_rate_given_adoption": (
                    harmful / accepted if accepted else None
                ),
                "harmful_rate_upper": (
                    clopper_pearson_upper(harmful, accepted, confidence=args.confidence)
                    if accepted
                    else None
                ),
                "harmful_share_of_all_samples": harmful / samples,
                "harmful_share_upper": clopper_pearson_upper(
                    harmful, samples, confidence=args.confidence
                ),
            }
        )

    summary = {
        "status": "completed",
        "analysis_type": "frozen_validation_artifacts_only",
        "selects_nothing": True,
        "test_split_touched": False,
        "confidence": args.confidence,
        "method": "one-sided exact Clopper-Pearson",
        "per_scenario": rows,
        "pooled": {
            "samples": pooled_samples,
            "accepted": pooled_accepted,
            "harmful_accepted": pooled_harmful,
            "harmful_rate_given_adoption": (
                pooled_harmful / pooled_accepted if pooled_accepted else None
            ),
            "harmful_rate_upper": clopper_pearson_upper(
                pooled_harmful, pooled_accepted, confidence=args.confidence
            ),
            "independence_caveat": (
                "The four scenarios reuse the same 64 base pairs, so these "
                "adoptions are not independent trials. Use the per-scenario "
                "bounds, where the 64 pairs within a scenario are distinct."
            ),
        },
        "interpretation": (
            "These bounds describe the validation distribution at the frozen "
            "operating point. They are not a guarantee about the sealed test "
            "split, which was consumed once under the same frozen threshold "
            "but is not resampled here, and they are not a guarantee about "
            "any other dataset."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"exact one-sided Clopper-Pearson, confidence {args.confidence:.2f}\n")
    print(f"{'scenario':18s} {'n':>3s} {'adopt':>6s} {'adopt UB':>9s} "
          f"{'harm':>5s} {'risk':>7s} {'risk UB':>8s}")
    for row in rows:
        risk = row["harmful_rate_given_adoption"]
        upper = row["harmful_rate_upper"]
        print(
            f"{row['scenario']:18s} {row['samples']:3d} "
            f"{row['accepted']:3d}/{row['samples']:<2d} "
            f"{row['adoption_rate_upper']:9.4f} "
            f"{row['harmful_accepted']:5d} "
            + (f"{risk:7.4f} {upper:8.4f}" if accepted_ok(row) else f"{'-':>7s} {'-':>8s}")
        )
    print(f"\nwrote {output}")


def accepted_ok(row: dict[str, Any]) -> bool:
    return bool(row["accepted"])


if __name__ == "__main__":
    main()
