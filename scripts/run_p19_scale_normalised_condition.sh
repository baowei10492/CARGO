#!/usr/bin/env bash
set -euo pipefail

# P19: restate the applicability condition in scale-free terms.
#
# Every dataset so far was compared at a fixed 8-pixel shift, a number
# calibrated on LEVIR-CD where changed objects are 37.5 px across. On
# datasets with objects an order of magnitude larger the same 8 pixels
# asks a much weaker question, so the apparent narrowness of the
# applicability window may be an artefact of the measuring scale.
#
# This sweeps displacement instead of fixing it and asks, for each
# dataset, how much displacement is needed to reach the overlap loss
# LEVIR sees at 8 px. Labels only: no model is loaded, nothing is
# trained, and no test split is read.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATA="${DATA:-/root/autodl-tmp/rs_cd_work/data}"
SHIFTS="${SHIFTS:-2 4 8 16 32 64 128}"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p19_scale_normalised/${tag}"
mkdir -p "${report_root}"

scan() {
  local name="$1" root="$2" manifest="$3" split="$4" threshold="$5"
  if [[ ! -f "${manifest}" ]]; then
    echo "SKIP ${name}: ${manifest} not found" >&2
    return 0
  fi
  echo "===== ${name} ====="
  python -u -m tools.scan_self_iou_curve \
    --dataset-name "${name}" \
    --dataset-root "${root}" \
    --manifest "${manifest}" \
    --split "${split}" \
    --mask-threshold "${threshold}" \
    --shifts ${SHIFTS} \
    --output "${report_root}/curve_${name}.json"
}

scan LEVIR-CD  "${DATA}/LEVIR-CD"       manifests/datasets/levir_cd.csv            val     127
scan WHU-CD    "${DATA}/WHU-CD"         manifests/datasets/whu_cd.csv              val     127
scan S2Looking "${DATA}/S2Looking"      manifests/datasets/s2looking.csv           val     127
scan GZ-CD     "${DATA}/GZ-CD-tiles"    manifests/datasets/gzcd.csv                val     127
scan SYSU-CD   "${DATA}/SYSU-CD"        manifests/datasets/sysu_cd.csv             val     127
scan PRCV2021  "${DATA}/PRCV2021"       manifests/datasets/prcv_cd.csv             val_set 127
scan DSIFN-CD  "${DATA}/DSIFN-CD/DSIFN" manifests/datasets/dsifn_cd_source_split.csv val     0

echo "===== SCALE COLLAPSE ANALYSIS ====="
python - "${report_root}" <<'PY'
import json
import sys
from pathlib import Path

# Area-weighted median equivalent diameter as published by the frozen
# morphology audits. These are NOT used in the ratio: each curve now
# measures its own diameter with the same primitive, and this table is
# only a cross-check that the two agree. S2Looking is absent on purpose,
# because no frozen audit ever measured it.
PUBLISHED_DIAMETER_PX = {
    "LEVIR-CD": 37.5,
    "WHU-CD": 91.6,
    "GZ-CD": 145.5,
    "SYSU-CD": 183.1,
    "PRCV2021": 223.4,
    "DSIFN-CD": 286.1,
}

root = Path(sys.argv[1])
rows = []
mismatches = []
touched_test = False
for path in sorted(root.glob("curve_*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    name = report["dataset"]
    measured = report["area_weighted_median_diameter_px"]
    published = PUBLISHED_DIAMETER_PX.get(name)
    touched_test = touched_test or bool(report.get("test_split_touched"))
    if published is not None and measured is not None:
        if abs(measured - published) > 0.55:
            mismatches.append((name, measured, published))
    rows.append(
        {
            "dataset": name,
            "diameter_px": measured,
            "published_diameter_px": published,
            "self_iou_at_8": report["self_iou_curve"].get("8"),
            "equivalent_shift_px": report["equivalent_shift_pixels"],
            "min_paired_support_fraction": report.get(
                "min_paired_support_fraction"
            ),
            "curve": report["self_iou_curve"],
        }
    )

if mismatches:
    print("\nDIAMETER MISMATCH against the frozen audits:")
    for name, measured, published in mismatches:
        print(f"  {name}: measured {measured:.1f} vs published {published:.1f}")
    raise SystemExit(
        "refusing to report a ratio built on numbers that do not reproduce "
        "the frozen morphology audits"
    )

print(f"\n{'dataset':>11s} {'diam':>7s} {'sIoU@8':>8s} "
      f"{'equiv px':>9s} {'equiv/diam':>11s}")
ratios = []
for row in sorted(rows, key=lambda item: item["diameter_px"] or 0):
    equivalent = row["equivalent_shift_px"]
    diameter = row["diameter_px"]
    ratio = (
        equivalent / diameter
        if equivalent is not None and diameter
        else None
    )
    if ratio is not None:
        ratios.append(ratio)
        row["equivalent_over_diameter"] = ratio
    print(
        f"{row['dataset']:>11s} {diameter or float('nan'):7.1f} "
        f"{row['self_iou_at_8'] or float('nan'):8.4f} "
        + (f"{equivalent:9.1f}" if equivalent is not None else "   beyond")
        + (f" {ratio:11.3f}" if ratio is not None else "           -")
    )

beyond = [
    row["dataset"] for row in rows if row["equivalent_shift_px"] is None
]

verdict = "insufficient_data"
spread = None
if len(ratios) >= 4:
    spread = max(ratios) / min(ratios)
    verdict = "collapses" if spread <= 1.6 else "does_not_collapse"
    print(f"\nratio range {min(ratios):.3f}-{max(ratios):.3f}, "
          f"spread {spread:.2f}x -> {verdict}")

# A dataset whose curve never reaches the reference within the scanned
# range is evidence against collapse, but it has no ratio and would
# otherwise just vanish from the spread, biasing the verdict toward the
# answer we want. Refuse to call it either way instead.
if beyond:
    print("\nnever reached the reference within the scanned range: "
          + ", ".join(beyond))
    if verdict == "collapses":
        verdict = "collapses_among_measured_only"
        print("verdict downgraded: the spread omits the datasets above")

summary = {
    "status": "completed",
    "analysis_type": "labels_only_no_training",
    "test_split_touched": touched_test,
    "reference": "LEVIR-CD self-IoU at 8 px",
    "datasets": rows,
    "ratio_spread_verdict": verdict,
    "ratio_spread": spread,
    "beyond_scanned_range": beyond,
    "interpretation": (
        "A collapse means the fixed 8-pixel protocol, not the method, "
        "is what made the applicability window look narrow: each dataset "
        "needs a displacement proportional to its object scale before the "
        "task loses comparable overlap. It does not by itself show the "
        "method works at those larger displacements, which the frozen "
        "8-pixel search radius cannot test."
    ),
}
(root / "scale_collapse.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "${report_root}" > reports/p19_scale_normalised/LATEST
sha256sum "${report_root}"/*.json > "${report_root}/SHA256SUMS.txt"
echo "P19 complete: ${report_root}"
