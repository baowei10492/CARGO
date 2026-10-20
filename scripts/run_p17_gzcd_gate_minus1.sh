#!/usr/bin/env bash
set -euo pipefail

# P17 Gate -1: build GZ-CD tiles and audit them.
# Tiling and the raster-level split follow
# docs/P17_GZCD_SECOND_POSITIVE_PROTOCOL.md, frozen before any tile was
# written. No model is loaded and no split is evaluated here.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

SOURCE_ROOT="${SOURCE_ROOT:-/root/autodl-tmp/rs_cd_work/data/GZ-CD/CD_Data_GZ}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/rs_cd_work/data/GZ-CD-tiles}"
MANIFEST="${MANIFEST:-manifests/datasets/gzcd.csv}"
TILE_SIZE="${TILE_SIZE:-512}"

test -d "${SOURCE_ROOT}/T1" || {
  echo "Missing ${SOURCE_ROOT}/T1" >&2
  echo "Unzip CD_Data_GZ.zip so that T1/, T2/ and labels_change/ sit" >&2
  echo "directly under SOURCE_ROOT." >&2
  exit 1
}

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p17_gzcd_gate_minus1/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_gzcd_tiles.py

echo "===== DRY RUN: TILE BUDGET ====="
python -m tools.build_gzcd_tiles \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --tile-size "${TILE_SIZE}" \
  --report "${report_root}/build_dry_run.json" \
  --dry-run

echo "===== BUILD TILES ====="
python -m tools.build_gzcd_tiles \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --tile-size "${TILE_SIZE}" \
  --report "${report_root}/build.json" \
  2>&1 | tee "${report_root}/build.log"

echo "===== DATASET AUDIT ====="
python -m tools.audit_dataset \
  --dataset-root "${OUTPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-json "${report_root}/data_audit_GZ-CD.json" \
  --output-md "${report_root}/data_audit_GZ-CD.md" \
  2>&1 | tail -20

echo "===== CROSS-SPLIT DUPLICATE AND OVERLAP AUDIT ====="
# Expected to be zero: GZ-CD has no augmentation, so any hit means two
# rasters cover overlapping ground and the split is not disjoint.
python -m tools.audit_augmentation_leakage \
  --dataset-root "${OUTPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --splits train val test \
  --output-json "${report_root}/overlap_audit.json" \
  2>&1 | tail -25

echo "===== LABEL MORPHOLOGY (ordering only, not a gate) ====="
python -m tools.audit_change_target_scale \
  --dataset-name GZ-CD \
  --dataset-root "${OUTPUT_ROOT}" \
  --manifest "${MANIFEST}" \
  --split val \
  --mask-threshold 127 \
  --output "${report_root}/morphology_GZ-CD.json" \
  2>&1 | tail -30

echo "===== GATE -1 SUMMARY ====="
python - "${report_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
build = json.loads((root / "build.json").read_text(encoding="utf-8"))
overlap = json.loads((root / "overlap_audit.json").read_text(encoding="utf-8"))
morph = json.loads(
    (root / "morphology_GZ-CD.json").read_text(encoding="utf-8")
)

print("tiles per split      :", build["tiles_per_split"])
print("geometric maxima     :", build["geometric_tiles"])
print("matches protocol     :", build["geometric_matches_protocol"])
print("dropped no-data      :", build["dropped_no_data_tiles"])
print("cross-split exact    :", overlap["cross_split_exact_duplicate_groups"])
print("cross-split near     :", overlap["cross_split_near_duplicate_pairs"])
print("change fraction      :", round(morph["change_fraction"], 4))
print(
    "self-IoU at 8 px     :",
    round(float(morph["mask_self_iou_on_paired_support"]["8"]), 4),
    "(LEVIR 0.520, WHU 0.749, S2Looking 0.610, DSIFN 0.888)",
)
print(
    "area-weighted diameter:",
    round(
        float(
            morph["components"]["area_weighted_equivalent_diameter"]["median"]
        ),
        1,
    ),
    "px (LEVIR 37.5, DSIFN 286.1)",
)
if not all(build["geometric_matches_protocol"].values()):
    raise SystemExit(
        "Geometric tile counts disagree with the frozen protocol table"
    )
if morph["positive_samples"] == 0:
    raise SystemExit("Morphology audit found no change pixels")
PY

printf '%s\n' "${report_root}" > reports/p17_gzcd_gate_minus1/LATEST
sha256sum "${report_root}"/*.json "${MANIFEST}" \
  > "${report_root}/SHA256SUMS.txt"
echo "P17 Gate -1 complete: ${report_root}"
