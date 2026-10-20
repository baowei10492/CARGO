#!/usr/bin/env bash
set -euo pipefail

# P15 Gate -1 follow-up: rebuild a leakage-free source-crop split.
# Runs only after the leakage audit reported unusable official splits.
# Reads images and hashes, never labels, and never reassigns the test
# split.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "${DATASET_ROOT:-}" && -f reports/p15_dsifn_gate_minus1/DATASET_ROOT ]]; then
  DATASET_ROOT="$(cat reports/p15_dsifn_gate_minus1/DATASET_ROOT)"
fi
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/rs_cd_work/data/DSIFN-CD/DSIFN}"
MANIFEST="${MANIFEST:-manifests/datasets/dsifn_cd.csv}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-manifests/datasets/dsifn_cd_source_split.csv}"
CLUSTER_HAMMING_MAX="${CLUSTER_HAMMING_MAX:-5}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.2}"

test -f "${MANIFEST}" || {
  echo "Missing manifest: ${MANIFEST}; run Gate -1 first." >&2
  exit 1
}

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p15_dsifn_resplit/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p15_dsifn_gates.py

echo "===== FULL LEAKAGE AUDIT WITH CACHED DIGESTS ====="
python -m tools.audit_augmentation_leakage \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --splits train val test \
  --example-limit 200 \
  --output-json "${report_root}/augmentation_leakage_full.json" \
  --output-records-csv "${report_root}/patch_digests.csv" \
  2>&1 | tail -40

echo "===== SPLIT-PAIR BREAKDOWN (all groups, not just examples) ====="
python - "${report_root}/augmentation_leakage_full.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print("exact groups by split pair:",
      report["cross_split_exact_groups_by_split_pair"])
print("near pairs by split pair  :",
      report["cross_split_near_pairs_by_split_pair"])
print("hamming histogram         :",
      report["near_pair_hamming_histogram"])
print("splits involved           :",
      report["splits_involved_in_leakage"])
PY

echo "===== REBUILD SOURCE-CROP SPLIT ====="
python -m tools.build_source_crop_split \
  --records-csv "${report_root}/patch_digests.csv" \
  --manifest "${MANIFEST}" \
  --output-manifest "${OUTPUT_MANIFEST}" \
  --output-json "${report_root}/source_split_report.json" \
  --protected-split test \
  --holdout-fraction "${HOLDOUT_FRACTION}" \
  --cluster-hamming-max "${CLUSTER_HAMMING_MAX}"

echo "===== VERIFY: NO CLUSTER SPANS TWO SPLITS ====="
python - "${OUTPUT_MANIFEST}" <<'PY'
import csv
import sys
from collections import defaultdict

splits = defaultdict(set)
counts = defaultdict(int)
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        splits[row["parent_id"]].add(row["split"])
        counts[row["split"]] += 1
spanning = {
    parent: sorted(values)
    for parent, values in splits.items()
    if len(values) > 1
}
print("split counts       :", dict(sorted(counts.items())))
print("clusters            :", len(splits))
print("clusters spanning   :", len(spanning))
if spanning:
    raise SystemExit(f"Cluster spans splits: {list(spanning)[:5]}")
print("OK: every source-crop cluster lives in exactly one split")
PY

printf '%s\n' "${report_root}" > reports/p15_dsifn_resplit/LATEST
sha256sum "${OUTPUT_MANIFEST}" \
  "${report_root}/source_split_report.json" \
  > "${report_root}/SHA256SUMS.txt"

echo "Re-split complete: ${report_root}"
echo "Use MANIFEST=${OUTPUT_MANIFEST} for Gate 0 onwards."
