#!/usr/bin/env bash
set -euo pipefail

# Validation-only morphology audit after P15 Gate 1 stop-loss.
# No training, detector inference, or test access.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DSIFN_ROOT="${DSIFN_ROOT:-/root/autodl-tmp/rs_cd_work/data/DSIFN-CD/DSIFN}"
DSIFN_MANIFEST="${DSIFN_MANIFEST:-manifests/datasets/dsifn_cd_source_split.csv}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
LEVIR_MANIFEST="${LEVIR_MANIFEST:-manifests/datasets/levir_cd.csv}"

for required in "${DSIFN_MANIFEST}" "${LEVIR_MANIFEST}"; do
  test -f "${required}" || {
    echo "Missing manifest: ${required}" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p15_target_scale_audit/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TEST ====="
python -m pytest -q tests/test_change_target_scale.py

echo "===== DSIFN SOURCE-SPLIT VAL ====="
python -u -m tools.audit_change_target_scale \
  --dataset-name DSIFN-CD \
  --dataset-root "${DSIFN_ROOT}" \
  --manifest "${DSIFN_MANIFEST}" \
  --split val \
  --mask-threshold 0 \
  --output "${report_root}/dsifn_val.json" \
  2>&1 | tee "${report_root}/dsifn_val.log"

echo "===== LEVIR VAL ====="
python -u -m tools.audit_change_target_scale \
  --dataset-name LEVIR-CD \
  --dataset-root "${LEVIR_ROOT}" \
  --manifest "${LEVIR_MANIFEST}" \
  --split val \
  --mask-threshold 127 \
  --output "${report_root}/levir_val.json" \
  2>&1 | tee "${report_root}/levir_val.log"

echo "===== DESCRIPTIVE COMPARISON ====="
python -m tools.compare_change_target_scale \
  --candidate "${report_root}/dsifn_val.json" \
  --reference "${report_root}/levir_val.json" \
  --output "${report_root}/comparison.json"

python - "${report_root}/TEST_SPLIT_STATUS.txt" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    "DSIFN and LEVIR test splits were not loaded or evaluated.\n"
    "This audit used validation masks only and performed no training.\n",
    encoding="utf-8",
)
PY

sha256sum \
  "${DSIFN_MANIFEST}" \
  "${LEVIR_MANIFEST}" \
  "${report_root}/dsifn_val.json" \
  "${report_root}/levir_val.json" \
  "${report_root}/comparison.json" \
  > "${report_root}/SHA256SUMS.txt"

printf '%s\n' "${report_root}" > reports/p15_target_scale_audit/LATEST
echo "P15 target-scale audit complete: ${report_root}"
cat "${report_root}/comparison.json"
