#!/usr/bin/env bash
set -euo pipefail

# Resume a P15 Gate 1 run whose original raw/oracle evaluations used
# unequal pixel support. Reuses the completed detector checkpoint and
# reruns validation only; no training and no test access.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DSIFN_ROOT="${DSIFN_ROOT:-/root/autodl-tmp/rs_cd_work/data/DSIFN-CD/DSIFN}"
ORACLE_CONFIG="${ORACLE_CONFIG:-configs/p15_dsifn_oracle_t2_shift.yaml}"
if [[ -z "${REPORT_ROOT:-}" ]]; then
  REPORT_ROOT="$(
    find reports/p15_dsifn_gate1 \
      -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' |
    sort -nr |
    head -n 1 |
    cut -d' ' -f2-
  )"
fi
test -n "${REPORT_ROOT}"
test -f "${REPORT_ROOT}/CHECKPOINT"
test -f "${REPORT_ROOT}/train_summary.json"
checkpoint="$(cat "${REPORT_ROOT}/CHECKPOINT")"
test -f "${checkpoint}"

raw_dir="${REPORT_ROOT}/raw_val_paired"
oracle_dir="${REPORT_ROOT}/oracle_val_paired"
test ! -e "${raw_dir}" || {
  echo "Refusing to overwrite existing ${raw_dir}" >&2
  exit 1
}
test ! -e "${oracle_dir}" || {
  echo "Refusing to overwrite existing ${oracle_dir}" >&2
  exit 1
}

echo "===== PAIRED-SUPPORT REGRESSION TESTS ====="
python -m pytest -q \
  tests/test_robustness.py \
  tests/test_audit_repairs.py \
  tests/test_p15_gate1.py

echo "===== RAW SHIFT ON SHARED RAW/ORACLE SUPPORT ====="
python -u -m tools.evaluate_robustness \
  --checkpoint "${checkpoint}" \
  --protocol "${ORACLE_CONFIG}" \
  --dataset-root "${DSIFN_ROOT}" \
  --split val \
  --device cuda \
  --paired-shift-support \
  --output-dir "${raw_dir}" \
  2>&1 | tee "${REPORT_ROOT}/raw_val_paired.log"

echo "===== ORACLE SHIFT ON SHARED RAW/ORACLE SUPPORT ====="
python -u -m tools.evaluate_robustness \
  --checkpoint "${checkpoint}" \
  --protocol "${ORACLE_CONFIG}" \
  --dataset-root "${DSIFN_ROOT}" \
  --split val \
  --device cuda \
  --paired-shift-support \
  --oracle-inverse-shift \
  --output-dir "${oracle_dir}" \
  2>&1 | tee "${REPORT_ROOT}/oracle_val_paired.log"

echo "===== REPAIRED PREREGISTERED GATE 1 DECISION ====="
python -u -m tools.summarize_p8_whu_headroom \
  --dataset DSIFN-CD \
  --raw "${raw_dir}/results.csv" \
  --oracle "${oracle_dir}/results.csv" \
  --train-summary "${REPORT_ROOT}/train_summary.json" \
  --shift8-headroom-min 0.05 \
  --output "${REPORT_ROOT}/gate1_decision.json" \
  2>&1 | tee "${REPORT_ROOT}/gate1_decision.log"

python - "${REPORT_ROOT}/PAIRING_REPAIR_NOTE.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "status": "completed",
            "reason": (
                "Original raw and oracle evaluations used different "
                "valid-pixel support and were rejected before a gate "
                "decision was produced."
            ),
            "repair": (
                "Both paths were rerun on the symmetric intersection "
                "support valid before and after inverse translation."
            ),
            "retraining": False,
            "test_split_touched": False,
            "superseded_outputs": ["raw_val", "oracle_val"],
            "authoritative_outputs": [
                "raw_val_paired",
                "oracle_val_paired",
                "gate1_decision.json",
            ],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

sha256sum \
  "${checkpoint}" \
  "${raw_dir}/results.csv" \
  "${oracle_dir}/results.csv" \
  "${REPORT_ROOT}/gate1_decision.json" \
  "${REPORT_ROOT}/PAIRING_REPAIR_NOTE.json" \
  > "${REPORT_ROOT}/PAIRED_SHA256SUMS.txt"

printf '%s\n' "${REPORT_ROOT}" > reports/p15_dsifn_gate1/LATEST
echo "P15 Gate 1 paired-support repair complete: ${REPORT_ROOT}"
cat "${REPORT_ROOT}/gate1_decision.json"
