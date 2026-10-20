#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/autodl-tmp/rs_cd_work/DiffRegCD}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/manifests/datasets/levir_cd.csv}"
PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
EXTERNAL_PYTHON="${EXTERNAL_ROOT}/.venv/bin/python"
VAL_PROTOCOL="${PROJECT_ROOT}/configs/p13_diffregcd_validation.yaml"
TEST_PROTOCOL="${PROJECT_ROOT}/configs/p13_diffregcd_final_test.yaml"
EXPECTED_COMMIT="6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${EXTERNAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

for required in \
  "${PROJECT_PYTHON}" \
  "${EXTERNAL_PYTHON}" \
  "${MANIFEST}" \
  "${VAL_PROTOCOL}" \
  "${TEST_PROTOCOL}"; do
  test -e "${required}" || {
    echo "Missing P13 final-test input: ${required}" >&2
    exit 1
  }
done

actual_commit="$(git -C "${EXTERNAL_ROOT}" rev-parse HEAD)"
test "${actual_commit}" = "${EXPECTED_COMMIT}" || {
  echo "DiffRegCD commit mismatch: ${actual_commit}" >&2
  exit 1
}

val_root="$(cat reports/p13_diffregcd_validation/LATEST)"
val_results="${val_root}/validation/results.json"
p12_root="$(cat reports/p12_final_test/LATEST)"
p12_results="${p12_root}/levir/results.json"
test -f "${val_results}"
test -f "${p12_results}"

mkdir -p reports/p13_diffregcd_external_freeze
freeze_tag="$(date -u +%Y%m%dT%H%M%SZ)"
freeze_root="reports/p13_diffregcd_external_freeze/${freeze_tag}"

echo "===== TARGETED TESTS ====="
"${PROJECT_PYTHON}" -m pytest -q \
  tests/test_diffregcd_external.py \
  tests/test_p13_diffregcd_final_test.py

echo "===== FREEZE OFFICIAL EXTERNAL PATH ====="
"${PROJECT_PYTHON}" -m tools.freeze_p13_diffregcd_external_test \
  --validation-results "${val_results}" \
  --validation-protocol "${VAL_PROTOCOL}" \
  --test-protocol "${TEST_PROTOCOL}" \
  --output-dir "${freeze_root}"
printf '%s\n' "${freeze_root}" \
  > reports/p13_diffregcd_external_freeze/LATEST

mkdir -p reports/p13_diffregcd_final_test
consumed="reports/p13_diffregcd_final_test/DIFFREGCD_LEVIR_TEST_CONSUMED.json"
if [[ -e "${consumed}" ]]; then
  echo "DiffRegCD LEVIR external test has already been started:" >&2
  cat "${consumed}" >&2
  exit 1
fi

test_tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p13_diffregcd_final_test/${test_tag}"
mkdir -p "${report_root}"
printf '%s\n' "${report_root}" \
  > reports/p13_diffregcd_final_test/LATEST

"${PROJECT_PYTHON}" - "${consumed}" "${report_root}" "${freeze_root}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "started_irreversible",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "report_root": sys.argv[2],
            "freeze_root": sys.argv[3],
            "external_method": "DiffRegCD",
            "inference_path": "official_unmodified",
            "allowed_reruns": 0,
            "reporting_only_no_selection": True,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "===== ONE-SHOT DIFFREGCD LEVIR TEST ====="
"${EXTERNAL_PYTHON}" -u -m tools.evaluate_diffregcd_external \
  --external-root "${EXTERNAL_ROOT}" \
  --protocol "${TEST_PROTOCOL}" \
  --freeze-manifest "${freeze_root}/freeze_manifest.json" \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --split test \
  --output-dir "${report_root}/levir" \
  2>&1 | tee "${report_root}/levir.log"

echo "===== SAME-SUPPORT COMPARISON WITH FROZEN P12 METHOD ====="
"${PROJECT_PYTHON}" -m tools.summarize_p13_external_comparison \
  --project-results "${p12_results}" \
  --external-results "${report_root}/levir/results.json" \
  --output-json "${report_root}/p12_vs_diffregcd.json" \
  --output-csv "${report_root}/p12_vs_diffregcd.csv"

"${PROJECT_PYTHON}" - "${consumed}" "${report_root}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
record["status"] = "completed"
record["completed_utc"] = datetime.now(timezone.utc).isoformat()
record["external_summary"] = str(
    Path(sys.argv[2]) / "levir" / "final_test_summary.json"
)
record["comparison"] = str(
    Path(sys.argv[2]) / "p12_vs_diffregcd.json"
)
path.write_text(
    json.dumps(record, indent=2) + "\n",
    encoding="utf-8",
)
PY

sha256sum \
  "${freeze_root}/freeze_manifest.json" \
  "${report_root}/levir/final_test_summary.json" \
  "${report_root}/levir/results.json" \
  "${report_root}/p12_vs_diffregcd.json" \
  "${consumed}" \
  > "${report_root}/SHA256SUMS.txt"

echo "P13 DiffRegCD one-shot test complete: ${report_root}"
cat "${report_root}/levir/final_test_summary.json"
cat "${report_root}/p12_vs_diffregcd.json"

