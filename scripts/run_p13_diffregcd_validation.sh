#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-/root/autodl-tmp/rs_cd_work/DiffRegCD}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/manifests/datasets/levir_cd.csv}"
PROJECT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
EXTERNAL_PYTHON="${EXTERNAL_ROOT}/.venv/bin/python"
PROTOCOL="${PROJECT_ROOT}/configs/p13_diffregcd_validation.yaml"
EXPECTED_COMMIT="6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_PARENT="${PROJECT_ROOT}/reports/p13_diffregcd_validation"
REPORT_ROOT="${REPORT_PARENT}/${STAMP}"

test -x "${PROJECT_PYTHON}"
test -x "${EXTERNAL_PYTHON}"
test -d "${EXTERNAL_ROOT}/.git"
test -d "${DATASET_ROOT}"
test -f "${MANIFEST}"
test -f "${PROTOCOL}"

ACTUAL_COMMIT="$(git -C "${EXTERNAL_ROOT}" rev-parse HEAD)"
test "${ACTUAL_COMMIT}" = "${EXPECTED_COMMIT}" || {
  echo "DiffRegCD commit mismatch: ${ACTUAL_COMMIT}" >&2
  exit 1
}

mkdir -p "${REPORT_ROOT}"
printf '%s\n' "${REPORT_ROOT}" > "${REPORT_PARENT}/LATEST"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${EXTERNAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

cd "${PROJECT_ROOT}"

echo "===== TARGETED TESTS ====="
"${PROJECT_PYTHON}" -m pytest -q \
  tests/test_audit_diffregcd_external.py \
  tests/test_diffregcd_external.py

echo "===== SOURCE AUDIT ====="
"${PROJECT_PYTHON}" -m tools.audit_diffregcd_external \
  --external-root "${EXTERNAL_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --output "${REPORT_ROOT}/source_audit.json"

echo "===== SYNTHETIC SMOKE (NO VALIDATION DATA) ====="
"${EXTERNAL_PYTHON}" -u -m tools.evaluate_diffregcd_external \
  --external-root "${EXTERNAL_ROOT}" \
  --protocol "${PROTOCOL}" \
  --output-dir "${REPORT_ROOT}/smoke" \
  --smoke-only

echo "===== PREREGISTERED LEVIR VALIDATION ====="
"${EXTERNAL_PYTHON}" -u -m tools.evaluate_diffregcd_external \
  --external-root "${EXTERNAL_ROOT}" \
  --protocol "${PROTOCOL}" \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --split val \
  --output-dir "${REPORT_ROOT}/validation"

echo "===== FINAL P13 VALIDATION DECISION ====="
cat "${REPORT_ROOT}/validation/decision.json"
echo "P13 report root: ${REPORT_ROOT}"
echo "TEST SPLIT WAS NOT EVALUATED."
