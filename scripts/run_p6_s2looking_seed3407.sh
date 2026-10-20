#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/rs_cd_work/rs_cd_project"
WORK_ROOT="/root/autodl-tmp/rs_cd_work"
DATA_PARENT="${WORK_ROOT}/data"
DATASET_ROOT="${DATA_PARENT}/S2Looking"
MOUNT_ROOT="/root/autodl-fs"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
MANIFEST="${PROJECT_ROOT}/manifests/datasets/s2looking.csv"
PROTOCOL="${PROJECT_ROOT}/configs/p4_robustness_diagnostic_v1.yaml"

BASE_CONFIG="${PROJECT_ROOT}/configs/p6_s2looking_dinov3_partial_ft.yaml"
STAGE_A_CONFIG="${PROJECT_ROOT}/configs/p6_s2looking_a2_registration_stage_a.yaml"
STAGE_B_CONFIG="${PROJECT_ROOT}/configs/p6_s2looking_gated_stage_b.yaml"

BASE_OUTPUT="${PROJECT_ROOT}/outputs/p6_s2looking_dinov3_partial_ft"
STAGE_A_OUTPUT="${PROJECT_ROOT}/outputs/p6_s2looking_a2_registration_stage_a"
STAGE_B_OUTPUT="${PROJECT_ROOT}/outputs/p6_s2looking_gated_stage_b"

REPORT_ROOT="${PROJECT_ROOT}/reports/p6_s2looking"
ROBUSTNESS_ROOT="${PROJECT_ROOT}/reports/robustness"

mkdir -p "${DATA_PARENT}" "${REPORT_ROOT}" "${ROBUSTNESS_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python environment not found: ${PYTHON}" >&2
  exit 1
fi

for required in \
  "${BASE_CONFIG}" \
  "${STAGE_A_CONFIG}" \
  "${STAGE_B_CONFIG}" \
  "${PROTOCOL}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file not found: ${required}" >&2
    exit 1
  fi
done

mapfile -t archives < <(
  find -L "${MOUNT_ROOT}" -maxdepth 4 -type f -iname 'S2LOOKING.zip' -print
)

if [[ "${#archives[@]}" -ne 1 ]]; then
  echo "Expected exactly one S2LOOKING.zip under ${MOUNT_ROOT}." >&2
  printf 'Found: %s\n' "${archives[@]:-none}" >&2
  exit 1
fi
ARCHIVE="${archives[0]}"

echo "===== P6 S2LOOKING PIPELINE ====="
echo "archive=${ARCHIVE}"
echo "dataset_root=${DATASET_ROOT}"
echo "project_root=${PROJECT_ROOT}"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "===== STORAGE ====="
df -h "${MOUNT_ROOT}" "${WORK_ROOT}"
du -sh "${ARCHIVE}"

echo "===== ARCHIVE TEST ====="
unzip -tq "${ARCHIVE}"

if [[ -d "${DATASET_ROOT}" ]]; then
  if [[ ! -d "${DATASET_ROOT}/train/Image1" ]]; then
    echo "Dataset directory exists but is incomplete: ${DATASET_ROOT}" >&2
    exit 1
  fi
  echo "Existing complete-looking dataset directory will be reused."
else
  echo "===== EXTRACT DATASET ====="
  unzip -q "${ARCHIVE}" -d "${DATA_PARENT}"
fi

echo "===== DATASET COUNTS ====="
for split in train val test; do
  t1_count="$(
    find "${DATASET_ROOT}/${split}/Image1" -maxdepth 1 -type f \
      -iname '*.png' | wc -l
  )"
  t2_count="$(
    find "${DATASET_ROOT}/${split}/Image2" -maxdepth 1 -type f \
      -iname '*.png' | wc -l
  )"
  mask_count="$(
    find "${DATASET_ROOT}/${split}/label" -maxdepth 1 -type f \
      -iname '*.png' | wc -l
  )"
  printf '%-5s Image1=%s Image2=%s label=%s\n' \
    "${split}" "${t1_count}" "${t2_count}" "${mask_count}"
  if [[ "${t1_count}" -eq 0 ]] ||
     [[ "${t1_count}" -ne "${t2_count}" ]] ||
     [[ "${t1_count}" -ne "${mask_count}" ]]; then
    echo "Pair count mismatch in split ${split}." >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"

echo "===== BUILD MANIFEST ====="
mkdir -p "$(dirname "${MANIFEST}")"
"${PYTHON}" -m tools.build_manifest \
  --dataset-root "${DATASET_ROOT}" \
  --output "${MANIFEST}" \
  --splits train val test \
  --t1-dir Image1 \
  --t2-dir Image2 \
  --mask-dir label \
  --parent-id-mode sample_id

echo "===== FULL DATA AUDIT ====="
"${PYTHON}" -m tools.audit_dataset \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --mask-threshold 127 \
  --output-json "${REPORT_ROOT}/data_audit.json" \
  --output-md "${REPORT_ROOT}/data_audit.md"

echo "===== UNIT TESTS ====="
"${PYTHON}" -m pytest -q

echo "===== GPU SMOKE: P3B S2LOOKING ====="
"${PYTHON}" -m tools.gpu_smoke \
  --config "${BASE_CONFIG}" \
  --batch-size 128 \
  --image-size 256 \
  --steps 10 \
  --output "${REPORT_ROOT}/gpu_smoke_p3b_bs128.json"

echo "===== TRAIN P3B S2LOOKING BASELINE ====="
PYTHONUNBUFFERED=1 "${PYTHON}" -u -m tools.train \
  --config "${BASE_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --device cuda \
  2>&1 | tee "${REPORT_ROOT}/p3b_train.log"

BASE_SUMMARY="$(
  find "${BASE_OUTPUT}" -name summary.json -printf '%T@ %p\n' |
    sort -nr |
    head -n 1 |
    cut -d' ' -f2-
)"
BASE_RUN_DIR="$(dirname "${BASE_SUMMARY}")"
BASE_CKPT="${BASE_RUN_DIR}/checkpoints/best.pth"
test -f "${BASE_CKPT}"

echo "base_summary=${BASE_SUMMARY}"
cat "${BASE_SUMMARY}"

echo "===== P3B VALIDATION ROBUSTNESS ====="
"${PYTHON}" -m tools.evaluate_robustness \
  --checkpoint "${BASE_CKPT}" \
  --protocol "${PROTOCOL}" \
  --dataset-root "${DATASET_ROOT}" \
  --split val \
  --device cuda \
  --output-dir "${ROBUSTNESS_ROOT}/p6_s2looking_p3b_val"

echo "===== TRAIN P5C STAGE A ====="
PYTHONUNBUFFERED=1 "${PYTHON}" -u -m tools.train \
  --config "${STAGE_A_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --initial-checkpoint "${BASE_CKPT}" \
  --device cuda \
  2>&1 | tee "${REPORT_ROOT}/stage_a_train.log"

STAGE_A_SUMMARY="$(
  find "${STAGE_A_OUTPUT}" -name summary.json -printf '%T@ %p\n' |
    sort -nr |
    head -n 1 |
    cut -d' ' -f2-
)"
STAGE_A_RUN_DIR="$(dirname "${STAGE_A_SUMMARY}")"
STAGE_A_CKPT="${STAGE_A_RUN_DIR}/checkpoints/last.pth"
test -f "${STAGE_A_CKPT}"

echo "stage_a_summary=${STAGE_A_SUMMARY}"
cat "${STAGE_A_SUMMARY}"

echo "===== STAGE A REGISTRATION EVALUATION ====="
"${PYTHON}" -m tools.evaluate_registration \
  --checkpoint "${STAGE_A_CKPT}" \
  --dataset-root "${DATASET_ROOT}" \
  --split val \
  --device cuda \
  --batch-size 16 \
  --output "${ROBUSTNESS_ROOT}/p6_s2looking_stage_a_registration.json"

echo "===== TRAIN P5C STAGE B ====="
PYTHONUNBUFFERED=1 "${PYTHON}" -u -m tools.train \
  --config "${STAGE_B_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --initial-checkpoint "${STAGE_A_CKPT}" \
  --device cuda \
  2>&1 | tee "${REPORT_ROOT}/stage_b_train.log"

STAGE_B_SUMMARY="$(
  find "${STAGE_B_OUTPUT}" -name summary.json -printf '%T@ %p\n' |
    sort -nr |
    head -n 1 |
    cut -d' ' -f2-
)"
STAGE_B_RUN_DIR="$(dirname "${STAGE_B_SUMMARY}")"
STAGE_B_EPOCH2="${STAGE_B_RUN_DIR}/checkpoints/epoch_002.pth"
FROZEN_CKPT="${STAGE_B_RUN_DIR}/checkpoints/epoch_002_threshold020.pth"
test -f "${STAGE_B_EPOCH2}"

echo "stage_b_summary=${STAGE_B_SUMMARY}"
cat "${STAGE_B_SUMMARY}"

echo "===== FREEZE EPOCH 2 + THRESHOLD 0.20 ====="
"${PYTHON}" - "${STAGE_B_EPOCH2}" "${FROZEN_CKPT}" <<'PY'
from pathlib import Path
import sys
import torch

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
checkpoint["config"]["model"]["alignment_gate"] = "threshold_nonzero"
checkpoint["config"]["model"]["zero_probability_threshold"] = 0.20
torch.save(checkpoint, target)
print(f"Saved frozen candidate to {target}")
PY

sha256sum "${FROZEN_CKPT}" |
  tee "${REPORT_ROOT}/frozen_checkpoint_sha256.txt"

echo "===== P5C VALIDATION ROBUSTNESS ====="
"${PYTHON}" -m tools.evaluate_robustness \
  --checkpoint "${FROZEN_CKPT}" \
  --protocol "${PROTOCOL}" \
  --dataset-root "${DATASET_ROOT}" \
  --split val \
  --device cuda \
  --output-dir "${ROBUSTNESS_ROOT}/p6_s2looking_p5c_val"

echo "===== COMPARE P5C VS P3B ====="
"${PYTHON}" -m tools.compare_robustness \
  --reference "${ROBUSTNESS_ROOT}/p6_s2looking_p3b_val/results.json" \
  --candidate "${ROBUSTNESS_ROOT}/p6_s2looking_p5c_val/results.json" \
  --reference-name p3b_s2looking \
  --candidate-name p5c_s2looking \
  --output "${ROBUSTNESS_ROOT}/p6_s2looking_p5c_vs_p3b.csv"

echo "===== FINAL PATHS ====="
echo "base_run_dir=${BASE_RUN_DIR}"
echo "stage_a_run_dir=${STAGE_A_RUN_DIR}"
echo "stage_b_run_dir=${STAGE_B_RUN_DIR}"
echo "frozen_checkpoint=${FROZEN_CKPT}"
echo "comparison=${ROBUSTNESS_ROOT}/p6_s2looking_p5c_vs_p3b.json"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "TEST SPLIT WAS NOT EVALUATED."
