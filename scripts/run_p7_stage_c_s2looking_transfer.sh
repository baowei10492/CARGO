#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
S2LOOKING_ROOT="${S2LOOKING_ROOT:-/root/autodl-tmp/rs_cd_work/data/S2Looking}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi

DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-outputs/p6_s2looking_dinov3_partial_ft/20260726T014320Z_seed3407/checkpoints/best.pth}"

# Frozen P7-3 seed-3407 run. Deliberately NOT resolved via LATEST:
# Phase B reruns moved LATEST to seed 3409, while the preregistered
# transfer artifacts are the original protocol-seed classifiers.
SAFE_REPORT="${SAFE_REPORT:-reports/p7_safe_alignment_classifier_levir/20260726T080834Z}"
CLASSIFIER_REG="${CLASSIFIER_REG:-$SAFE_REPORT/safe_alignment_experiment/safe_alignment_registration.pth}"
CLASSIFIER_TASK="${CLASSIFIER_TASK:-$SAFE_REPORT/safe_alignment_experiment/safe_alignment_task.pth}"

for required in \
  "$P7_CHECKPOINT" \
  "$DETECTOR_CHECKPOINT" \
  "$CLASSIFIER_REG" \
  "$CLASSIFIER_TASK" \
  manifests/datasets/s2looking.csv \
  configs/p7_stage_c_s2looking_transfer.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p7_stage_c_s2looking_transfer/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p7_stage_c_s2looking_transfer/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_P7_CHECKPOINT"
printf '%s\n' "$DETECTOR_CHECKPOINT" \
  > "$report_root/SOURCE_DETECTOR_CHECKPOINT"
printf '%s\n' "$CLASSIFIER_REG" > "$report_root/SOURCE_CLASSIFIER_REG"
printf '%s\n' "$CLASSIFIER_TASK" > "$report_root/SOURCE_CLASSIFIER_TASK"

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

python -u -m tools.evaluate_safe_alignment_transfer \
  --p7-checkpoint "$P7_CHECKPOINT" \
  --detector-checkpoint "$DETECTOR_CHECKPOINT" \
  --classifier-registration "$CLASSIFIER_REG" \
  --classifier-task "$CLASSIFIER_TASK" \
  --protocol configs/p7_stage_c_s2looking_transfer.yaml \
  --dataset-root "$S2LOOKING_ROOT" \
  --manifest manifests/datasets/s2looking.csv \
  --device cuda \
  --output-dir "$report_root/transfer_experiment" \
  2>&1 | tee "$report_root/transfer_experiment.log"

cp "$report_root/transfer_experiment/decision.json" \
  "$report_root/decision.json"

echo "P7 Stage C S2Looking transfer complete: $report_root"
cat "$report_root/decision.json"
