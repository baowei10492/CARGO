#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi

SAFE_REPORT="${SAFE_REPORT:-reports/p7_safe_alignment_classifier_levir/20260726T080834Z}"
if [[ ! -d "$SAFE_REPORT" ]]; then
  SAFE_REPORT="$(cat reports/p7_safe_alignment_classifier_levir/LATEST)"
fi
EXPERIMENT="$SAFE_REPORT/safe_alignment_experiment"
TASK_ROUTER="${TASK_ROUTER:-$EXPERIMENT/safe_alignment_task.pth}"
REGISTRATION_ROUTER="${REGISTRATION_ROUTER:-$EXPERIMENT/safe_alignment_registration.pth}"

for required in \
  "$P7_CHECKPOINT" \
  "$TASK_ROUTER" \
  "$REGISTRATION_ROUTER" \
  configs/p10_efficiency.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p10_efficiency/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p10_efficiency/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_CHECKPOINT"
printf '%s\n' "$TASK_ROUTER" > "$report_root/SOURCE_TASK_ROUTER"
printf '%s\n' "$REGISTRATION_ROUTER" \
  > "$report_root/SOURCE_REGISTRATION_ROUTER"

python -m pytest -q \
  tests/test_p10_efficiency.py \
  tests/test_audit_repairs.py \
  2>&1 | tee "$report_root/pytest.log"

python -u -m tools.benchmark_p10_efficiency \
  --checkpoint "$P7_CHECKPOINT" \
  --task-router "$TASK_ROUTER" \
  --registration-router "$REGISTRATION_ROUTER" \
  --protocol configs/p10_efficiency.yaml \
  --device cuda \
  --output "$report_root/efficiency.json" \
  2>&1 | tee "$report_root/efficiency.log"

sha256sum \
  "$report_root/efficiency.json" \
  "$P7_CHECKPOINT" \
  "$TASK_ROUTER" \
  "$REGISTRATION_ROUTER" \
  > "$report_root/SHA256SUMS.txt"

echo "P10 efficiency complete: $report_root"
cat "$report_root/efficiency.json"
