#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
# The original P7-3 seed-3407 run (earliest report dir; UTC tags sort).
if [[ -z "${P7_3_EXPERIMENT:-}" ]]; then
  baseline_dir="$(ls -d reports/p7_safe_alignment_classifier_levir/2*/ | head -n 1)"
  P7_3_EXPERIMENT="${baseline_dir%/}/safe_alignment_experiment"
fi

for required in \
  "$P7_CHECKPOINT" \
  "$P7_3_EXPERIMENT/safe_alignment_registration.pth" \
  "$P7_3_EXPERIMENT/safe_alignment_task.pth" \
  manifests/datasets/levir_cd.csv \
  configs/p9b_perturbation_robustness.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p9b_perturbation_robustness/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p9b_perturbation_robustness/LATEST
printf '%s\n' "$P7_3_EXPERIMENT" > "$report_root/SOURCE_P7_3"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

python -u -m tools.evaluate_perturbation_robustness \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p9b_perturbation_robustness.yaml \
  --registration-classifier \
    "$P7_3_EXPERIMENT/safe_alignment_registration.pth" \
  --task-classifier "$P7_3_EXPERIMENT/safe_alignment_task.pth" \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --device cuda \
  --output-dir "$report_root/perturbations" \
  2>&1 | tee "$report_root/perturbations.log"

cp "$report_root/perturbations/decision.json" "$report_root/decision.json"
echo "P9-B perturbation robustness complete: $report_root"
