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

if [[ -z "${ROUTER_REPORT:-}" ]]; then
  ROUTER_REPORT="$(cat reports/p7_task_benefit_router_levir/LATEST)"
fi
FEATURES_CSV="${FEATURES_CSV:-$ROUTER_REPORT/router_experiment/training_router_features.csv}"
TASK_ROUTER_CHECKPOINT="${TASK_ROUTER_CHECKPOINT:-$ROUTER_REPORT/router_experiment/task_router.pth}"

for required in \
  "$P7_CHECKPOINT" \
  "$FEATURES_CSV" \
  "$TASK_ROUTER_CHECKPOINT" \
  manifests/datasets/levir_cd.csv \
  configs/p7_safe_alignment_classifier.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

SEED="${SEED:-}"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$SEED" ]]; then
  tag="${tag}_seed${SEED}"
fi
report_root="reports/p7_safe_alignment_classifier_levir/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p7_safe_alignment_classifier_levir/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_CHECKPOINT"
printf '%s\n' "$FEATURES_CSV" > "$report_root/SOURCE_FEATURES_CSV"
sha256sum "$FEATURES_CSV" > "$report_root/SOURCE_FEATURES_CSV.sha256"

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

seed_args=()
if [[ -n "$SEED" ]]; then
  seed_args=(--seed "$SEED")
fi

python -u -m tools.train_evaluate_safe_alignment_classifier \
  --checkpoint "$P7_CHECKPOINT" \
  --features-csv "$FEATURES_CSV" \
  --task-router-checkpoint "$TASK_ROUTER_CHECKPOINT" \
  --protocol configs/p7_safe_alignment_classifier.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --device cuda \
  ${seed_args[@]+"${seed_args[@]}"} \
  --output-dir "$report_root/safe_alignment_experiment" \
  2>&1 | tee "$report_root/safe_alignment_experiment.log"

cp "$report_root/safe_alignment_experiment/decision.json" \
  "$report_root/decision.json"

echo "P7-3 safe-alignment classifier complete: $report_root"
cat "$report_root/decision.json"
