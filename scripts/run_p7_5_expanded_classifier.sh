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
TILE_FEATURES_CSV="${TILE_FEATURES_CSV:-reports/p7_5_screening/20260726T114749Z/screening/joined_training_rows.csv}"

# Baseline = the ORIGINAL P7-3 seed-3407 run (earliest report directory;
# UTC tags sort chronologically). Same seed + same split salt -> paired
# comparison where the feature set is the only difference.
if [[ -z "${BASELINE_METRICS:-}" ]]; then
  baseline_dir="$(ls -d reports/p7_safe_alignment_classifier_levir/2*/ | head -n 1)"
  BASELINE_METRICS="${baseline_dir%/}/safe_alignment_experiment/validation_scenario_metrics.csv"
fi

for required in \
  "$P7_CHECKPOINT" \
  "$FEATURES_CSV" \
  "$TASK_ROUTER_CHECKPOINT" \
  "$TILE_FEATURES_CSV" \
  "$BASELINE_METRICS" \
  manifests/datasets/levir_cd.csv \
  configs/p7_5_expanded_classifier.yaml; do
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
report_root="reports/p7_5_expanded_classifier_levir/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p7_5_expanded_classifier_levir/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_CHECKPOINT"
printf '%s\n' "$FEATURES_CSV" > "$report_root/SOURCE_FEATURES_CSV"
printf '%s\n' "$TILE_FEATURES_CSV" \
  > "$report_root/SOURCE_TILE_FEATURES_CSV"
printf '%s\n' "$BASELINE_METRICS" > "$report_root/SOURCE_BASELINE"
sha256sum "$FEATURES_CSV" "$TILE_FEATURES_CSV" "$BASELINE_METRICS" \
  > "$report_root/SOURCES.sha256"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

seed_args=()
if [[ -n "$SEED" ]]; then
  seed_args=(--seed "$SEED")
fi

python -u -m tools.train_evaluate_safe_alignment_classifier \
  --checkpoint "$P7_CHECKPOINT" \
  --features-csv "$FEATURES_CSV" \
  --tile-features-csv "$TILE_FEATURES_CSV" \
  --baseline-scenario-metrics "$BASELINE_METRICS" \
  --task-router-checkpoint "$TASK_ROUTER_CHECKPOINT" \
  --protocol configs/p7_5_expanded_classifier.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --device cuda \
  ${seed_args[@]+"${seed_args[@]}"} \
  --output-dir "$report_root/expanded_experiment" \
  2>&1 | tee "$report_root/expanded_experiment.log"

cp "$report_root/expanded_experiment/decision.json" \
  "$report_root/decision.json"

echo "P7-5 expanded classifier complete: $report_root"
cat "$report_root/decision.json"
