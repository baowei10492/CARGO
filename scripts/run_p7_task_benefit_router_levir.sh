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

for required in \
  "$P7_CHECKPOINT" \
  manifests/datasets/levir_cd.csv \
  configs/p7_task_benefit_router.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p7_task_benefit_router_levir/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p7_task_benefit_router_levir/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_CHECKPOINT"

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

python -u -m tools.train_evaluate_task_benefit_router \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p7_task_benefit_router.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --device cuda \
  --output-dir "$report_root/router_experiment" \
  2>&1 | tee "$report_root/router_experiment.log"

cp "$report_root/router_experiment/decision.json" \
  "$report_root/decision.json"

echo "P7 task-benefit router complete: $report_root"
cat "$report_root/decision.json"
