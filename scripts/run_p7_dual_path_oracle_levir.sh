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
  configs/p7_dual_path_task_oracle.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p7_dual_path_oracle_levir/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p7_dual_path_oracle_levir/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/CHECKPOINT"

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

python -u -m tools.evaluate_dual_path_oracle \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p7_dual_path_task_oracle.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --split val \
  --device cuda \
  --output-dir "$report_root/oracle_val" \
  2>&1 | tee "$report_root/oracle_val.log"

cp "$report_root/oracle_val/decision.json" \
  "$report_root/decision.json"

echo "P7 dual-path task Oracle complete: $report_root"
cat "$report_root/decision.json"
