#!/usr/bin/env bash
set -euo pipefail

# P27 baseline row: RoMa + similarity fit in the P25 step 1 harness with the
# rotation fixed at 0 (configs/p27_eval_perturbation.yaml). Validation crops,
# label-free apart from the descriptive change-fraction column.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_eval_perturbation.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
test -f docs/P27_LOCALIZER_EVALUATION_PROTOCOL.md

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/roma_baseline/$tag"
mkdir -p "$report_root"

python -u -m tools.run_p25_affine_step1 \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/step1.log"

sha256sum "$CONFIG" "$report_root"/run/*.json "$report_root"/run/*.csv > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/roma_baseline/LATEST
echo "P27 RoMa baseline complete: $report_root"
python -m tools.print_step1_summary "$report_root/run/step1_decision.json"
