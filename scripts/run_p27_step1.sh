#!/usr/bin/env bash
set -euo pipefail

# P27 E1 (docs/P27_LOCALIZER_EVALUATION_PROTOCOL.md): a frozen P27 localizer
# on the validation crops, once per checkpoint, at most two checkpoints ever.
# CHECKPOINT defaults to best.pth of the latest training run.

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
run_root="${RUN_ROOT:-$(cat outputs/p27_localizer/LATEST)}"
CHECKPOINT="${CHECKPOINT:-$run_root/checkpoints/best.pth}"
test -f "$CHECKPOINT"
test -f "$run_root/config.yaml"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/step1/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_step1.py tests/test_p27_localizer.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P27 E1 ($CHECKPOINT) ====="
python -u -m tools.run_p27_step1 \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --training-config "$run_root/config.yaml" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/step1.log"

sha256sum "$CONFIG" "$CHECKPOINT" "$run_root/config.yaml" "$report_root"/run/* > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/step1/LATEST
echo "P27 E1 complete: $report_root"
python -m tools.print_step1_summary "$report_root/run/step1_decision.json"
