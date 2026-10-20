#!/usr/bin/env bash
set -euo pipefail

# P27 F1 step (docs/P27_F1_AMENDMENT.md): the frozen P27 localizer in front of
# the frozen DINOv3 detectors on every validation crop. This is the
# checkpoint's single use of the validation crops.
# CHECKPOINT defaults to best.pth of the latest training run.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_f1_gate.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
test -f docs/P27_F1_AMENDMENT.md
run_root="${RUN_ROOT:-$(cat outputs/p27_localizer/LATEST)}"
run_root="${run_root%/}"
CHECKPOINT="${CHECKPOINT:-$run_root/checkpoints/best.pth}"
test -f "$CHECKPOINT"
test -f "$run_root/config.yaml"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/f1/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_f1.py tests/test_p27_step1.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P27 F1 ($CHECKPOINT) ====="
python -u -m tools.run_p27_f1 \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --training-config "$run_root/config.yaml" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/f1.log"

sha256sum "$CONFIG" configs/p27_eval_perturbation.yaml "$CHECKPOINT" "$run_root/config.yaml" \
  "$report_root"/run/* > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/f1/LATEST
echo "P27 F1 complete: $report_root"
