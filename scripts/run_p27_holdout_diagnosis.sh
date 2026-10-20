#!/usr/bin/env bash
set -euo pipefail

# P27 development diagnosis on the training hold-out (tools/diagnose_p27_holdout.py).
# No validation or test data; not a gate. RUN_ROOT defaults to the newest
# training run, CHECKPOINT_NAME to best.pth.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pth}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

run_root="${RUN_ROOT:-$(ls -td outputs/p27_localizer/2*/ | head -n 1)}"
run_root="${run_root%/}"
test -f "$run_root/config.yaml"
test -f "$run_root/checkpoints/$CHECKPOINT_NAME"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/holdout_diagnosis/$tag"
mkdir -p "$(dirname "$report_root")"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_diagnosis.py

echo "===== P27 HOLD-OUT DIAGNOSIS ($run_root, $CHECKPOINT_NAME) ====="
python -u -m tools.diagnose_p27_holdout \
  --run-dir "$run_root" \
  --checkpoint "$CHECKPOINT_NAME" \
  --device cuda \
  --output-dir "$report_root" \
  2>&1 | tee "$report_root.log"
mv "$report_root.log" "$report_root/diagnosis.log"
sha256sum "$run_root/checkpoints/$CHECKPOINT_NAME" "$report_root"/*.csv "$report_root"/*.json > "$report_root/SHA256SUMS.txt"
echo "P27 hold-out diagnosis complete: $report_root"
