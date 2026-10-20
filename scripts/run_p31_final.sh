#!/usr/bin/env bash
set -uo pipefail

# Final measurements (main record 192.66), one run of about 15 minutes on one RTX 4090:
#   border_counts   DINOv3 detector and both augmentation baselines on the P30 test crops: clean, border-free
#                   and per-crop zero-border views, per-crop counts; must reproduce the P30 test rows
#   qualitative     border-free panels and tiles (P30 test run, localizer v2, P30 selectors)
#   timing          DiffRegCD's released test path, time per 256 px pair (DiffRegCD environment)
# Everything is packed into ../results_final_20260920.tgz for download.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

test -f reports/p30/test/LATEST && test -f reports/p30/selector/LATEST && test -f outputs/p30_localizer/LATEST
root="reports/p31/final/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"
printf '%s\n' "$root" > reports/p31/final/LATEST

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31_final.py tests/test_p30_evaluation.py 2>&1 | tee "$root/pytest.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "tests failed" >&2; exit 1; }

failed=0
echo "===== SAME-CROP BORDER COMPARISON AND PER-CROP COUNTS ====="
python -u -m tools.p31_border_counts --output-dir "$root/border_counts" 2>&1 | tee "$root/border_counts.log" || failed=1

echo "===== BORDER-FREE QUALITATIVE PANELS ====="
python -u -m tools.p31_qualitative --output-dir "$root/qualitative" 2>&1 | tee "$root/qualitative.log" || failed=1

echo "===== DIFFREGCD TIMING (DiffRegCD environment) ====="
EXTERNAL_ROOT="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['external']['root'])" configs/p27_diffregcd.yaml)"
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$EXTERNAL_ROOT" WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$EXTERNAL_ROOT/.venv/bin/python" -u -m tools.p31_diffregcd_timing --output "$root/diffregcd_timing.json" \
  2>&1 | tee "$root/diffregcd_timing.log" || failed=1

tar czf ../results_final_20260920.tgz "$root"
ls -lh ../results_final_20260920.tgz
echo "P31 final measurements finished with status $failed: $root"
exit "$failed"
