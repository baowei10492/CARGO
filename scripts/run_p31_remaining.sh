#!/usr/bin/env bash
set -uo pipefail

# The two unfinished parts of the overnight session (main record 192.73), in parallel on one RTX 4090, about 2.5 hours:
#   swapped   DiffRegCD's input convention, rows not finished yet (BIT, CD-Lamba, CDxLSTM; DINOv3 is kept)
#   errors    per-crop registration errors, variants not finished yet (v2_full is kept and re-checked)
# Waits for both, then packs the overnight and swapped directories into ../results_remaining.tgz.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

overnight="reports/p31/overnight/run"
swapped="reports/p31/swapped/run"
mkdir -p "$overnight" "$swapped"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31_overnight.py tests/test_p31_swapped.py 2>&1 | tee "$overnight/pytest_remaining.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "tests failed" >&2; exit 1; }

echo "===== SWAPPED CONVENTION (background) ====="
python -u -m tools.p31_swapped_convention --output-dir "$swapped" >> "$swapped/swapped.log" 2>&1 &
swapped_pid=$!

echo "===== PER-CROP REGISTRATION ERRORS ====="
python -u -m tools.p31_registration_errors --output-dir "$overnight/errors" 2>&1 | tee -a "$overnight/errors.log"
errors_status="${PIPESTATUS[0]}"

echo "===== WAITING FOR THE SWAPPED CONVENTION ====="
wait "$swapped_pid"
swapped_status=$?
grep -E "raw|Error|error" "$swapped/swapped.log" | tail -n 16

tar czf ../results_remaining.tgz "$overnight" "$swapped"
ls -lh ../results_remaining.tgz
echo "P31 remaining parts finished: errors status $errors_status, swapped status $swapped_status"
test "$errors_status" -eq 0 && test "$swapped_status" -eq 0
