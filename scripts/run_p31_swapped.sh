#!/usr/bin/env bash
set -uo pipefail

# DiffRegCD's input convention for the P30/P31 rows (main record 192.72): T1 perturbed, T2 reference, label in the
# T2 frame. Four rows x three datasets on the test crops, about 1 hour on one RTX 4090 (longer while sharing the
# GPU). Can run next to scripts/run_p31_overnight.sh. Packed into ../results_swapped.tgz.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

root="${SWAPPED_ROOT:-reports/p31/swapped/run}"
mkdir -p "$root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31_swapped.py 2>&1 | tee "$root/pytest.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "tests failed" >&2; exit 1; }

echo "===== SWAPPED CONVENTION ====="
python -u -m tools.p31_swapped_convention --output-dir "$root" 2>&1 | tee -a "$root/swapped.log"
status="${PIPESTATUS[0]}"

tar czf ../results_swapped.tgz "$root"
ls -lh ../results_swapped.tgz
echo "P31 swapped convention finished with status $status: $root"
exit "$status"
