#!/usr/bin/env bash
set -uo pipefail

# P31 plug-in rows under the border-free protocol (docs/P31_BORDER_FREE_REMAINDER.md):
# per row features -> selector training -> validation -> test; finished steps are skipped,
# so the script can be restarted. ROWS limits the rows (default: bit cdxlstm cdlamba).

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p31_border_free.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || { echo "$CONFIG is not marked preregistered" >&2; exit 1; }
test -f docs/P31_BORDER_FREE_REMAINDER.md
test -f outputs/p30_localizer/LATEST || { echo "localizer v2 missing (outputs/p30_localizer/LATEST)" >&2; exit 1; }
mkdir -p reports/p31

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31.py tests/test_p30_evaluation.py 2>&1 | tee -a reports/p31/pytest.log
test "${PIPESTATUS[0]}" -eq 0 || { echo "tests failed" >&2; exit 1; }

failed=0
for row in ${ROWS:-bit cdxlstm cdlamba}; do
  echo "===== P31 ROW $row ====="
  if python -u -m tools.run_p31_plugins all --row "$row" --config "$CONFIG" 2>&1 | tee -a "reports/p31/$row.log"; then
    echo "########## $row: done ##########"
  else
    echo "########## $row: FAILED ##########"
    failed=1
  fi
done
echo "P31 plug-in rows finished with status $failed"
exit "$failed"
