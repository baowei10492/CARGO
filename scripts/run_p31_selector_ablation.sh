#!/usr/bin/env bash
set -euo pipefail

# P31 selector ablation under the border-free protocol (tools/p31_selector_ablation.py), CPU only.
# Run on the server that trained the border-free selectors (P30 and the P31 plug-in rows), after
# scripts/run_p31_plugins.sh has finished: the retrained full selector must reproduce the frozen one.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for row in bit cdlamba cdxlstm; do
  test -f "reports/p31/$row/test/p31_test.json" || { echo "row $row has no test run yet" >&2; exit 1; }
done
test -f reports/p30/test/LATEST && test -f reports/p30/selector/LATEST
mkdir -p reports/p31/selector_ablation

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31_selector_ablation.py tests/test_p27_ablation.py 2>&1 | tee reports/p31/selector_ablation/pytest.log

echo "===== SELECTOR ABLATION (border-free rows) ====="
python -u -m tools.p31_selector_ablation 2>&1 | tee reports/p31/selector_ablation/selector_ablation.log
echo "P31 selector ablation complete: reports/p31/selector_ablation"
