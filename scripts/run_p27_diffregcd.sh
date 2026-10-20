#!/usr/bin/env bash
set -euo pipefail

# DiffRegCD on the P27 affine protocol (configs/p27_diffregcd.yaml), in the
# DiffRegCD environment. Resumable per dataset; DATASETS limits the list.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_diffregcd.yaml}"

cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
EXTERNAL_ROOT="$("$PROJECT_PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['external']['root'])" "$CONFIG")"
EXTERNAL_PYTHON="$EXTERNAL_ROOT/.venv/bin/python"
grep -q "^status: preregistered" "$CONFIG"
test -x "$EXTERNAL_PYTHON"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$EXTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=disabled

report_root="reports/p27_localizer/diffregcd/test"
mkdir -p "$report_root"

echo "===== TARGETED TESTS (project environment) ====="
"$PROJECT_PYTHON" -m pytest -q tests/test_p27_experiments.py 2>&1 | tee "$report_root/pytest.log"

echo "===== DIFFREGCD ON THE P27 PROTOCOL (DiffRegCD environment) ====="
"$EXTERNAL_PYTHON" -u -m tools.run_p27_diffregcd_f1 --config "$CONFIG" --output-dir "$report_root" \
  ${DATASETS:+--datasets $DATASETS} 2>&1 | tee -a "$report_root/diffregcd.log"
sha256sum "$CONFIG" "$report_root"/*.json > "$report_root/SHA256SUMS.txt"
echo "P27 DiffRegCD complete: $report_root"
