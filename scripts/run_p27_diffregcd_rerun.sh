#!/usr/bin/env bash
set -euo pipefail

# DiffRegCD re-run with the correct wiring (docs/P27_DIFFREGCD_RERUN.md), in the DiffRegCD
# environment: mechanism (~40 min) -> border_free (~2.3 h) -> bordered (~1.8 h).
# Resumable: RERUN_ROOT=<earlier output dir> continues where it stopped.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_diffregcd_rerun.yaml}"

cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
DIFFREGCD_CONFIG="$("$PROJECT_PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['diffregcd_config'])" "$CONFIG")"
EXTERNAL_ROOT="$("$PROJECT_PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['external']['root'])" "$DIFFREGCD_CONFIG")"
EXTERNAL_PYTHON="$EXTERNAL_ROOT/.venv/bin/python"
grep -q "^status: preregistered" "$CONFIG"
test -f docs/P27_DIFFREGCD_RERUN.md
test -x "$EXTERNAL_PYTHON"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$EXTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=disabled

base="reports/p27_localizer/diffregcd/rerun"
root="${RERUN_ROOT:-$base/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$root"
printf '%s\n' "$root" > "$base/LATEST"

echo "===== TARGETED TESTS (project environment) ====="
"$PROJECT_PYTHON" -m pytest -q tests/test_p27_diffregcd_rerun.py tests/test_p30_evaluation.py 2>&1 | tee -a "$root/pytest.log"

echo "===== DIFFREGCD RE-RUN (DiffRegCD environment) -> $root ====="
"$EXTERNAL_PYTHON" -u -m tools.p27_diffregcd_rerun --config "$CONFIG" --output-dir "$root" 2>&1 | tee -a "$root/rerun.log"
sha256sum "$CONFIG" "$DIFFREGCD_CONFIG" docs/P27_DIFFREGCD_RERUN.md "$root"/*/*.json "$root"/summary.json > "$root/SHA256SUMS.txt"
echo "DiffRegCD re-run complete: $root"
