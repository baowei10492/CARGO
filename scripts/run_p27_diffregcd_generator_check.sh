#!/usr/bin/env bash
set -euo pipefail

# DiffRegCD on its own perturbation generator (docs/P27_DIFFREGCD_GENERATOR_CHECK.md),
# in the DiffRegCD environment, with the weights and seeds of the P27 DiffRegCD run.
# About 50 minutes on one RTX 4090.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_diffregcd.yaml}"

cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
EXTERNAL_ROOT="$("$PROJECT_PYTHON" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['external']['root'])" "$CONFIG")"
EXTERNAL_PYTHON="$EXTERNAL_ROOT/.venv/bin/python"
grep -q "^status: preregistered" "$CONFIG"
test -f docs/P27_DIFFREGCD_GENERATOR_CHECK.md
test -x "$EXTERNAL_PYTHON"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$EXTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=disabled
export DIFFREGCD_ROOT="$EXTERNAL_ROOT"

root="reports/p27_localizer/diffregcd/generator_check"
tag="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"

echo "===== TARGETED TESTS (project environment) ====="
"$PROJECT_PYTHON" -m pytest -q tests/test_p27_diffregcd_generator_check.py 2>&1 | tee "$root/pytest_$tag.log"

echo "===== DIFFREGCD ON ITS OWN GENERATOR (DiffRegCD environment) ====="
"$EXTERNAL_PYTHON" -u -m tools.p27_diffregcd_generator_check --config "$CONFIG" --output-dir "$root/$tag" \
  2>&1 | tee "$root/$tag.log"
sha256sum "$CONFIG" docs/P27_DIFFREGCD_GENERATOR_CHECK.md "$root/$tag"/*.json > "$root/$tag/SHA256SUMS.txt"
printf '%s\n' "$root/$tag" > "$root/LATEST"
echo "DiffRegCD generator check complete: $root/$tag"
