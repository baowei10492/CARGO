#!/usr/bin/env bash
set -euo pipefail

# P31 localizer ablation under the border-free protocol (docs/P31_BORDER_FREE_REMAINDER.md), test split,
# reporting only. Resumable: ABLATION_ROOT=<earlier output dir> keeps the finished variants.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p31_border_free.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG"
test -f docs/P31_BORDER_FREE_REMAINDER.md
test -f outputs/p30_localizer/LATEST
test -f reports/p30/test/LATEST

base="reports/p31/localizer_ablation"
root="${ABLATION_ROOT:-$base/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$root"
printf '%s\n' "$root" > "$base/LATEST"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31.py 2>&1 | tee -a "$root/pytest.log"

echo "===== LOCALIZER ABLATION (border-free, test) -> $root ====="
python -u -m tools.p31_localizer_ablation --config "$CONFIG" --output-dir "$root" 2>&1 | tee -a "$root/ablation.log"
sha256sum "$CONFIG" "$root"/*.json > "$root/SHA256SUMS.txt"
echo "P31 localizer ablation complete: $root"
