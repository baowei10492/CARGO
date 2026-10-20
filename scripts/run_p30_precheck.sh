#!/usr/bin/env bash
set -euo pipefail

# P30 pre-check (docs/P30_PRECHECK.md): the frozen P27 localizer and the original
# DINOv3 detectors on validation crops, with and without the zero border.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

root="reports/p30_precheck/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"
python -m pytest -q tests/test_p30_precheck.py 2>&1 | tee "$root/pytest.log"
python -u -m tools.p30_precheck --output "$root/precheck.json" 2>&1 | tee "$root/precheck.log"
sha256sum docs/P30_PRECHECK.md tools/p30_precheck.py "$root/precheck.json" > "$root/SHA256SUMS.txt"
echo "P30 pre-check: $root"
