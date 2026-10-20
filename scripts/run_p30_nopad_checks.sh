#!/usr/bin/env bash
set -euo pipefail

# P30 no-padding check (docs/P30_NOPAD_CHECK.md): the P29b diagnosis and the P30
# pre-check again, on validation crops that keep 64 real pixels to every tile edge.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

for dataset in levir dsifn whu; do
  sha256sum -c "outputs/p29_aug_detectors/$dataset/SHA256SUMS.txt"
done
root="reports/p30_nopad/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"
python -m pytest -q tests/test_p30_margin_crops.py tests/test_p29_border_diagnosis.py tests/test_p30_precheck.py \
  2>&1 | tee "$root/pytest.log"
echo "===== DIAGNOSIS (margin 64) ====="
python -u -m tools.p29_border_diagnosis --margin 64 --output "$root/diagnosis.json" 2>&1 | tee "$root/diagnosis.log"
echo "===== PRE-CHECK (margin 64) ====="
python -u -m tools.p30_precheck --margin 64 --output "$root/precheck.json" 2>&1 | tee "$root/precheck.log"
sha256sum docs/P30_NOPAD_CHECK.md tools/p29_border_diagnosis.py tools/p30_precheck.py "$root"/*.json \
  > "$root/SHA256SUMS.txt"
echo "P30 no-padding check: $root"
