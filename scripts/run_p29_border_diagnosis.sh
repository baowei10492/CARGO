#!/usr/bin/env bash
set -euo pipefail

# P29b border-cue diagnosis (docs/P29B_BORDER_CUE_DIAGNOSIS.md), validation crops only.
# Needs the P29 augmented detectors (outputs/p29_aug_detectors/<dataset>/CHECKPOINT).

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

for dataset in levir dsifn whu; do
  sha256sum -c "outputs/p29_aug_detectors/$dataset/SHA256SUMS.txt"
done
root="reports/p29_border_diagnosis/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$root"
python -m pytest -q tests/test_p29_border_diagnosis.py 2>&1 | tee "$root/pytest.log"
python -u -m tools.p29_border_diagnosis --output "$root/diagnosis.json" 2>&1 | tee "$root/diagnosis.log"
sha256sum docs/P29B_BORDER_CUE_DIAGNOSIS.md tools/p29_border_diagnosis.py "$root/diagnosis.json" > "$root/SHA256SUMS.txt"
echo "P29b diagnosis: $root"
