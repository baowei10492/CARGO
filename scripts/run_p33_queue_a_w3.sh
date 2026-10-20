#!/usr/bin/env bash
set -uo pipefail

# P33 queue A, third worker (single 48 GB server, 2026-10-09): RoMa + the selector recipe on DSIFN-CD once both
# retrainings of queue B have ended and freed their GPU memory (a third RoMa worker next to the two trainings does not
# fit). Claims DSIFN-CD like the queue A workers; does nothing if one of them has claimed it already.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

CLAIMS=reports/p33/selector/roma_ts_ransac/claims
echo "[w3] $(date -u +%FT%TZ) waiting for the retrainings to end"
while pgrep -f "^python -u -m tools\.train_similarity_localizer" > /dev/null; do sleep 300; done
if mkdir "$CLAIMS/DSIFN-CD" 2>/dev/null; then
  echo "[w3] $(date -u +%FT%TZ) start DSIFN-CD"
  python -u -m tools.p33_selector_runs pipeline --localizer roma_ts_ransac \
    --rows dinov3 bit cdlamba cdxlstm --datasets DSIFN-CD || { echo "[w3] DSIFN-CD FAILED"; exit 1; }
  echo "[w3] $(date -u +%FT%TZ) end DSIFN-CD"
else
  echo "[w3] $(date -u +%FT%TZ) DSIFN-CD already claimed"
fi
