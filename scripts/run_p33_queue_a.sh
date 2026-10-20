#!/usr/bin/env bash
set -uo pipefail

# P33 queue A (docs/P33_REVIEWER_EXPERIMENTS.md): RoMa, 3 DoF, RANSAC 3 px, + the paper's selector recipe on the twelve
# combinations of Table 6 (DINOv3, BIT, CD-Lamba, CDxLSTM x LEVIR-CD, DSIFN-CD, WHU-CD). Two worker processes claim
# datasets in turn (mkdir is atomic), so the long RoMa passes run side by side; RoMa estimates are cached per pair.
# Resumable: finished files are kept; to redo a dataset whose worker died, remove its claim directory.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

LOGS=reports/p33/logs
CLAIMS=reports/p33/selector/roma_ts_ransac/claims
mkdir -p "$LOGS" "$CLAIMS"

worker() {
  local label="$1"; shift
  local status=0
  for dataset in "$@"; do
    if mkdir "$CLAIMS/$dataset" 2>/dev/null; then
      echo "[$label] $(date -u +%FT%TZ) start $dataset"
      python -u -m tools.p33_selector_runs pipeline --localizer roma_ts_ransac \
        --rows dinov3 bit cdlamba cdxlstm --datasets "$dataset" || { echo "[$label] $dataset FAILED"; status=1; }
      echo "[$label] $(date -u +%FT%TZ) end $dataset"
    fi
  done
  return "$status"
}

worker w1 LEVIR-CD DSIFN-CD WHU-CD > "$LOGS/queue_a_w1.log" 2>&1 &
first=$!
sleep 60
worker w2 WHU-CD DSIFN-CD LEVIR-CD > "$LOGS/queue_a_w2.log" 2>&1 &
second=$!
wait "$first"; s1=$?
wait "$second"; s2=$?
echo "P33 queue A finished: worker 1 status $s1, worker 2 status $s2 ($(date -u +%FT%TZ))"
