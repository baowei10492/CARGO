#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
ROOT="$PWD/reports/round1_20261103/model_mismatch_stress_v2"
RUNNERS="$PWD/tools/round1_revision_20261103"
SEEDS="$PWD/reports/round1_20261103/WHU_detector_seeds"
mkdir -p "$ROOT"
exec 9>"$ROOT/queue.lock"
flock -n 9 || { echo 'Stress queue is already running.' >&2; exit 1; }
exec 8>"$SEEDS/queue.lock"
flock -n 8 || { echo 'WHU seed queue is still running; retry after completion.' >&2; exit 1; }
if [[ ! -f "$SEEDS/COMPLETE.json" ]]; then
  echo 'WHU seed queue has no COMPLETE.json; do not run concurrently.' >&2
  exit 1
fi
GPU_USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ ! "$GPU_USED" =~ ^[0-9]+$ ]] || (( GPU_USED > 2048 )); then
  echo "GPU not idle (used=${GPU_USED} MiB); no experiment started." >&2
  exit 1
fi
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
trap 'rc=$?; if (( rc != 0 )); then printf "QUEUE_FAILED exit=%s date=%s\n" "$rc" "$(date -Is)"; fi' EXIT
printf 'QUEUE_START E2_STRESS %s\n' "$(date -Is)"
.venv/bin/python -u "$RUNNERS/round1_stress.py" \
  --project "$PWD" --protocol "$RUNNERS/round1_stress_protocol.json" --output "$ROOT"
printf 'QUEUE_DONE E2_STRESS %s\n' "$(date -Is)"
