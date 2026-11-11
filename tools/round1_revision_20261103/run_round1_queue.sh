#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/rs_cd_work/rs_cd_project
TASK_ROOT="${TASK_ROOT:-$PWD/reports/round1_20261103}"
RUNNER_DIR="$PWD/tools/round1_revision_20261103"
mkdir -p "$TASK_ROOT"
exec 9>"$TASK_ROOT/queue.lock"
if ! flock -n 9; then
  echo 'Another round1 queue is already running.' >&2
  exit 1
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "FAILED exit=%s date=%s\n" "$rc" "$(date -Is)"; fi' EXIT
printf 'QUEUE_START %s\n' "$(date -Is)"
for stage in E3 E2 E1 E7; do
  printf 'STAGE_START %s %s\n' "$stage" "$(date -Is)"
  .venv/bin/python -u "$RUNNER_DIR/round1_formal.py" \
    --project "$PWD" --protocol "$RUNNER_DIR/round1_protocol.json" \
    --output "$TASK_ROOT" --stage "$stage"
  printf 'STAGE_DONE %s %s\n' "$stage" "$(date -Is)"
done
printf 'QUEUE_DONE %s\n' "$(date -Is)"
