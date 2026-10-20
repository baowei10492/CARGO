#!/usr/bin/env bash
set -euo pipefail

# Start the three P33 queues in the background (docs/P33_REVIEWER_EXPERIMENTS.md). Each survives the SSH session.
#   bash scripts/run_p33_all.sh
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
mkdir -p reports/p33/logs
for queue in a b c; do
  nohup bash "scripts/run_p33_queue_$queue.sh" > "reports/p33/logs/queue_$queue.out" 2>&1 &
  echo "queue $queue: pid $! -> reports/p33/logs/queue_$queue.out"
  sleep 20
done
