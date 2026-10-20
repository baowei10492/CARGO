#!/usr/bin/env bash
set -uo pipefail

# Single 48 GB server (2026-10-09): queue C ends with RoMa on identical pairs, which must not load a further RoMa next
# to queue A's two workers while the two retrainings of queue B hold their memory. If that step starts while a
# retraining runs, it is paused (SIGSTOP) and resumed when no retraining is left; nothing else is touched.

# queue C's command exactly (a looser pattern also caught a 4-crop smoke run of the identity command)
STEP="^python -u -m tools\.p33_selector_runs identity --roma-crops 300 --output-root reports/p33/identity_roma"
TRAINING="^python -u -m tools\.train_similarity_localizer"
echo "[gate] $(date -u +%FT%TZ) watching"
paused=0
while pgrep -f "$TRAINING" > /dev/null; do
  if [ "$paused" -eq 0 ] && pgrep -f "$STEP" > /dev/null; then
    pkill -STOP -f "$STEP" && paused=1 && echo "[gate] $(date -u +%FT%TZ) identity step paused"
  fi
  sleep 5
done
if [ "$paused" -eq 1 ]; then
  pkill -CONT -f "$STEP" && echo "[gate] $(date -u +%FT%TZ) identity step resumed"
fi
echo "[gate] $(date -u +%FT%TZ) retrainings ended; done"
