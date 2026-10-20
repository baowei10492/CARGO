#!/usr/bin/env bash
set -uo pipefail

# Every P27 plug-in row in turn (docs/P27_PLUGIN_AMENDMENT.md). A row that
# fails, for example because its weights are not on this machine, is reported
# and the next row still runs. DETECTORS overrides the list.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
for detector in ${DETECTORS:-bit cdlamba cdxlstm}; do
  echo "########## $detector ##########"
  if bash scripts/run_p27_plugin.sh "$detector" all; then
    echo "########## $detector: finished ##########"
  else
    echo "########## $detector: FAILED (exit $?) ##########"
  fi
done
