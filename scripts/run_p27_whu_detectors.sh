#!/usr/bin/env bash
set -uo pipefail

# P27 WHU-CD rows of the P21 detectors (docs/P27_EXPERIMENTS_AMENDMENT.md):
# train each detector with the P21 recipe on WHU-CD, write the pointer file
# the plug-in configs read, then run the whole plug-in row. A detector whose
# training fails or whose best validation F1 is below 0.5 is reported and the
# next one still runs. DETECTORS overrides the order (fastest first).

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

train_one() {
  local detector="$1"
  local config="configs/p21/${detector}_whu.yaml"
  local pointer_dir="outputs/p27_whu_detectors/$detector"
  if [ -f "$pointer_dir/CHECKPOINT" ]; then
    echo "$detector: already trained ($(cat "$pointer_dir/CHECKPOINT"))"
    return 0
  fi
  mkdir -p "$pointer_dir"
  local marker="$pointer_dir/.before_training"
  touch "$marker"
  echo "===== TRAIN $detector ON WHU-CD ====="
  python -u -m tools.train --config "$config" --device cuda 2>&1 | tee "$pointer_dir/train.log" || return 1
  local summary
  summary="$(find "outputs/p21_${detector}_whu" -name summary.json -newer "$marker" | sort | tail -n 1)"
  test -n "$summary" || { echo "$detector: no summary.json" >&2; return 1; }
  python - "$summary" "$pointer_dir" "$PROJECT_ROOT" <<'PY' || return 1
import json
import math
import sys
from pathlib import Path

summary_path, pointer_dir, project_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
best = summary.get("best_f1")
if summary.get("status") != "completed" or best is None or not math.isfinite(float(best)) or float(best) < 0.5:
    raise SystemExit(f"training did not produce a usable detector: status {summary.get('status')}, best F1 {best}")
checkpoint = summary_path.parent / "checkpoints" / "best.pth"
if not checkpoint.is_file():
    raise SystemExit(f"missing {checkpoint}")
relative = checkpoint.resolve().relative_to(project_root)
(pointer_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(pointer_dir / "CHECKPOINT").write_text(f"{relative.as_posix()}\n", encoding="utf-8")
print(f"best validation F1 {float(best):.4f} at epoch {summary.get('best_epoch')}; pointer -> {relative}")
PY
  sha256sum "$config" "$(cat "$pointer_dir/CHECKPOINT")" > "$pointer_dir/SHA256SUMS.txt"
}

for detector in ${DETECTORS:-bit cdxlstm cdlamba}; do
  echo "########## ${detector}_whu ##########"
  if train_one "$detector" && bash scripts/run_p27_plugin.sh "${detector}_whu" all; then
    echo "########## ${detector}_whu: finished ##########"
  else
    echo "########## ${detector}_whu: FAILED ##########"
  fi
done
