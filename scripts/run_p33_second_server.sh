#!/usr/bin/env bash
set -uo pipefail

# P33 on the second server (port 31352, vGPU 32 GB, 12 CPU cores; 2026-10-09), taken over from the main server
# (port 36611), which keeps LEVIR-CD, the retrainings, queue C and DiffRegCD:
#   - queue A's WHU-CD and DSIFN-CD: RoMa + the selector recipe on the four detectors, from the RoMa estimates the
#     main server had already cached
#   - queue B's S2Looking part: the DINOv3 detector with the paper's recipe, then the selector pipeline
# The results are copied to the main server before its queue B reaches S2Looking, so that it skips it.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
LOGS=reports/p33/logs
mkdir -p "$LOGS"

roma() {
  echo "[roma $1] $(date -u +%FT%TZ) start"
  python -u -m tools.p33_selector_runs pipeline --localizer roma_ts_ransac --rows dinov3 bit cdlamba cdxlstm \
    --datasets "$1" > "$LOGS/second_roma_$1.log" 2>&1 || { echo "[roma $1] FAILED"; return 1; }
  echo "[roma $1] $(date -u +%FT%TZ) end"
}

# queue B's train_s2looking_detector, verbatim
train_s2looking_detector() {
  local pointer_dir="outputs/p33_s2looking_detector"
  if [ -f "$pointer_dir/CHECKPOINT" ]; then echo "S2Looking detector: trained ($(cat "$pointer_dir/CHECKPOINT"))"; return 0; fi
  mkdir -p "$pointer_dir"
  touch "$pointer_dir/.before_training"
  echo "===== TRAIN DINOv3 DETECTOR ON S2Looking ($(date -u +%FT%TZ)) ====="
  python -u -m tools.train --config configs/p33/dinov3_s2looking.yaml --device cuda > "$pointer_dir/train.log" 2>&1 \
    || { echo "S2Looking detector: training failed"; return 1; }
  local summary
  summary="$(find outputs/p33_dinov3_s2looking -name summary.json -newer "$pointer_dir/.before_training" | sort | tail -n 1)"
  test -n "$summary" || { echo "S2Looking detector: no summary.json"; return 1; }
  python - "$summary" "$pointer_dir" "$PROJECT_ROOT" <<'PY' || return 1
import json
import math
import sys
from pathlib import Path

summary_path, pointer_dir, project_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
best = summary.get("best_f1")
if summary.get("status") != "completed" or best is None or not math.isfinite(float(best)):
    raise SystemExit(f"training did not produce a usable detector: status {summary.get('status')}, best F1 {best}")
checkpoint = summary_path.parent / "checkpoints" / "best.pth"
relative = checkpoint.resolve().relative_to(project_root)
(pointer_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(pointer_dir / "CHECKPOINT").write_text(f"{relative.as_posix()}\n", encoding="utf-8")
print(f"S2Looking detector: best validation F1 {float(best):.4f} at epoch {summary.get('best_epoch')} -> {relative}")
PY
  sha256sum configs/p33/dinov3_s2looking.yaml "$(cat "$pointer_dir/CHECKPOINT")" > "$pointer_dir/SHA256SUMS.txt"
}

roma WHU-CD &
first=$!
sleep 60
roma DSIFN-CD &
second=$!
sleep 60
status=0
if train_s2looking_detector; then
  echo "===== S2Looking SELECTOR AND TEST ($(date -u +%FT%TZ)) ====="
  python -u -m tools.p33_selector_runs pipeline --localizer v2 --s2looking > "$LOGS/second_s2looking.log" 2>&1 \
    && echo "S2Looking: done ($(date -u +%FT%TZ))" || { echo "S2Looking: pipeline FAILED"; status=1; }
else
  status=1
fi
wait "$first" || status=1
wait "$second" || status=1
echo "P33 second server finished with status $status ($(date -u +%FT%TZ))"
exit "$status"
