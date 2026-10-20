#!/usr/bin/env bash
set -uo pipefail

# P29 augmentation baseline (docs/P29_AUGMENTATION_BASELINE.md):
#   check   python -m tools.p29_preflight (files, hashes, GPU memory); nothing is written
#   train   retrain the DINOv3 detector of each dataset with test-matched misalignment
#           augmentation (configs/p29/dinov3_aug_<dataset>.yaml) and write the pointer file
#           outputs/p29_aug_detectors/<dataset>/CHECKPOINT (skipped when it exists)
#   module  the whole P27 plug-in row "dinov3aug": the augmented detectors without correction
#           and with the P27 localizer and selector (F1 step, selector, validation, test once)
#   all     check -> train -> module (default)
# DATASETS overrides the training order (fastest first).

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
STAGE="${1:-all}"

stage_check() {
  python -u -m tools.p29_preflight
}

train_one() {
  local dataset="$1"
  local config="configs/p29/dinov3_aug_${dataset}.yaml"
  local pointer_dir="outputs/p29_aug_detectors/$dataset"
  if [ -f "$pointer_dir/CHECKPOINT" ]; then
    echo "$dataset: already trained ($(cat "$pointer_dir/CHECKPOINT"))"
    return 0
  fi
  mkdir -p "$pointer_dir"
  local marker="$pointer_dir/.before_training"
  touch "$marker"
  echo "===== TRAIN DINOv3 + AUGMENTATION ON $dataset ====="
  python -u -m tools.train --config "$config" --device cuda 2>&1 | tee "$pointer_dir/train.log" || return 1
  local summary
  summary="$(find "outputs/p29_dinov3_aug_${dataset}" -name summary.json -newer "$marker" | sort | tail -n 1)"
  test -n "$summary" || { echo "$dataset: no summary.json" >&2; return 1; }
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

stage_train() {
  python -m pytest -q tests/test_p29_augmentation.py || return 1
  local failed=0
  for dataset in ${DATASETS:-dsifn whu levir}; do
    if train_one "$dataset"; then
      echo "########## $dataset: trained ##########"
    else
      echo "########## $dataset: FAILED ##########"
      failed=1
    fi
  done
  return "$failed"
}

stage_module() {
  for dataset in levir dsifn whu; do
    test -f "outputs/p29_aug_detectors/$dataset/CHECKPOINT" || { echo "$dataset is not trained yet" >&2; return 1; }
  done
  bash scripts/run_p27_plugin.sh dinov3aug all || return 1
  python -m tools.p27_summary --output reports/p27_localizer/summary.csv
}

case "$STAGE" in
  check) stage_check ;;
  train) stage_train ;;
  module) stage_module ;;
  all) stage_check && stage_train && stage_module ;;
  *) echo "unknown stage $STAGE (check, train, module, all)" >&2; exit 1 ;;
esac
status=$?
echo "P29 augmentation baseline: $STAGE finished with status $status"
exit "$status"
