#!/usr/bin/env bash
set -uo pipefail

# P30 training (docs/P30_BORDER_FREE_PROTOCOL.md):
#   localizer  the P27 localizer, version 2, on border-free targets (configs/p30_localizer_v2.yaml);
#              pointer outputs/p30_localizer/LATEST
#   aug        the DINOv3 detectors retrained with border-free augmentation (configs/p30/dinov3_aug_*.yaml);
#              pointers outputs/p30_aug_detectors/<dataset>/CHECKPOINT
#   all        localizer -> aug (default). Finished parts are skipped.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8
STAGE="${1:-all}"

stage_localizer() {
  if [ -f outputs/p30_localizer/LATEST ]; then
    echo "localizer v2: already trained ($(cat outputs/p30_localizer/LATEST))"
    return 0
  fi
  python -m pytest -q tests/test_p30_localizer.py tests/test_p27_training.py || return 1
  local run="outputs/p30_localizer/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p outputs/p30_localizer
  echo "===== TRAIN LOCALIZER V2 (border-free) -> $run ====="
  python -u -m tools.train_similarity_localizer --config configs/p30_localizer_v2.yaml --output-dir "$run" \
    2>&1 | tee "outputs/p30_localizer/train_$(basename "$run").log" || return 1
  test -f "$run/checkpoints/best.pth" || { echo "no best.pth in $run" >&2; return 1; }
  sha256sum configs/p30_localizer_v2.yaml "$run/checkpoints/best.pth" > "$run/SHA256SUMS.txt"
  printf '%s\n' "$run" > outputs/p30_localizer/LATEST
  echo "localizer v2: $(cat "$run/train_summary.json")"
}

train_aug() {
  local dataset="$1"
  local config="configs/p30/dinov3_aug_${dataset}.yaml"
  local pointer_dir="outputs/p30_aug_detectors/$dataset"
  if [ -f "$pointer_dir/CHECKPOINT" ]; then
    echo "$dataset: already trained ($(cat "$pointer_dir/CHECKPOINT"))"
    return 0
  fi
  mkdir -p "$pointer_dir"
  local marker="$pointer_dir/.before_training"
  touch "$marker"
  echo "===== TRAIN DINOv3 + BORDER-FREE AUGMENTATION ON $dataset ====="
  python -u -m tools.train --config "$config" --device cuda 2>&1 | tee "$pointer_dir/train.log" || return 1
  local summary
  summary="$(find "outputs/p30_dinov3_aug_${dataset}" -name summary.json -newer "$marker" | sort | tail -n 1)"
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
relative = checkpoint.resolve().relative_to(project_root)
(pointer_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(pointer_dir / "CHECKPOINT").write_text(f"{relative.as_posix()}\n", encoding="utf-8")
print(f"best validation F1 {float(best):.4f} at epoch {summary.get('best_epoch')}; pointer -> {relative}")
PY
  sha256sum "$config" "$(cat "$pointer_dir/CHECKPOINT")" > "$pointer_dir/SHA256SUMS.txt"
}

stage_aug() {
  python -m pytest -q tests/test_p30_augmentation.py tests/test_p29_augmentation.py || return 1
  local failed=0
  for dataset in ${DATASETS:-dsifn whu levir}; do
    if train_aug "$dataset"; then echo "########## $dataset: trained ##########"; else echo "########## $dataset: FAILED ##########"; failed=1; fi
  done
  return "$failed"
}

case "$STAGE" in
  localizer) stage_localizer ;;
  aug) stage_aug ;;
  all) stage_localizer; stage_aug ;;
  *) echo "unknown stage $STAGE (localizer, aug, all)" >&2; exit 1 ;;
esac
status=$?
echo "P30 training: $STAGE finished with status $status"
exit "$status"
