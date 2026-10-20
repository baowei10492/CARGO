#!/usr/bin/env bash
set -uo pipefail

# P33 queue B (docs/P33_REVIEWER_EXPERIMENTS.md), in order of importance after queue A:
#   1. the registration module retrained without change-aware weights and without change supervision (configs/p33/),
#      each 50k steps as the paper's v2, the two trainings side by side; then always-registering rows (Table 7),
#      per-crop errors (Figure 4) and the selector recipe on DINOv3 and BIT
#   2. S2Looking, which the registration module never saw: the DINOv3 detector trained with the paper's recipe, then
#      the selector recipe and the test split (levels 0-3; level 0 = the original pairs with their own offsets)
#   3. DiffRegCD with the interpolation control (its own environment), after the trainings so the GPU is not shared
# Finished parts are skipped.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p reports/p33/logs

python -m pytest -q tests/test_p33_reviewer.py tests/test_p27_diagnosis.py tests/test_p30_localizer.py \
  tests/test_translation_scale_baselines.py || { echo "tests failed; stop"; exit 1; }

train_localizer() {
  local name="$1" config="$2" dir="outputs/p33_localizer_$1"
  if [ -f "$dir/LATEST" ]; then echo "localizer $name: trained ($(cat "$dir/LATEST"))"; return 0; fi
  local run="$dir/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$dir"
  echo "===== TRAIN LOCALIZER $name -> $run ($(date -u +%FT%TZ)) ====="
  python -u -m tools.train_similarity_localizer --config "$config" --output-dir "$run" \
    > "$dir/train_$(basename "$run").log" 2>&1 || { echo "localizer $name: training failed"; return 1; }
  test -f "$run/checkpoints/best.pth" || { echo "localizer $name: no best.pth"; return 1; }
  sha256sum "$config" "$run/checkpoints/best.pth" > "$run/SHA256SUMS.txt"
  printf '%s\n' "$run" > "$dir/LATEST"
  echo "localizer $name: $(cat "$run/train_summary.json")"
}

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

status=0
# 1. retrained registration modules, trained side by side
train_localizer conf_only configs/p33/localizer_conf_only.yaml > reports/p33/logs/train_conf_only.out 2>&1 &
first=$!
sleep 60
train_localizer no_change_label configs/p33/localizer_no_change_label.yaml > reports/p33/logs/train_no_change_label.out 2>&1 &
second=$!
wait "$first"; s1=$?
wait "$second"; s2=$?
cat reports/p33/logs/train_conf_only.out reports/p33/logs/train_no_change_label.out
if [ "$s1" -eq 0 ] && [ "$s2" -eq 0 ]; then
  ABLATION="$(cat reports/p31/localizer_ablation/LATEST)"
  echo "===== ALWAYS-REGISTERING ROWS AND ERRORS ($(date -u +%FT%TZ)) ====="
  python -u -m tools.p31_localizer_ablation --output-dir "$ABLATION" --variants v2r_conf_only v2r_no_change_label || status=1
  python -u -m tools.p31_registration_errors --output-dir reports/p31/overnight/run/errors \
    --variants v2r_conf_only v2r_no_change_label || status=1
  for variant in v2r_conf_only v2r_no_change_label; do
    echo "===== SELECTOR ON $variant ($(date -u +%FT%TZ)) ====="
    python -u -m tools.p33_selector_runs pipeline --localizer "$variant" --rows dinov3 bit || status=1
  done
else
  status=1
fi

# 2. S2Looking
if train_s2looking_detector; then
  echo "===== S2Looking SELECTOR AND TEST ($(date -u +%FT%TZ)) ====="
  python -u -m tools.p33_selector_runs pipeline --localizer v2 --s2looking || status=1
else
  status=1
fi

# 3. DiffRegCD interpolation control (DiffRegCD environment)
DIFFREGCD_CONFIG="$(python -c "import yaml; print(yaml.safe_load(open('configs/p27_diffregcd_rerun.yaml'))['diffregcd_config'])")"
EXTERNAL_ROOT="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['external']['root'])" "$DIFFREGCD_CONFIG")"
echo "===== DIFFREGCD INTERPOLATION CONTROL ($(date -u +%FT%TZ)) ====="
PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$EXTERNAL_ROOT" WANDB_MODE=disabled \
  "$EXTERNAL_ROOT/.venv/bin/python" -u -m tools.p33_diffregcd_interp || status=1

echo "P33 queue B finished with status $status ($(date -u +%FT%TZ))"
exit "$status"
