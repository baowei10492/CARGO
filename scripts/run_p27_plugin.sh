#!/usr/bin/env bash
set -euo pipefail

# P27 plug-in rows (docs/P27_PLUGIN_AMENDMENT.md).
#   bash scripts/run_p27_plugin.sh <row> <check|f1|selector_train|selector_eval|test|all>
# row: bit, cdlamba, cdxlstm (LEVIR-CD + DSIFN-CD) or bit_whu, cdlamba_whu, cdxlstm_whu (WHU-CD,
# docs/P27_EXPERIMENTS_AMENDMENT.md), or dinov3aug (the three DINOv3 detectors retrained with
# misalignment augmentation, docs/P29_AUGMENTATION_BASELINE.md)
# check  loads the localizer and both detectors of the row and prints where they come from
# all    f1 -> selector_train -> calibration go rule -> selector_eval -> test

DETECTOR="${1:?row: bit, cdlamba, cdxlstm, <detector>_whu or dinov3aug}"
STAGE="${2:?stage: check, f1, selector_train, selector_eval, test or all}"
case "$DETECTOR" in
  bit|cdlamba|cdxlstm|bit_whu|cdlamba_whu|cdxlstm_whu|dinov3aug) ;;
  *) echo "unknown row $DETECTOR" >&2; exit 1 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

F1_CONFIG="configs/p27_plugins/${DETECTOR}_f1.yaml"
SELECTOR_CONFIG="configs/p27_plugins/${DETECTOR}_selector.yaml"
TEST_CONFIG="configs/p27_plugins/${DETECTOR}_test.yaml"
ROOT_DIR="reports/p27_localizer/plugins/$DETECTOR"
test -f docs/P27_PLUGIN_AMENDMENT.md
case "$DETECTOR" in  # only the rows built from frozen P21 cells read the P21 freeze manifest
  bit|cdlamba|cdxlstm) test -f reports/p21/VALIDATION_FROZEN.json ;;
esac
LOCALIZER_RUN="$(python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c['localizer_run'])" "$SELECTOR_CONFIG")"
LOCALIZER_FILE="$(python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c['localizer_checkpoint'])" "$SELECTOR_CONFIG")"
CHECKPOINT="$LOCALIZER_RUN/checkpoints/$LOCALIZER_FILE"
test -f "$CHECKPOINT"

new_report() {
  local root
  root="$ROOT_DIR/$1/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$root"
  printf '%s\n' "$root"
}

stage_check() {
  python -u -m tools.run_p27_f1 --config "$F1_CONFIG" --checkpoint "$CHECKPOINT" \
    --training-config "$LOCALIZER_RUN/config.yaml" --device cuda \
    --output-dir "$ROOT_DIR/check_unused" --check-only
}

stage_f1() {
  local root
  root="$(new_report f1)"
  echo "===== $DETECTOR F1 ($root) ====="
  python -m pytest -q tests/test_p27_f1.py tests/test_p27_plugins.py 2>&1 | tee "$root/pytest.log"
  python -u -m tools.run_p27_f1 --config "$F1_CONFIG" --checkpoint "$CHECKPOINT" \
    --training-config "$LOCALIZER_RUN/config.yaml" --device cuda --output-dir "$root/run" \
    2>&1 | tee "$root/f1.log"
  sha256sum "$F1_CONFIG" "$CHECKPOINT" "$root"/run/* > "$root/SHA256SUMS.txt"
  printf '%s\n' "$root" > "$ROOT_DIR/f1/LATEST"
}

stage_selector_train() {
  local root
  root="$(new_report selector_train)"
  echo "===== $DETECTOR SELECTOR TRAIN ($root) ====="
  python -m pytest -q tests/test_p27_selector.py 2>&1 | tee "$root/pytest.log"
  python -u -m tools.run_p27_selector features --config "$SELECTOR_CONFIG" --device cuda \
    --output-dir "$root/features" 2>&1 | tee "$root/features.log"
  python -u -m tools.run_p27_selector train --config "$SELECTOR_CONFIG" \
    --features-dir "$root/features" --output-dir "$root/bundles" 2>&1 | tee "$root/train.log"
  sha256sum "$SELECTOR_CONFIG" "$root"/features/* "$root"/bundles/* > "$root/SHA256SUMS.txt"
  printf '%s\n' "$root" > "$ROOT_DIR/selector_train/LATEST"
}

calibration_go() {
  python - "$(cat "$ROOT_DIR/selector_train/LATEST")/bundles/train_summary.json" <<'PY'
import json
import sys

datasets = json.load(open(sys.argv[1], encoding="utf-8"))["datasets"]
go = {name: entry["selector_gains_at_levels_2_3"] for name, entry in datasets.items()}
print(f"calibration go rule: {go}")
sys.exit(0 if all(go.values()) else 3)
PY
}

stage_selector_eval() {
  local train_root root
  train_root="$(cat "$ROOT_DIR/selector_train/LATEST")"
  root="$(new_report selector_eval)"
  echo "===== $DETECTOR SELECTOR ON VALIDATION ($root) ====="
  python -u -m tools.run_p27_selector evaluate --config "$SELECTOR_CONFIG" --device cuda \
    --bundle-dir "$train_root/bundles" --output-dir "$root/run" 2>&1 | tee "$root/evaluate.log"
  sha256sum "$SELECTOR_CONFIG" "$train_root"/bundles/*.pth "$root"/run/* > "$root/SHA256SUMS.txt"
  printf '%s\n' "$root" > "$ROOT_DIR/selector_eval/LATEST"
}

stage_test() {
  local train_root root
  train_root="$(cat "$ROOT_DIR/selector_train/LATEST")"
  root="$(new_report test)"
  echo "===== $DETECTOR TEST ($root) ====="
  python -u -m tools.run_p27_selector test --config "$SELECTOR_CONFIG" --test-config "$TEST_CONFIG" \
    --device cuda --bundle-dir "$train_root/bundles" --output-dir "$root/run" 2>&1 | tee "$root/test.log"
  sha256sum "$SELECTOR_CONFIG" "$TEST_CONFIG" "$train_root"/bundles/*.pth "$root"/run/* > "$root/SHA256SUMS.txt"
  printf '%s\n' "$root" > "$ROOT_DIR/test/LATEST"
}

case "$STAGE" in
  check) stage_check ;;
  f1) stage_f1 ;;
  selector_train) stage_selector_train ;;
  selector_eval) stage_selector_eval ;;
  test) stage_test ;;
  all)
    stage_f1
    stage_selector_train
    if calibration_go; then
      stage_selector_eval
      stage_test
    else
      echo "$DETECTOR: the selector shows no calibration gain at levels 2 and 3; validation and test not run"
    fi
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
echo "$DETECTOR $STAGE complete"
