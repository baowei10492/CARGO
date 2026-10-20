#!/usr/bin/env bash
set -euo pipefail

# P30 evaluation (docs/P30_BORDER_FREE_PROTOCOL.md), after scripts/run_p30_training.sh:
#   selector  features on border-free training crops, P30 selectors (reads no validation or test data);
#             pointer reports/p30/selector/LATEST
#   val       rows 1-5 on the validation crops, once per set of components; pointer reports/p30/val/LATEST
#   test      rows 1-5 on the test crops, once, after val
#   all       selector -> val (default). The test is its own stage, run after reading the validation table.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p30_border_free.yaml}"
STAGE="${1:-all}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || { echo "$CONFIG is not marked preregistered" >&2; exit 1; }
test -f docs/P30_BORDER_FREE_PROTOCOL.md
test -f outputs/p30_localizer/LATEST || { echo "localizer v2 is not trained yet (outputs/p30_localizer/LATEST)" >&2; exit 1; }

require_detectors() {
  local missing=0
  for dataset in levir dsifn whu; do
    for pointer in "outputs/p29_aug_detectors/$dataset/CHECKPOINT" "outputs/p30_aug_detectors/$dataset/CHECKPOINT"; do
      if [ ! -f "$pointer" ] || [ ! -f "$(cat "$pointer")" ]; then echo "missing: $pointer" >&2; missing=1; fi
    done
  done
  test -f reports/p27_localizer/selector/train/LATEST || { echo "missing: P27 selector bundles" >&2; missing=1; }
  return "$missing"
}

stage_selector() {
  local root="reports/p30/selector/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$root"
  echo "===== TARGETED TESTS ====="
  python -m pytest -q tests/test_p30_evaluation.py tests/test_p30_margin_crops.py tests/test_p27_selector.py \
    2>&1 | tee "$root/pytest.log"
  echo "===== FEATURES (border-free training crops) ====="
  python -u -m tools.run_p30 features --config "$CONFIG" --device cuda --output-dir "$root/features" \
    2>&1 | tee "$root/features.log"
  echo "===== TRAIN AND CALIBRATE THE P30 SELECTORS ====="
  python -u -m tools.run_p30 train --config "$CONFIG" --features-dir "$root/features" --output-dir "$root/bundles" \
    2>&1 | tee "$root/train.log"
  sha256sum "$CONFIG" "$root"/features/* "$root"/bundles/* > "$root/SHA256SUMS.txt"
  printf '%s\n' "$root" > reports/p30/selector/LATEST
  echo "P30 selectors: $root"
}

stage_evaluate() {
  local split="$1"
  require_detectors
  test -f reports/p30/selector/LATEST || { echo "the P30 selectors are not trained yet" >&2; exit 1; }
  local bundles="$(cat reports/p30/selector/LATEST)/bundles"
  local root="reports/p30/$split/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "reports/p30/$split"
  echo "===== P30 ROWS 1-5 ON $split ====="
  python -u -m tools.run_p30 evaluate --config "$CONFIG" --bundle-dir "$bundles" --split "$split" --device cuda \
    --output-dir "$root" 2>&1 | tee "reports/p30/$split/$(basename "$root").log"
  sha256sum "$root"/* > "$root/SHA256SUMS.txt"
  printf '%s\n' "$root" > "reports/p30/$split/LATEST"
  echo "P30 $split: $root"
}

case "$STAGE" in
  selector) stage_selector ;;
  val) stage_evaluate val ;;
  test) stage_evaluate test ;;
  all) require_detectors; stage_selector; stage_evaluate val ;;
  *) echo "unknown stage $STAGE (selector, val, test, all)" >&2; exit 1 ;;
esac
