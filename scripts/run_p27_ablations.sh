#!/usr/bin/env bash
set -uo pipefail

# P27 ablations (docs/P27_EXPERIMENTS_AMENDMENT.md), in the project environment:
#   1. selector ablations (CPU) for every row whose test run exists
#   2. localizer ablation and matching baselines, DINOv3 detectors, test split (GPU, resumable)
#   3. efficiency (GPU, a few minutes)
#   4. qualitative panels of the DINOv3 row
# STEPS overrides the list, e.g. STEPS="selector efficiency". STEPS=figure renders the
# tiles of the paper figure (assembled locally with tools/p27_figure_qualitative.py).

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ROOT="reports/p27_localizer/ablations"
mkdir -p "$ROOT"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"

row_paths() {  # row -> "config train_pointer test_pointer"
  if [ "$1" = dinov3 ]; then
    echo "configs/p27_selector.yaml reports/p27_localizer/selector/train/LATEST reports/p27_localizer/test/LATEST"
  else
    echo "configs/p27_plugins/$1_selector.yaml reports/p27_localizer/plugins/$1/selector_train/LATEST reports/p27_localizer/plugins/$1/test/LATEST"
  fi
}

step_selector() {
  for row in dinov3 bit cdlamba cdxlstm bit_whu cdlamba_whu cdxlstm_whu; do
    read -r config train_pointer test_pointer <<< "$(row_paths "$row")"
    if [ ! -f "$train_pointer" ] || [ ! -f "$test_pointer" ]; then
      echo "selector ablation: $row has no finished test yet, skipped"
      continue
    fi
    local out="$ROOT/selector/$row/$TAG"
    mkdir -p "$(dirname "$out")"
    echo "===== SELECTOR ABLATION $row ====="
    python -u -m tools.p27_selector_ablation --config "$config" --train-root "$(cat "$train_pointer")" \
      --records "$(cat "$test_pointer")/run/test_records.csv" --output-dir "$out" 2>&1 | tee "$ROOT/selector_${row}_$TAG.log"
  done
}

step_localizer() {
  echo "===== LOCALIZER ABLATION (test split, resumable) ====="
  python -u -m tools.p27_localizer_ablation --split test --output-dir "$ROOT/localizer/test" \
    2>&1 | tee -a "$ROOT/localizer_test.log"
}

step_efficiency() {
  echo "===== EFFICIENCY ====="
  python -u -m tools.p27_efficiency --roma --output "$ROOT/efficiency/$TAG.json" 2>&1 | tee "$ROOT/efficiency_$TAG.log"
}

step_qualitative() {
  echo "===== QUALITATIVE PANELS (DINOv3 row) ====="
  python -u -m tools.p27_qualitative --test-root "$(cat reports/p27_localizer/test/LATEST)" \
    --train-root "$(cat reports/p27_localizer/selector/train/LATEST)" --output-dir "$ROOT/qualitative/$TAG" \
    2>&1 | tee "$ROOT/qualitative_$TAG.log"
}

step_figure() {  # the crops of the paper figure (configs/p27_qualitative_figure.json), as tiles
  echo "===== QUALITATIVE FIGURE TILES (DINOv3 row) ====="
  python -u -m tools.p27_qualitative --test-root "$(cat reports/p27_localizer/test/LATEST)" \
    --train-root "$(cat reports/p27_localizer/selector/train/LATEST)" --select configs/p27_qualitative_figure.json \
    --output-dir "$ROOT/figure/$TAG" 2>&1 | tee "$ROOT/figure_$TAG.log"
}

for step in ${STEPS:-selector localizer efficiency qualitative}; do
  if ! "step_$step"; then
    echo "########## $step: FAILED ##########"
  fi
done
echo "P27 ablations done ($TAG)"
