#!/usr/bin/env bash
set -euo pipefail

# P27 selector step, part 2 (docs/P27_SELECTOR_AMENDMENT.md): the trained
# selectors on the validation crops, once. Run only after the calibration
# table of part 1 has been read.
# TRAIN_ROOT defaults to the latest part-1 run.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_selector.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
train_root="${TRAIN_ROOT:-$(cat reports/p27_localizer/selector/train/LATEST)}"
test -f "$train_root/bundles/train_summary.json"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/selector/eval/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_selector.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P27 SELECTOR ON VALIDATION ($train_root) ====="
python -u -m tools.run_p27_selector evaluate \
  --config "$CONFIG" \
  --device cuda \
  --bundle-dir "$train_root/bundles" \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/evaluate.log"

sha256sum "$CONFIG" "$train_root"/bundles/*.pth "$report_root"/run/* > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/selector/eval/LATEST
echo "P27 selector evaluation complete: $report_root"
