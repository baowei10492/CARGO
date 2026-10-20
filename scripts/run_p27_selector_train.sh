#!/usr/bin/env bash
set -euo pipefail

# P27 selector step, part 1 (docs/P27_SELECTOR_AMENDMENT.md): features on the
# training splits, one classifier per dataset, threshold on the localizer's
# hold-out tiles. Reads no validation or test data.

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
test -f docs/P27_SELECTOR_AMENDMENT.md

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/selector/train/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_selector.py tests/test_p27_f1.py 2>&1 | tee "$report_root/pytest.log"

echo "===== FEATURES (training splits) ====="
python -u -m tools.run_p27_selector features \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/features" \
  2>&1 | tee "$report_root/features.log"

echo "===== TRAIN AND CALIBRATE ====="
python -u -m tools.run_p27_selector train \
  --config "$CONFIG" \
  --features-dir "$report_root/features" \
  --output-dir "$report_root/bundles" \
  2>&1 | tee "$report_root/train.log"

sha256sum "$CONFIG" "$report_root"/features/* "$report_root"/bundles/* > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/selector/train/LATEST
echo "P27 selector training complete: $report_root"
