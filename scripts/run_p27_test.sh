#!/usr/bin/env bash
set -euo pipefail

# P27 final test (configs/p27_test.yaml): the frozen localizer, detectors and
# the selectors that went through the validation run, on every test crop,
# once. TRAIN_ROOT defaults to the latest selector training run.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_selector.yaml}"
TEST_CONFIG="${TEST_CONFIG:-configs/p27_test.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$TEST_CONFIG" || {
  echo "$TEST_CONFIG is not marked preregistered" >&2
  exit 1
}
train_root="${TRAIN_ROOT:-$(cat reports/p27_localizer/selector/train/LATEST)}"
test -f "$train_root/bundles/train_summary.json"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/test/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_selector.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P27 TEST ($train_root) ====="
python -u -m tools.run_p27_selector test \
  --config "$CONFIG" \
  --test-config "$TEST_CONFIG" \
  --device cuda \
  --bundle-dir "$train_root/bundles" \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/test.log"

sha256sum "$CONFIG" "$TEST_CONFIG" "$train_root"/bundles/*.pth "$report_root"/run/* > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/test/LATEST
echo "P27 test complete: $report_root"
