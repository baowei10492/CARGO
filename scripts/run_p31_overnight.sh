#!/usr/bin/env bash
set -uo pipefail

# Overnight session (main record 192.71), about 4 hours on one RTX 4090:
#   seeds          selector retrained with 5 classifier seeds, 12 cells (CPU, in the background; needs the selector
#                  training features, i.e. the machine that ran the selector ablation)
#   input_swap     the four frozen detectors with the earlier image replaced (same date / unrelated / black)
#   dose           dose-response on self-pairs with pasted change: localizer v2, v2 without reliability, RoMa
#   errors         per-crop registration errors on the border-free test crops, localizer variants, SIFT, RoMa
# Finished parts are kept on a rerun (same output directory). Everything is packed into
# ../results_overnight.tgz for download.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

root="${OVERNIGHT_ROOT:-reports/p31/overnight/run}"
mkdir -p "$root"

echo "===== PREFLIGHT ====="
missing=0
for file in reports/p30/test/LATEST reports/p30/selector/LATEST reports/p31/localizer_ablation/LATEST \
            outputs/p30_localizer/LATEST reports/p31/bit/test/p31_test.json reports/p31/cdlamba/test/p31_test.json \
            reports/p31/cdxlstm/test/p31_test.json; do
  test -f "$file" || { echo "missing: $file"; missing=1; }
done
test "$missing" -eq 0 || { echo "preflight failed: this machine lacks P30/P31 results" >&2; exit 1; }
seeds_ready=1
p30_selector="$(cat reports/p30/selector/LATEST)"
for name in LEVIR-CD DSIFN-CD WHU-CD; do
  for dir in "$p30_selector" reports/p31/bit reports/p31/cdlamba reports/p31/cdxlstm; do
    test -f "$dir/features/features_$name.csv" || { echo "no selector training features: $dir/features/features_$name.csv"; seeds_ready=0; }
  done
done
test "$seeds_ready" -eq 1 && echo "selector training features: all 12 present" || echo "WARNING: the seeds part is skipped on this machine"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31_overnight.py tests/test_p30_evaluation.py 2>&1 | tee "$root/pytest.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo "tests failed" >&2; exit 1; }

failed=0
seeds_pid=""
if [ "$seeds_ready" -eq 1 ] && [ ! -f "$root/selector_seeds.json" ]; then
  echo "===== SELECTOR SEEDS (background, CPU) ====="
  python -u -m tools.p31_selector_seeds --output "$root/selector_seeds.json" > "$root/selector_seeds.log" 2>&1 &
  seeds_pid=$!
fi

if [ ! -f "$root/input_swap.json" ]; then
  echo "===== INPUT SWAP ====="
  python -u -m tools.p31_input_swap --output "$root/input_swap.json" 2>&1 | tee "$root/input_swap.log"
  test "${PIPESTATUS[0]}" -eq 0 || failed=1
fi

echo "===== DOSE-RESPONSE ====="
python -u -m tools.p31_dose_response --output-dir "$root/dose" 2>&1 | tee -a "$root/dose.log"
test "${PIPESTATUS[0]}" -eq 0 || failed=1

echo "===== PER-CROP REGISTRATION ERRORS ====="
python -u -m tools.p31_registration_errors --output-dir "$root/errors" 2>&1 | tee -a "$root/errors.log"
test "${PIPESTATUS[0]}" -eq 0 || failed=1

if [ -n "$seeds_pid" ]; then
  echo "===== WAITING FOR SELECTOR SEEDS ====="
  wait "$seeds_pid" || failed=1
  tail -n 30 "$root/selector_seeds.log"
fi

tar czf ../results_overnight.tgz "$root"
ls -lh ../results_overnight.tgz
echo "P31 overnight session finished with status $failed: $root"
exit "$failed"
