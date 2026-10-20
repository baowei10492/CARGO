#!/usr/bin/env bash
set -euo pipefail

# P21 amendment D (docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md section 12): the
# one-shot LEVIR-CD test of BIT, CD-Lamba and CDxLSTM. Reporting only.
#
#   python -m tools.freeze_p21_validation   # once, after every cell is recorded
#   scripts/run_p21_final_test.sh
#
# Refuses to start unless amendment D is in the protocol, the amendment C
# inference code is installed, the validation matrix is frozen and no
# detector's LEVIR-CD test marker exists. Each detector writes its marker
# before reading its first test image. There are no reruns.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
PROTOCOL_DOC=docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md
FREEZE_MANIFEST=reports/p21/VALIDATION_FROZEN.json
TEST_ROOT=reports/p21_final_test

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "===== 0. PRECONDITIONS ====="
grep -q "修订 D" "$PROTOCOL_DOC" || {
  echo "Amendment D is missing from $PROTOCOL_DOC" >&2
  exit 1
}
# CDxLSTM must infer on 256-pixel tiles exactly as in its validation cells.
grep -q "inference_tile_size" src/rs_cd/runtime.py || {
  echo "Amendment C code is not installed (src/rs_cd/runtime.py)" >&2
  exit 1
}
test -f "$FREEZE_MANIFEST" || {
  echo "$FREEZE_MANIFEST is missing: run python -m tools.freeze_p21_validation first" >&2
  exit 1
}
test -d "$LEVIR_ROOT" || {
  echo "LEVIR-CD not found at $LEVIR_ROOT" >&2
  exit 1
}
for slug in bit cdlamba cdxlstm; do
  marker="$TEST_ROOT/${slug}_LEVIR_TEST_CONSUMED.json"
  if [[ -e "$marker" ]]; then
    echo "The LEVIR-CD test was already started for $slug:" >&2
    cat "$marker" >&2
    exit 1
  fi
done

python -m pytest -q \
  tests/test_p21_final_test.py \
  tests/test_external_rschange.py \
  tests/test_dual_path_adapter.py

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="$TEST_ROOT/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > "$TEST_ROOT/LATEST"
sha256sum "$FREEZE_MANIFEST" | tee "$report_root/FREEZE_MANIFEST_SHA256"

for pair in "BIT bit" "CD-Lamba cdlamba" "CDxLSTM cdxlstm"; do
  read -r detector slug <<< "$pair"
  echo "===== LEVIR-CD TEST: $detector ====="
  python -u -m tools.evaluate_p21_final_test \
    --detector "$detector" \
    --freeze-manifest "$FREEZE_MANIFEST" \
    --protocol configs/p21_levir_final_test.yaml \
    --dataset-root "$LEVIR_ROOT" \
    --manifest manifests/datasets/levir_cd.csv \
    --device cuda \
    --consumed-marker "$TEST_ROOT/${slug}_LEVIR_TEST_CONSUMED.json" \
    --output-dir "$report_root/$slug" \
    2>&1 | tee "$report_root/$slug.log"
done

echo "===== SUMMARY ====="
python - "$report_root" <<'PY' | tee "$report_root/summary.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print(f"{'detector':<10}{'method':<16}{'clean':>8}{'2px':>8}{'4px':>8}{'8px':>8}{'mean':>8}")
for slug in ("bit", "cdlamba", "cdxlstm"):
    summary = json.loads((root / slug / "final_test_summary.json").read_text(encoding="utf-8"))
    for method, block in summary["methods"].items():
        shift = block["shift_f1"]
        print(
            f"{summary['detector']:<10}{method:<16}{block['clean_f1']:>8.4f}"
            f"{shift['2px']:>8.4f}{shift['4px']:>8.4f}{shift['8px']:>8.4f}"
            f"{block['shift_mean_f1']:>8.4f}"
        )
    capture = summary["capture_descriptive"]["mean_capture_rate"]
    break_even = summary["break_even_clean_fraction"]
    capture_text = "n/a" if capture is None else f"{capture:.3f}"
    break_even_text = "n/a" if break_even is None else f"{break_even:.3f}"
    print(f"{'':<10}capture {capture_text}  break-even p {break_even_text}")
PY

sha256sum \
  "$report_root"/*/final_test_summary.json \
  "$report_root"/*/results.json \
  "$TEST_ROOT"/*_LEVIR_TEST_CONSUMED.json \
  > "$report_root/SHA256SUMS.txt"
echo "P21 LEVIR-CD test complete: $report_root"
