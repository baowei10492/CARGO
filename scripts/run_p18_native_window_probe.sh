#!/usr/bin/env bash
set -euo pipefail

# P18: separate ground coverage from working GSD in the WHU localizer
# failure. The original probe (main record 77) downsampled every block to
# 512 px, so coverage and effective GSD moved together. Here the block is
# kept at its native 0.075 m resolution and only the window grows.
#
# No training, no validation or test consumption: train-raster blocks
# only, measured with the frozen LEVIR localizer that never saw WHU.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
RAW="${RAW:-/root/autodl-tmp/rs_cd_work/data/_raw_whu/Building change detection dataset_add}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/rs_cd_work/data/WHU-CD-native-window-probe}"
SCALES="${SCALES:-512 1024 2048 3072}"
MAX_BLOCKS="${MAX_BLOCKS:-60}"
PROTOCOL="${PROTOCOL:-configs/p18_native_window_probe.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
ORIGINAL_PROBE="${ORIGINAL_PROBE:-$(cat reports/p8_scale_probe/LATEST)}"

for required in "$P7_CHECKPOINT" "$PROTOCOL" "$ORIGINAL_PROBE"; do
  test -e "$required" || {
    echo "Missing required input: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p18_native_window_probe/$tag"
mkdir -p "$report_root"
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_P7_CHECKPOINT"
printf '%s\n' "$ORIGINAL_PROBE" > "$report_root/ORIGINAL_PROBE"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_scale_probe.py tests/test_native_window_probe.py

native_args=()
downsampled_args=()

for scale in $SCALES; do
  split="$(printf 'scale%04d' "$scale")"
  root="${PROBE_ROOT}/native_${scale}"
  echo "===== BUILD NATIVE BLOCKS: window ${scale} px ====="
  # output-size equal to the window means no downsampling, so the block
  # keeps the native 0.075 m GSD and only its coverage changes.
  python -u -m tools.build_whu_scale_probe \
    --raw-root "$RAW" \
    --output-root "$root" \
    --scales "$scale" \
    --output-size "$scale" \
    --max-blocks "$MAX_BLOCKS" \
    2>&1 | tee "$report_root/build_${split}.log"

  python -m tools.build_manifest \
    --dataset-root "$root" \
    --output "manifests/datasets/whu_native_window_${scale}.csv" \
    --splits "$split" \
    2>&1 | tail -3

  echo "===== MEASURE: window ${scale} px at native GSD ====="
  python -u -m tools.diagnose_transfer_offset_consistency \
    --checkpoint "$P7_CHECKPOINT" \
    --protocol "$PROTOCOL" \
    --dataset-root "$root" \
    --manifest "manifests/datasets/whu_native_window_${scale}.csv" \
    --split "$split" \
    --device cuda \
    --output-dir "$report_root/$split" \
    2>&1 | tee "$report_root/$split.log"

  native_args+=(--native "$scale=$report_root/$split/offset_consistency_summary.json")
  downsampled_args+=(--downsampled "$scale=$ORIGINAL_PROBE/$split/offset_consistency_summary.json")
done

echo "===== VERDICT: COVERAGE VERSUS GSD ====="
python -m tools.summarize_native_window_probe \
  "${native_args[@]}" \
  "${downsampled_args[@]}" \
  --scenario shift:s2:t2 \
  --baseline-scale 512 \
  --output-json "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

printf '%s\n' "$report_root" > reports/p18_native_window_probe/LATEST
sha256sum "$report_root"/*.json "$report_root"/*/offset_consistency_summary.json \
  > "$report_root/SHA256SUMS.txt"
echo "P18 native-window probe complete: $report_root"
