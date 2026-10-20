#!/usr/bin/env bash
set -euo pipefail

# WHU scale/coverage probe. No training, no val consumption:
# train-raster region only, measured with the LEVIR P7C instrument
# (never trained on WHU). CPU build ~1-2 min, GPU measurement minutes.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
RAW="${RAW:-/root/autodl-tmp/rs_cd_work/data/_raw_whu/Building change detection dataset_add}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/rs_cd_work/data/WHU-CD-scale-probe}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
LEVIR_512_SUMMARY="$(cat reports/p8_matched_image_size_control/LATEST)/levir_matched_512/offset_consistency_summary.json"
WHU_NATIVE_SUMMARY="$(cat reports/p8_whu_localizer/LATEST)/offset_diagnostic/offset_consistency_summary.json"

for required in \
  "$P7_CHECKPOINT" \
  "$LEVIR_512_SUMMARY" \
  "$WHU_NATIVE_SUMMARY" \
  configs/p8_whu_eval_protocol.yaml; do
  test -e "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p8_scale_probe/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p8_scale_probe/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_P7_CHECKPOINT"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

python -u -m tools.build_whu_scale_probe \
  --raw-root "$RAW" \
  --output-root "$PROBE_ROOT" \
  --scales 512 1024 2048 3072 \
  --output-size 512 \
  --max-blocks 60 \
  2>&1 | tee "$report_root/build_probe.log"
cp "$PROBE_ROOT/probe_summary.json" "$report_root/probe_summary.json"

python -m tools.build_manifest \
  --dataset-root "$PROBE_ROOT" \
  --output manifests/datasets/whu_scale_probe.csv \
  --splits scale0512 scale1024 scale2048 scale3072 \
  2>&1 | tee "$report_root/build_manifest.log"

summary_args=()
for split in scale0512 scale1024 scale2048 scale3072; do
  python -u -m tools.diagnose_transfer_offset_consistency \
    --checkpoint "$P7_CHECKPOINT" \
    --protocol configs/p8_whu_eval_protocol.yaml \
    --dataset-root "$PROBE_ROOT" \
    --manifest manifests/datasets/whu_scale_probe.csv \
    --split "$split" \
    --device cuda \
    --output-dir "$report_root/$split" \
    2>&1 | tee "$report_root/$split.log"
  summary_args+=(
    --summary
    "$split=$report_root/$split/offset_consistency_summary.json"
  )
done

python -m tools.summarize_scale_probe \
  "${summary_args[@]}" \
  --probe-summary "$report_root/probe_summary.json" \
  --levir-reference "$LEVIR_512_SUMMARY" \
  --whu-native-reference "$WHU_NATIVE_SUMMARY" \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

echo "Scale probe complete: $report_root"
cat "$report_root/decision.json"
