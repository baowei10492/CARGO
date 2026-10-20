#!/usr/bin/env bash
set -euo pipefail

# Attribution control for the WHU localizer stage-A failure.
# Varies ONE thing on the reference dataset: evaluation image size.
# No training. Validation splits only. Minutes, not hours.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
WHU_REPORT="${WHU_REPORT:-$(cat reports/p8_whu_localizer/LATEST)}"
WHU_OFFSETS="$WHU_REPORT/offset_diagnostic/offset_consistency_summary.json"

for required in \
  "$P7_CHECKPOINT" \
  "$WHU_OFFSETS" \
  manifests/datasets/levir_cd.csv \
  configs/p8_matched_control_eval_protocol.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p8_matched_image_size_control/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p8_matched_image_size_control/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_P7_CHECKPOINT"
printf '%s\n' "$WHU_REPORT" > "$report_root/SOURCE_WHU_REPORT"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

# ---------------------------------------------------------------- part 1
# Free evidence already on disk: in-sample localizability. All three
# datasets train the localizer on 256 px crops, so these curves are NOT
# confounded by evaluation image size.
{
  echo "===== in-sample localizer curves (train crop 256 for all) ====="
  for name in \
    outputs/p7b_decoupled_input_registration_stage_a \
    outputs/c2b_s2looking_decoupled_stage_a \
    outputs/p8_whu_decoupled_stage_a; do
    log="$(
      find "$name" -name train.log -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | head -n 1 | cut -d' ' -f2-
    )"
    if [[ -z "$log" ]]; then
      log="$(
        find reports -path "*${name##*/}*" -name train.log \
          -printf '%T@ %p\n' 2>/dev/null |
        sort -nr | head -n 1 | cut -d' ' -f2-
      )"
    fi
    echo "--- $name"
    if [[ -n "$log" && -f "$log" ]]; then
      grep -oE \
        'epoch=[0-9]+ .*train_offset_epe=[0-9.]+ train_clean_pck1=[0-9.]+ train_shift_pck1=[0-9.]+' \
        "$log" | awk 'NR==1 || NR%3==0 || 1' | tail -n 16
    else
      echo "    train.log not found under $name (search by hand)"
    fi
  done
} 2>&1 | tee "$report_root/insample_curves.txt"

# ---------------------------------------------------------------- part 2
# LEVIR at its native 1024, then LEVIR centre-cropped to WHU's 512.
# Same checkpoint, same tool, same protocol: image size is the only
# variable.
python -u -m tools.diagnose_transfer_offset_consistency \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p8_matched_control_eval_protocol.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --device cuda \
  --output-dir "$report_root/levir_native_1024" \
  2>&1 | tee "$report_root/levir_native_1024.log"

python -u -m tools.diagnose_transfer_offset_consistency \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p8_matched_control_eval_protocol.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --image-size 512 \
  --device cuda \
  --output-dir "$report_root/levir_matched_512" \
  2>&1 | tee "$report_root/levir_matched_512.log"

python -m tools.summarize_matched_image_size_control \
  --reference-native \
    "$report_root/levir_native_1024/offset_consistency_summary.json" \
  --reference-matched \
    "$report_root/levir_matched_512/offset_consistency_summary.json" \
  --target "$WHU_OFFSETS" \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

echo "Matched image-size control complete: $report_root"
cat "$report_root/decision.json"
