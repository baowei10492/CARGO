#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
WHU_ROOT="${WHU_ROOT:-/root/autodl-tmp/rs_cd_work/data/WHU-CD}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

# Warm start: LEVIR P7C registration stack + WHU P3b detector.
if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
if [[ -z "${DETECTOR_CHECKPOINT:-}" ]]; then
  p3b_report="$(cat reports/p8_whu_p3b/LATEST)"
  DETECTOR_CHECKPOINT="$(cat "$p3b_report/CHECKPOINT")"
fi

for required in \
  "$P7_CHECKPOINT" \
  "$DETECTOR_CHECKPOINT" \
  manifests/datasets/whu_cd.csv \
  configs/p8_whu_decoupled_stage_a.yaml \
  configs/p8_whu_eval_protocol.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p8_whu_localizer/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p8_whu_localizer/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_P7_CHECKPOINT"
printf '%s\n' "$DETECTOR_CHECKPOINT" \
  > "$report_root/SOURCE_DETECTOR_CHECKPOINT"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

# search_radius is unchanged (4), so no radius-dependent head is dropped.
python -m tools.build_c2b_init_checkpoint \
  --p7-checkpoint "$P7_CHECKPOINT" \
  --detector-checkpoint "$DETECTOR_CHECKPOINT" \
  --output "$report_root/p8_init.pth" \
  2>&1 | tee "$report_root/init_checkpoint.log"

python -m tools.gpu_smoke \
  --config configs/p8_whu_decoupled_stage_a.yaml \
  --batch-size 64 \
  --image-size 256 \
  --steps 5 \
  --output "$report_root/gpu_smoke_bs64.json" \
  2>&1 | tee "$report_root/gpu_smoke_bs64.log"

python -u -m tools.train \
  --config configs/p8_whu_decoupled_stage_a.yaml \
  --dataset-root "$WHU_ROOT" \
  --initial-checkpoint "$report_root/p8_init.pth" \
  --device cuda \
  2>&1 | tee "$report_root/train.log"

summary="$(
  find outputs/p8_whu_decoupled_stage_a \
    -name summary.json \
    -printf '%T@ %p\n' |
  sort -nr |
  head -n 1 |
  cut -d' ' -f2-
)"
test -n "$summary"
run_dir="$(dirname "$summary")"
checkpoint="$run_dir/checkpoints/last.pth"
test -f "$checkpoint"
printf '%s\n' "$run_dir" > "$report_root/RUN_DIR"
printf '%s\n' "$checkpoint" > "$report_root/CHECKPOINT"
cp "$summary" "$report_root/train_summary.json"

python -m tools.evaluate_registration \
  --checkpoint "$checkpoint" \
  --dataset-root "$WHU_ROOT" \
  --split val \
  --device cuda \
  --batch-size 8 \
  --severities 0 1 2 3 \
  --output "$report_root/registration_val.json" \
  2>&1 | tee "$report_root/registration_val.log"

python -u -m tools.diagnose_transfer_offset_consistency \
  --checkpoint "$checkpoint" \
  --protocol configs/p8_whu_eval_protocol.yaml \
  --dataset-root "$WHU_ROOT" \
  --manifest manifests/datasets/whu_cd.csv \
  --device cuda \
  --output-dir "$report_root/offset_diagnostic" \
  2>&1 | tee "$report_root/offset_diagnostic.log"

# Same gate code and same numeric thresholds as the C2b S2Looking stage,
# so the two datasets are adjudicated by one instrument. The clean q
# reference 0.833 is the LEVIR-localizer-on-WHU screening value (main
# record 70.7); passing it in resolves the preregistered screening
# ambiguity in the decision file itself.
python -m tools.summarize_c2b_stage_a \
  --registration-results "$report_root/registration_val.json" \
  --offset-summary \
    "$report_root/offset_diagnostic/offset_consistency_summary.json" \
  --transfer-clean-q-reference 0.833 \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

# Descriptive only (no gate): tile dispersion with the WHU-NATIVE
# localizer. The 1024/512 thresholds are NOT reused - they were shown
# non-transferable across measurement configurations (main record 70.7).
python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "$checkpoint" \
  --dataset-root "$WHU_ROOT" \
  --manifest manifests/datasets/whu_cd.csv \
  --split val \
  --label whu_native_localizer \
  --tile-size 256 \
  --center-crop 512 \
  --severities 1 2 3 \
  --batch-size 4 \
  --device cuda \
  --output-dir "$report_root/hygiene_native" \
  2>&1 | tee "$report_root/hygiene_native.log"

echo "P8 WHU localizer stage-A complete: $report_root"
cat "$report_root/decision.json"
