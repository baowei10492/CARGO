#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
S2LOOKING_ROOT="${S2LOOKING_ROOT:-/root/autodl-tmp/rs_cd_work/data/S2Looking}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-outputs/p6_s2looking_dinov3_partial_ft/20260726T014320Z_seed3407/checkpoints/best.pth}"

for required in \
  "$P7_CHECKPOINT" \
  "$DETECTOR_CHECKPOINT" \
  manifests/datasets/s2looking.csv \
  configs/c2b_s2looking_decoupled_stage_a.yaml \
  configs/p7_stage_c_s2looking_transfer.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/c2b_stage_a_s2looking/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/c2b_stage_a_s2looking/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_P7_CHECKPOINT"
printf '%s\n' "$DETECTOR_CHECKPOINT" \
  > "$report_root/SOURCE_DETECTOR_CHECKPOINT"

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

python -m tools.build_c2b_init_checkpoint \
  --p7-checkpoint "$P7_CHECKPOINT" \
  --detector-checkpoint "$DETECTOR_CHECKPOINT" \
  --output "$report_root/c2b_init.pth" \
  2>&1 | tee "$report_root/init_checkpoint.log"

python -m tools.gpu_smoke \
  --config configs/c2b_s2looking_decoupled_stage_a.yaml \
  --batch-size 64 \
  --image-size 256 \
  --steps 5 \
  --output "$report_root/gpu_smoke_bs64.json" \
  2>&1 | tee "$report_root/gpu_smoke_bs64.log"

python -u -m tools.train \
  --config configs/c2b_s2looking_decoupled_stage_a.yaml \
  --dataset-root "$S2LOOKING_ROOT" \
  --initial-checkpoint "$report_root/c2b_init.pth" \
  --device cuda \
  2>&1 | tee "$report_root/train.log"

summary="$(
  find outputs/c2b_s2looking_decoupled_stage_a \
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
  --dataset-root "$S2LOOKING_ROOT" \
  --split val \
  --device cuda \
  --batch-size 8 \
  --severities 0 1 2 3 \
  --output "$report_root/registration_val.json" \
  2>&1 | tee "$report_root/registration_val.log"

python -u -m tools.diagnose_transfer_offset_consistency \
  --checkpoint "$checkpoint" \
  --protocol configs/p7_stage_c_s2looking_transfer.yaml \
  --dataset-root "$S2LOOKING_ROOT" \
  --manifest manifests/datasets/s2looking.csv \
  --device cuda \
  --output-dir "$report_root/offset_diagnostic" \
  2>&1 | tee "$report_root/offset_diagnostic.log"

python -m tools.summarize_c2b_stage_a \
  --registration-results "$report_root/registration_val.json" \
  --offset-summary "$report_root/offset_diagnostic/offset_consistency_summary.json" \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

echo "C2b stage-A S2Looking complete: $report_root"
cat "$report_root/decision.json"
