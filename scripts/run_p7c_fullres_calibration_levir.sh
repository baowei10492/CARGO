#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7B_CKPT:-}" ]]; then
  p7b_report="$(cat reports/p7b_stage1_levir/LATEST)"
  P7B_CKPT="$(cat "$p7b_report/CHECKPOINT")"
fi

for required in \
  "$P7B_CKPT" \
  manifests/datasets/levir_cd.csv \
  configs/p7c_fullres_shift_calibration.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p7c_fullres_calibration_levir/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p7c_fullres_calibration_levir/LATEST

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

python -m tools.gpu_smoke \
  --config configs/p7c_fullres_shift_calibration.yaml \
  --batch-size 4 \
  --image-size 1024 \
  --steps 3 \
  --output "$report_root/gpu_smoke_bs4_1024.json" \
  2>&1 | tee "$report_root/gpu_smoke_bs4_1024.log"

python -u -m tools.train \
  --config configs/p7c_fullres_shift_calibration.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --initial-checkpoint "$P7B_CKPT" \
  --device cuda \
  2>&1 | tee "$report_root/train.log"

summary="$(
  find outputs/p7c_fullres_shift_calibration \
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
  --dataset-root "$LEVIR_ROOT" \
  --split val \
  --device cuda \
  --batch-size 8 \
  --severities 0 1 2 3 \
  --output "$report_root/registration_val.json" \
  2>&1 | tee "$report_root/registration_val.log"

python -m tools.summarize_p7_stage1 \
  --registration-results "$report_root/registration_val.json" \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

estimator_passed="$(
  python -c \
    "import json; print(str(json.load(open('$report_root/decision.json'))['estimator_passed']).lower())"
)"
if [[ "$estimator_passed" == "true" ]]; then
  python -m tools.evaluate_input_alignment_probe \
    --checkpoint "$checkpoint" \
    --dataset-root "$LEVIR_ROOT" \
    --split val \
    --device cuda \
    --batch-size 4 \
    --offset-decoding hard \
    --alignment-gate threshold_nonzero \
    --zero-probability-threshold 0.5 \
    --output "$report_root/input_alignment_val.json" \
    2>&1 | tee "$report_root/input_alignment_val.log"
else
  printf '%s\n' \
    "Estimator gates failed; input-alignment segmentation evaluation skipped." \
    | tee "$report_root/input_alignment_skipped.txt"
fi

echo "P7-1C LEVIR complete: $report_root"
cat "$report_root/decision.json"
