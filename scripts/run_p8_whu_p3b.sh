#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
WHU_ROOT="${WHU_ROOT:-/root/autodl-tmp/rs_cd_work/data/WHU-CD}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

for required in \
  manifests/datasets/whu_cd.csv \
  configs/p8_whu_dinov3_partial_ft.yaml \
  configs/p8_whu_oracle_t2_shift.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p8_whu_p3b/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p8_whu_p3b/LATEST

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

python -m tools.gpu_smoke \
  --config configs/p8_whu_dinov3_partial_ft.yaml \
  --batch-size 32 \
  --image-size 256 \
  --steps 5 \
  --output "$report_root/gpu_smoke_bs32.json" \
  2>&1 | tee "$report_root/gpu_smoke_bs32.log"

python -u -m tools.train \
  --config configs/p8_whu_dinov3_partial_ft.yaml \
  --dataset-root "$WHU_ROOT" \
  --device cuda \
  2>&1 | tee "$report_root/train.log"

summary="$(
  find outputs/p8_whu_dinov3_partial_ft \
    -name summary.json \
    -printf '%T@ %p\n' |
  sort -nr |
  head -n 1 |
  cut -d' ' -f2-
)"
test -n "$summary"
run_dir="$(dirname "$summary")"
checkpoint="$run_dir/checkpoints/best.pth"
test -f "$checkpoint"
printf '%s\n' "$run_dir" > "$report_root/RUN_DIR"
printf '%s\n' "$checkpoint" > "$report_root/CHECKPOINT"
cp "$summary" "$report_root/train_summary.json"

# Decisive gate input: input-level oracle inverse shift on val.
# Needs only the detector and the known injected offset, so it is
# independent of every registration-side instrument problem.
python -u -m tools.evaluate_robustness \
  --checkpoint "$checkpoint" \
  --protocol configs/p8_whu_oracle_t2_shift.yaml \
  --dataset-root "$WHU_ROOT" \
  --split val \
  --device cuda \
  --output-dir "$report_root/oracle_val" \
  --oracle-inverse-shift \
  2>&1 | tee "$report_root/oracle_val.log"

python -u -m tools.evaluate_robustness \
  --checkpoint "$checkpoint" \
  --protocol configs/p8_whu_oracle_t2_shift.yaml \
  --dataset-root "$WHU_ROOT" \
  --split val \
  --device cuda \
  --output-dir "$report_root/raw_val" \
  2>&1 | tee "$report_root/raw_val.log"

python -u -m tools.summarize_p8_whu_headroom \
  --raw "$report_root/raw_val/results.csv" \
  --oracle "$report_root/oracle_val/results.csv" \
  --train-summary "$report_root/train_summary.json" \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

echo "P8 WHU P3b + headroom complete: $report_root"
cat "$report_root/decision.json"
