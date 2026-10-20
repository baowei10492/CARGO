#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
S2_ROOT="${S2_ROOT:-/root/autodl-tmp/rs_cd_work/data/S2Looking}"

P3B_LEVIR_CKPT="${P3B_LEVIR_CKPT:-outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth}"
P5C_LEVIR_CKPT="${P5C_LEVIR_CKPT:-outputs/p5c_a2_gated_multiscale_stage_b/20260725T051644Z_seed3407/checkpoints/epoch_002_threshold020.pth}"
P3B_S2_CKPT="${P3B_S2_CKPT:-outputs/p6_s2looking_dinov3_partial_ft/20260726T014320Z_seed3407/checkpoints/best.pth}"
P5C_S2_CKPT="${P5C_S2_CKPT:-outputs/p6_s2looking_gated_stage_b/20260726T022943Z_seed3407/checkpoints/epoch_002_threshold020.pth}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for required in \
  "$P3B_LEVIR_CKPT" \
  "$P5C_LEVIR_CKPT" \
  "$P3B_S2_CKPT" \
  "$P5C_S2_CKPT" \
  manifests/datasets/levir_cd.csv \
  manifests/datasets/s2looking.csv; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p7_phase0/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p7_phase0/LATEST

python -m pytest -q \
  2>&1 | tee "$report_root/pytest.log"

python -m tools.audit_geometry \
  --protocol configs/p7_geometry_audit.yaml \
  --checkpoint "$P5C_LEVIR_CKPT" \
  --device cuda \
  --output "$report_root/geometry.json" \
  2>&1 | tee "$report_root/geometry.log"

python -m tools.evaluate_feature_oracle \
  --checkpoint "$P5C_LEVIR_CKPT" \
  --protocol configs/p7_feature_oracle_t2.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --split val \
  --device cuda \
  --output-dir "$report_root/levir_feature_oracle" \
  2>&1 | tee "$report_root/levir_feature_oracle.log"

python -m tools.evaluate_registration_controls \
  --checkpoint "$P3B_LEVIR_CKPT" \
  --protocol configs/p7_registration_controls_phase_only_t2.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --split val \
  --device cuda \
  --output-dir "$report_root/levir_phase_control" \
  2>&1 | tee "$report_root/levir_phase_control.log"

python -m tools.evaluate_feature_oracle \
  --checkpoint "$P5C_S2_CKPT" \
  --protocol configs/p7_feature_oracle_t2.yaml \
  --dataset-root "$S2_ROOT" \
  --manifest manifests/datasets/s2looking.csv \
  --split val \
  --device cuda \
  --output-dir "$report_root/s2_feature_oracle" \
  2>&1 | tee "$report_root/s2_feature_oracle.log"

python -m tools.evaluate_registration_controls \
  --checkpoint "$P3B_S2_CKPT" \
  --protocol configs/p7_registration_controls_phase_only_t2.yaml \
  --dataset-root "$S2_ROOT" \
  --manifest manifests/datasets/s2looking.csv \
  --split val \
  --device cuda \
  --output-dir "$report_root/s2_phase_control" \
  2>&1 | tee "$report_root/s2_phase_control.log"

python -m tools.summarize_p7_phase0 \
  --run-root "$report_root" \
  --output "$report_root/decision.json" \
  2>&1 | tee "$report_root/decision.log"

echo "P7 Phase 0 complete: $report_root"
