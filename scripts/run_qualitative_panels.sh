#!/usr/bin/env bash
set -euo pipefail

# Qualitative panels on the LEVIR validation split using frozen weights.
# Reporting-only: no threshold is computed, no gate is evaluated, and the
# sealed test split is never opened.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
SOURCE_P3B="${SOURCE_P3B:-outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

freeze_root="$(cat reports/p11_structure_freeze/LATEST)"
test -f "$freeze_root/STRUCTURE_FROZEN"

read_source() {
  python - "$freeze_root/freeze_manifest.json" "$1" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["sources"][sys.argv[2]])
PY
}

P7_CHECKPOINT="$(read_source p7_checkpoint)"
TASK_ROUTER="$(read_source task_router)"

for required in \
  "$P7_CHECKPOINT" \
  "$TASK_ROUTER" \
  "$SOURCE_P3B" \
  manifests/datasets/levir_cd.csv \
  configs/p12_levir_final_test.yaml; do
  test -f "$required" || {
    echo "Missing input: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
output="reports/qualitative_panels/$tag"

python -u -m tools.render_qualitative_panels \
  --p7-checkpoint "$P7_CHECKPOINT" \
  --detector-checkpoint "$SOURCE_P3B" \
  --classifier-task "$TASK_ROUTER" \
  --protocol configs/p12_levir_final_test.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --split val \
  --device cuda \
  --per-category 3 \
  --output-dir "$output" \
  2>&1 | tee "/tmp/qualitative_panels_${tag}.log"

mkdir -p reports/qualitative_panels
printf '%s\n' "$output" > reports/qualitative_panels/LATEST
cp "/tmp/qualitative_panels_${tag}.log" "$output/run.log"

echo "Qualitative panels: $output/figures"
python - "$output/qualitative_index.json" <<'PY'
import json
import sys

index = json.load(open(sys.argv[1], encoding="utf-8"))
print("split                :", index["split"])
print("test_split_touched   :", index["test_split_touched"])
print("router threshold     :", index["router_threshold"])
for category, figures in index["figures"].items():
    print(f"{category:28s}: {len(figures)} panel(s)")
    for record in index["cases"][category]:
        print(
            f"    {record['sample_id']:>10s} {record['scenario']:>16s}"
            f"  raw {record['raw_f1']:.4f}"
            f"  aligned {record['aligned_f1']:.4f}"
            f"  p {record['router_probability']:.4f}"
            f"  selected {record['selected_aligned']}"
        )
PY
