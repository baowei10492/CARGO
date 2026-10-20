#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
SEED="${SEED:-1234}"
SOURCE_P3B="${SOURCE_P3B:-outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth}"

if [[ "$SEED" != "1234" ]]; then
  echo "Frozen replication seed is 1234, got $SEED" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

freeze_root="$(cat reports/p11_structure_freeze/LATEST)"
test -f "$freeze_root/STRUCTURE_FROZEN"
sha256sum -c "$freeze_root/FROZEN_SHA256SUMS.txt"

for required in \
  "$SOURCE_P3B" \
  manifests/datasets/levir_cd.csv \
  configs/p7_task_aware_input_registration_stage_a.yaml \
  configs/p7b_decoupled_input_registration_stage_a.yaml \
  configs/p7c_fullres_shift_calibration.yaml \
  configs/p7_task_benefit_router.yaml \
  configs/p7_safe_alignment_classifier.yaml \
  docs/P11_SECOND_DETECTOR_SEED.md; do
  test -f "$required" || {
    echo "Missing replication input: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}"
report_root="reports/p11_second_detector_seed/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p11_second_detector_seed/LATEST
printf '%s\n' "$freeze_root" > "$report_root/SOURCE_FREEZE"
printf '%s\n' "$SOURCE_P3B" > "$report_root/SOURCE_P3B"

export P11_REPORT_ROOT="$report_root"
export P11_SOURCE_P3B="$SOURCE_P3B"
export P11_SEED="$SEED"
python - <<'PY'
import os
from pathlib import Path

import torch
import yaml

root = Path(os.environ["P11_REPORT_ROOT"])
source = Path(os.environ["P11_SOURCE_P3B"])
seed = int(os.environ["P11_SEED"])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
config = checkpoint["config"]
config["experiment"]["name"] = f"p11_p3b_seed{seed}"
config["experiment"]["seed"] = seed
config["experiment"]["output_root"] = "outputs"
config["data"]["dataset_root"] = (
    "/root/autodl-tmp/rs_cd_work/data/LEVIR-CD"
)
config["data"]["manifest"] = "manifests/datasets/levir_cd.csv"
config["training"]["initial_checkpoint"] = None
(root / "p3b_seed1234_from_frozen.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

router = yaml.safe_load(
    Path("configs/p7_task_benefit_router.yaml").read_text(
        encoding="utf-8"
    )
)
router["seed"] = seed
(root / "p7_task_benefit_router_seed1234.yaml").write_text(
    yaml.safe_dump(router, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY

sha256sum \
  "$SOURCE_P3B" \
  "$report_root/p3b_seed1234_from_frozen.yaml" \
  "$report_root/p7_task_benefit_router_seed1234.yaml" \
  configs/p7_task_aware_input_registration_stage_a.yaml \
  configs/p7b_decoupled_input_registration_stage_a.yaml \
  configs/p7c_fullres_shift_calibration.yaml \
  configs/p7_safe_alignment_classifier.yaml \
  > "$report_root/INPUT_SHA256SUMS.txt"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

echo "===== P11-1 TRAIN P3B SEED 1234 ====="
python -u -m tools.train \
  --config "$report_root/p3b_seed1234_from_frozen.yaml" \
  --dataset-root "$LEVIR_ROOT" \
  --seed "$SEED" \
  --device cuda \
  2>&1 | tee "$report_root/p3b_train.log"

p3b_summary="$(
  find "outputs/p11_p3b_seed${SEED}" \
    -name summary.json -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$p3b_summary"
p3b_run="$(dirname "$p3b_summary")"
p3b_checkpoint="$p3b_run/checkpoints/best.pth"
test -f "$p3b_checkpoint"
printf '%s\n' "$p3b_run" > "$report_root/P3B_RUN_DIR"
printf '%s\n' "$p3b_checkpoint" > "$report_root/P3B_CHECKPOINT"
cp "$p3b_summary" "$report_root/p3b_summary.json"

echo "===== P11-2 TRAIN P7-1A SEED 1234 ====="
marker_a="$report_root/.before_p7a"
touch "$marker_a"
python -u -m tools.train \
  --config configs/p7_task_aware_input_registration_stage_a.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --initial-checkpoint "$p3b_checkpoint" \
  --seed "$SEED" \
  --device cuda \
  2>&1 | tee "$report_root/p7a_train.log"
p7a_summary="$(
  find outputs/p7_task_aware_input_registration_stage_a \
    -name summary.json -newer "$marker_a" -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$p7a_summary"
p7a_run="$(dirname "$p7a_summary")"
p7a_checkpoint="$p7a_run/checkpoints/last.pth"
test -f "$p7a_checkpoint"
printf '%s\n' "$p7a_checkpoint" > "$report_root/P7A_CHECKPOINT"
cp "$p7a_summary" "$report_root/p7a_summary.json"

echo "===== P11-3 TRAIN P7-1B SEED 1234 ====="
marker_b="$report_root/.before_p7b"
touch "$marker_b"
python -u -m tools.train \
  --config configs/p7b_decoupled_input_registration_stage_a.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --initial-checkpoint "$p7a_checkpoint" \
  --seed "$SEED" \
  --device cuda \
  2>&1 | tee "$report_root/p7b_train.log"
p7b_summary="$(
  find outputs/p7b_decoupled_input_registration_stage_a \
    -name summary.json -newer "$marker_b" -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$p7b_summary"
p7b_run="$(dirname "$p7b_summary")"
p7b_checkpoint="$p7b_run/checkpoints/last.pth"
test -f "$p7b_checkpoint"
printf '%s\n' "$p7b_checkpoint" > "$report_root/P7B_CHECKPOINT"
cp "$p7b_summary" "$report_root/p7b_summary.json"

echo "===== P11-4 TRAIN P7-1C SEED 1234 ====="
marker_c="$report_root/.before_p7c"
touch "$marker_c"
python -u -m tools.train \
  --config configs/p7c_fullres_shift_calibration.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --initial-checkpoint "$p7b_checkpoint" \
  --seed "$SEED" \
  --device cuda \
  2>&1 | tee "$report_root/p7c_train.log"
p7c_summary="$(
  find outputs/p7c_fullres_shift_calibration \
    -name summary.json -newer "$marker_c" -printf '%T@ %p\n' |
  sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$p7c_summary"
p7c_run="$(dirname "$p7c_summary")"
p7c_checkpoint="$p7c_run/checkpoints/last.pth"
test -f "$p7c_checkpoint"
printf '%s\n' "$p7c_checkpoint" > "$report_root/P7C_CHECKPOINT"
cp "$p7c_summary" "$report_root/p7c_summary.json"

python -m tools.evaluate_registration \
  --checkpoint "$p7c_checkpoint" \
  --dataset-root "$LEVIR_ROOT" \
  --split val \
  --device cuda \
  --batch-size 8 \
  --severities 0 1 2 3 \
  --output "$report_root/p7c_registration_val.json" \
  2>&1 | tee "$report_root/p7c_registration_val.log"

echo "===== P11-5 GENERATE P7-2 FEATURES SEED 1234 ====="
router_dir="$report_root/task_benefit_router"
python -u -m tools.train_evaluate_task_benefit_router \
  --checkpoint "$p7c_checkpoint" \
  --protocol "$report_root/p7_task_benefit_router_seed1234.yaml" \
  --dataset-root "$LEVIR_ROOT" \
  --device cuda \
  --output-dir "$router_dir" \
  2>&1 | tee "$report_root/task_benefit_router.log"

echo "===== P11-6 REPLICATE FROZEN P7-3 SEED 1234 ====="
safe_dir="$report_root/safe_alignment_experiment"
python -u -m tools.train_evaluate_safe_alignment_classifier \
  --checkpoint "$p7c_checkpoint" \
  --features-csv "$router_dir/training_router_features.csv" \
  --task-router-checkpoint "$router_dir/task_router.pth" \
  --protocol configs/p7_safe_alignment_classifier.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --device cuda \
  --seed "$SEED" \
  --output-dir "$safe_dir" \
  2>&1 | tee "$report_root/safe_alignment_experiment.log"

cp "$safe_dir/decision.json" "$report_root/decision.json"

export P11_P3B_CHECKPOINT="$p3b_checkpoint"
export P11_P7C_CHECKPOINT="$p7c_checkpoint"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["P11_REPORT_ROOT"])
decision = json.load(open(root / "decision.json", encoding="utf-8"))
p3b = json.load(open(root / "p3b_summary.json", encoding="utf-8"))
p7c = json.load(open(root / "p7c_summary.json", encoding="utf-8"))
summary = {
    "status": "completed",
    "seed": int(os.environ["P11_SEED"]),
    "structure_freeze": (root / "SOURCE_FREEZE").read_text().strip(),
    "p3b_checkpoint": os.environ["P11_P3B_CHECKPOINT"],
    "p3b_best_epoch": p3b["best_epoch"],
    "p3b_best_val_f1": p3b["best_f1"],
    "p7c_checkpoint": os.environ["P11_P7C_CHECKPOINT"],
    "p7c_best_epoch": p7c["best_epoch"],
    "p7c_best_val_f1": p7c["best_f1"],
    "p7_3_decision": decision,
    "test_split_untouched": True,
    "selection_policy": (
        "replication only; seed must not be selected by superiority"
    ),
}
(root / "pipeline_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

sha256sum \
  "$p3b_checkpoint" \
  "$p7c_checkpoint" \
  "$router_dir/task_router.pth" \
  "$safe_dir/safe_alignment_task.pth" \
  "$safe_dir/safe_alignment_registration.pth" \
  "$report_root/pipeline_summary.json" \
  > "$report_root/OUTPUT_SHA256SUMS.txt"

echo "P11 second detector seed replication complete: $report_root"
cat "$report_root/pipeline_summary.json"
