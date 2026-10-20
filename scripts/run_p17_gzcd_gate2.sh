#!/usr/bin/env bash
set -euo pipefail

# P17 Gate 2: the full chain on GZ-CD.
#
# Main record 120.4 established that the preregistered absolute gates
# cannot be reached here, because the oracle ceiling measured in Gate 1
# sits below them. The run proceeds anyway for one reason: to measure how
# much of that ceiling the router actually captures, so that "the gate is
# unreachable" is separated from "the method does not work". The verdict
# against the frozen gates is recorded unchanged, and the capture rate is
# descriptive only.
#
# Four stages, all on train/val. The GZ-CD test split is never opened.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
GZCD_ROOT="${GZCD_ROOT:-/root/autodl-tmp/rs_cd_work/data/GZ-CD-tiles}"
MANIFEST="${MANIFEST:-manifests/datasets/gzcd.csv}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

STAGE_A_CONFIG=configs/p17_gzcd_decoupled_stage_a.yaml
CALIBRATION_CONFIG=configs/p17_gzcd_fullres_calibration.yaml
ROUTER_CONFIG=configs/p17_gzcd_task_benefit_router.yaml
CLASSIFIER_CONFIG=configs/p17_gzcd_safe_alignment_classifier.yaml

gate1_root="$(cat reports/p17_gzcd_gate1/LATEST)"
for required in \
  "$MANIFEST" \
  "$STAGE_A_CONFIG" \
  "$CALIBRATION_CONFIG" \
  "$ROUTER_CONFIG" \
  "$CLASSIFIER_CONFIG" \
  "$gate1_root/gate1_decision.json"; do
  test -f "$required" || {
    echo "Missing Gate 2 input: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p17_gzcd_gate2/$tag"
mkdir -p "$report_root"

echo "===== PREREQUISITE AND CEILING GUARD ====="
python - "$gate1_root/gate1_decision.json" "$CLASSIFIER_CONFIG" \
  "$report_root/ceiling_guard.json" <<'PY'
import json
import sys
from pathlib import Path

import yaml

gate1 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gates = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))["gates"]
if not gate1.get("decisive_gate_passed"):
    raise SystemExit("Gate 1 did not pass; Gate 2 must not run")

headroom = gate1["input_oracle_headroom"]
ceiling_mean = sum(headroom.values()) / len(headroom)
ceiling_shift8 = headroom["shift:s3:t2"]
record = {
    "status": "recorded_before_run",
    "gate1_headroom": headroom,
    "oracle_ceiling_mean": ceiling_mean,
    "oracle_ceiling_shift8": ceiling_shift8,
    "required_mean_gain": gates["shift_mean_f1_gain_min"],
    "required_shift8_gain": gates["shift8_f1_gain_min"],
    "mean_gain_reachable": ceiling_mean >= gates["shift_mean_f1_gain_min"],
    "shift8_gain_reachable": ceiling_shift8 >= gates["shift8_f1_gain_min"],
    "note": (
        "Recorded before the run so the outcome cannot be reinterpreted "
        "afterwards. Gates are not modified."
    ),
}
Path(sys.argv[3]).write_text(
    json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(record, indent=2, ensure_ascii=False))
PY

echo "===== TARGETED TESTS ====="
python -m pytest -q \
  tests/test_safe_alignment.py \
  tests/test_dual_path.py \
  tests/test_gzcd_tiles.py \
  2>&1 | tee "$report_root/pytest.log"

echo "===== WARM START: LEVIR REGISTRATION STACK + GZ-CD DETECTOR ====="
# Without this the change detector inside the P7 model is randomly
# initialised. The first Gate 2 attempt omitted it and produced a raw
# validation F1 of 0.178, the degenerate all-change value, instead of the
# 0.7791 Gate 1 measured, so every downstream number was meaningless.
if [[ -z "${P7C_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7C_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi
if [[ -z "${DETECTOR_CHECKPOINT:-}" ]]; then
  DETECTOR_CHECKPOINT="$(
    find outputs/p17_gzcd_dinov3_partial_ft -name best.pth \
      -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
fi
test -f "$P7C_CHECKPOINT"
test -f "$DETECTOR_CHECKPOINT"
printf '%s\n' "$P7C_CHECKPOINT" > "$report_root/SOURCE_P7C_CHECKPOINT"
printf '%s\n' "$DETECTOR_CHECKPOINT" > "$report_root/SOURCE_DETECTOR_CHECKPOINT"

python -m tools.build_c2b_init_checkpoint \
  --p7-checkpoint "$P7C_CHECKPOINT" \
  --detector-checkpoint "$DETECTOR_CHECKPOINT" \
  --output "$report_root/gzcd_init.pth" \
  2>&1 | tee "$report_root/init_checkpoint.log"

echo "===== STAGE 1/4: LOCALIZER STAGE A ====="
python -u -m tools.train \
  --config "$STAGE_A_CONFIG" \
  --dataset-root "$GZCD_ROOT" \
  --initial-checkpoint "$report_root/gzcd_init.pth" \
  --device cuda \
  2>&1 | tee "$report_root/stage_a.log"

echo "===== DETECTOR SANITY GUARD ====="
# The frozen detector must reproduce the accuracy Gate 1 measured.
python - "$report_root/stage_a.log" "$gate1_root/gate1_decision.json" <<'PYGUARD'
import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
expected = float(
    json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["clean_f1"]
)
values = [float(value) for value in re.findall(r"val_f1=([0-9.]+)", log)]
if not values:
    raise SystemExit("No val_f1 lines found in the stage-A log")
observed = max(values)
if abs(observed - expected) > 0.05:
    raise SystemExit(
        f"Detector sanity guard failed: stage-A val F1 {observed:.4f} "
        f"does not reproduce the Gate 1 detector {expected:.4f}. The warm "
        "start did not load the trained detector, so no downstream number "
        "may be read."
    )
print(f"detector reproduces Gate 1: {observed:.4f} vs {expected:.4f}")
PYGUARD

stage_a_summary="$(
  find outputs/p17_gzcd_decoupled_stage_a -name summary.json \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$stage_a_summary"
STAGE_A_CHECKPOINT="$(dirname "$stage_a_summary")/checkpoints/best.pth"
test -f "$STAGE_A_CHECKPOINT"
printf '%s\n' "$STAGE_A_CHECKPOINT" > "$report_root/STAGE_A_CHECKPOINT"

echo "===== STAGE 2/4: FULL-RESOLUTION q CALIBRATION ====="
python -u -m tools.train \
  --config "$CALIBRATION_CONFIG" \
  --dataset-root "$GZCD_ROOT" \
  --device cuda \
  --initial-checkpoint "$STAGE_A_CHECKPOINT" \
  2>&1 | tee "$report_root/calibration.log"

calibration_summary="$(
  find outputs/p17_gzcd_fullres_calibration -name summary.json \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$calibration_summary"
P7_CHECKPOINT="$(dirname "$calibration_summary")/checkpoints/best.pth"
test -f "$P7_CHECKPOINT"
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/P7_CHECKPOINT"

echo "===== STAGE 3/4: TASK-BENEFIT ROUTER ====="
python -u -m tools.train_evaluate_task_benefit_router \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol "$ROUTER_CONFIG" \
  --dataset-root "$GZCD_ROOT" \
  --manifest "$MANIFEST" \
  --device cuda \
  --output-dir "$report_root/router_experiment" \
  2>&1 | tee "$report_root/router.log"

echo "===== STAGE 4/4: SAFE ALIGNMENT CLASSIFIER ====="
python -u -m tools.train_evaluate_safe_alignment_classifier \
  --checkpoint "$P7_CHECKPOINT" \
  --features-csv "$report_root/router_experiment/training_router_features.csv" \
  --task-router-checkpoint "$report_root/router_experiment/task_router.pth" \
  --protocol "$CLASSIFIER_CONFIG" \
  --dataset-root "$GZCD_ROOT" \
  --manifest "$MANIFEST" \
  --device cuda \
  --output-dir "$report_root/safe_alignment_experiment" \
  2>&1 | tee "$report_root/classifier.log"

echo "===== ORACLE CAPTURE RATE (DESCRIPTIVE, NOT A GATE) ====="
python -u -m tools.summarize_oracle_capture \
  --scenario-metrics "$report_root/safe_alignment_experiment/validation_scenario_metrics.csv" \
  --decision "$report_root/safe_alignment_experiment/decision.json" \
  --protocol "$CLASSIFIER_CONFIG" \
  --dataset GZ-CD \
  --output-json "$report_root/oracle_capture.json" \
  2>&1 | tee "$report_root/oracle_capture.log"

printf '%s\n' "$report_root" > reports/p17_gzcd_gate2/LATEST
cat > "$report_root/TEST_SPLIT_STATUS.txt" <<'EOF'
The GZ-CD test split was not opened by this run. Gate 3 requires Gate 2
to pass against the preregistered absolute gates; main record 120.4
established that those gates are unreachable on this dataset, so the
one-shot test stays sealed.
EOF

sha256sum \
  "$MANIFEST" \
  "$STAGE_A_CONFIG" "$CALIBRATION_CONFIG" \
  "$ROUTER_CONFIG" "$CLASSIFIER_CONFIG" \
  "$P7_CHECKPOINT" \
  "$report_root/safe_alignment_experiment/decision.json" \
  "$report_root/oracle_capture.json" \
  > "$report_root/SHA256SUMS.txt"

echo "P17 Gate 2 complete: $report_root"
cat "$report_root/safe_alignment_experiment/decision.json"
