#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
P7_CHECKPOINT="${P7_CHECKPOINT:-$(cat "$p7c_report/CHECKPOINT")}"
SAFE_REPORT="${SAFE_REPORT:-reports/p7_safe_alignment_classifier_levir/20260726T080834Z}"
if [[ ! -d "$SAFE_REPORT" ]]; then
  SAFE_REPORT="$(cat reports/p7_safe_alignment_classifier_levir/LATEST)"
fi
EXPERIMENT="$SAFE_REPORT/safe_alignment_experiment"
TASK_ROUTER="${TASK_ROUTER:-$EXPERIMENT/safe_alignment_task.pth}"
REGISTRATION_ROUTER="${REGISTRATION_ROUTER:-$EXPERIMENT/safe_alignment_registration.pth}"
P10_REPORT="${P10_REPORT:-$(cat reports/p10_efficiency/LATEST)}"

files=(
  "$P7_CHECKPOINT"
  "$TASK_ROUTER"
  "$REGISTRATION_ROUTER"
  "$P10_REPORT/efficiency.json"
  src/rs_cd/dual_path.py
  src/rs_cd/geometry.py
  src/rs_cd/robustness.py
  src/rs_cd/models/dinov3_task_aware_input_registration_cd.py
  configs/p7_safe_alignment_classifier.yaml
  configs/p10_efficiency.yaml
  tools/train_evaluate_safe_alignment_classifier.py
  tools/evaluate_perturbation_robustness.py
  tools/benchmark_p10_efficiency.py
  docs/P11_STRUCTURE_FREEZE.md
)
for required in "${files[@]}"; do
  test -f "$required" || {
    echo "Missing freeze input: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p11_structure_freeze/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p11_structure_freeze/LATEST
sha256sum "${files[@]}" > "$report_root/FROZEN_SHA256SUMS.txt"

export FREEZE_REPORT_ROOT="$report_root"
export FREEZE_P7_CHECKPOINT="$P7_CHECKPOINT"
export FREEZE_TASK_ROUTER="$TASK_ROUTER"
export FREEZE_REGISTRATION_ROUTER="$REGISTRATION_ROUTER"
export FREEZE_P10_REPORT="$P10_REPORT"
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["FREEZE_REPORT_ROOT"])
manifest = {
    "status": "structure_frozen",
    "frozen_utc": datetime.now(timezone.utc).isoformat(),
    "primary_method": "safe_veto_task",
    "safety_efficiency_ablation": "safe_veto_registration",
    "fallback": "raw",
    "task_veto_execution": "constant_dual_path",
    "registration_veto_execution": "conditional_aligned_path",
    "p9b_primary_passed": False,
    "p9b_registration_ablation_passed": True,
    "test_policy": {
        "final_p7_levir_test_runs_allowed": 1,
        "final_p7_s2looking_test_runs_allowed": 1,
        "test_remains_sealed_until_second_detector_seed": True,
    },
    "sources": {
        "p7_checkpoint": os.environ["FREEZE_P7_CHECKPOINT"],
        "task_router": os.environ["FREEZE_TASK_ROUTER"],
        "registration_router": os.environ["FREEZE_REGISTRATION_ROUTER"],
        "p10_report": os.environ["FREEZE_P10_REPORT"],
        "hash_manifest": str(root / "FROZEN_SHA256SUMS.txt"),
    },
    "prohibited_post_freeze": [
        "change_router_features",
        "retune_probability_threshold",
        "change_primary_method",
        "select_seed_by_validation_or_test_superiority",
        "repeat_test_evaluation",
    ],
    "next_action": "p11_second_detector_seed_replication",
}
(root / "freeze_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

sha256sum "$report_root/freeze_manifest.json" \
  >> "$report_root/FROZEN_SHA256SUMS.txt"
touch "$report_root/STRUCTURE_FROZEN"

echo "P11 structure freeze complete: $report_root"
cat "$report_root/freeze_manifest.json"
echo
cat "$report_root/FROZEN_SHA256SUMS.txt"
