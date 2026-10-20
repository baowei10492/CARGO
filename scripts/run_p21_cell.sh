#!/usr/bin/env bash
set -euo pipefail

# P21: one detector x dataset cell, end to end, on train/val only.
#
#   scripts/run_p21_cell.sh BIT LEVIR-CD
#
# Everything cell-specific comes from configs/p21_matrix.yaml, so the nine
# cells run one identical chain (docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md,
# sections 2, 5, 6 and amendment A in section 9):
#
#   0  guards: amendment present, matrix entry resolved, targeted tests
#   1  smoke: import / forward / backward, P7 anchor, zero-offset identity
#   2  stage-1 training of the detector under this project's data protocol
#   3  instrument guard on LEVIR-CD, inherited by the other datasets
#   4  raw and input-oracle robustness on val (precondition a, headroom)
#   5  compose the dual-path checkpoint around the frozen localizer
#   6  task-benefit router and 7 safe-alignment classifier (P12 recipe)
#   8  cell summary against the preregistered gates
#
# No test split is opened. A failed guard stops the chain before any
# downstream number exists (main record section 85).

DETECTOR="${1:?usage: run_p21_cell.sh DETECTOR DATASET}"
DATASET="${2:?usage: run_p21_cell.sh DETECTOR DATASET}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
MATRIX="${MATRIX:-configs/p21_matrix.yaml}"
PROTOCOL_DOC=docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "===== 0. GUARDS ====="
grep -q "修订 A" "$PROTOCOL_DOC" || {
  echo "Amendment A is missing from $PROTOCOL_DOC; no P21 cell may run" >&2
  exit 1
}

resolved="$(python - "$MATRIX" "$DETECTOR" "$DATASET" <<'PY'
import shlex
import sys
from pathlib import Path

import yaml

matrix = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
detector_name, dataset_name = sys.argv[2], sys.argv[3]
if detector_name not in matrix["detectors"]:
    raise SystemExit(f"unknown detector {detector_name!r}")
if dataset_name not in matrix["datasets"]:
    raise SystemExit(f"unknown dataset {dataset_name!r}")
detector = matrix["detectors"][detector_name]
dataset = matrix["datasets"][dataset_name]
localizer = dataset["localizer"]
if not isinstance(localizer, dict):
    raise SystemExit(
        f"{dataset_name}: localizer is {localizer!r}; resolve it in "
        f"{sys.argv[1]} (and amendment A) before running this column"
    )
guard = detector["levir_guard"]
values = {
    "DETECTOR_SLUG": detector["slug"],
    "DATASET_SLUG": dataset["slug"],
    "STAGE1_CONFIG": matrix["stage1_config_pattern"].format(
        detector=detector["slug"], dataset=dataset["slug"]
    ),
    "DATASET_ROOT": dataset["dataset_root"],
    "MANIFEST": dataset["manifest"],
    "ORACLE_PROTOCOL": dataset["oracle_protocol"],
    "ROUTER_PROTOCOL": dataset["router_protocol"],
    "CLASSIFIER_PROTOCOL": dataset["classifier_protocol"],
    "LOCALIZER_POINTER": localizer["pointer"],
    "LOCALIZER_FILE": localizer["file"],
    "LOCALIZER_REQUIRE": " ".join(localizer.get("require_files", [])),
    "GUARD_MODE": dataset["instrument_guard"],
    "GUARD_KIND": guard["kind"],
    "GUARD_VALUE": guard.get("value", ""),
    "GUARD_REFERENCE": guard.get("reference_detector", ""),
    "GUARD_TOLERANCE": matrix["guard_tolerance"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
eval "$resolved"

localizer_report="$(cat "$LOCALIZER_POINTER")"
test ! -f "$localizer_report/VOID.json" || {
  echo "Localizer report $localizer_report is voided" >&2
  exit 1
}
for required in $LOCALIZER_REQUIRE; do
  test -f "$localizer_report/$required" || {
    echo "Localizer report $localizer_report lacks $required" >&2
    exit 1
  }
done
LOCALIZER_CHECKPOINT="$(cat "$localizer_report/$LOCALIZER_FILE")"
for required in \
  "$STAGE1_CONFIG" "$MANIFEST" "$ORACLE_PROTOCOL" \
  "$ROUTER_PROTOCOL" "$CLASSIFIER_PROTOCOL" "$LOCALIZER_CHECKPOINT"; do
  test -f "$required" || {
    echo "Missing P21 input: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
cell_root="reports/p21/cells/${DETECTOR_SLUG}_${DATASET_SLUG}"
report_root="$cell_root/$tag"
mkdir -p "$report_root"
printf '%s\n' "$resolved" > "$report_root/RESOLVED_CELL.env"
printf '%s\n' "$localizer_report" > "$report_root/LOCALIZER_REPORT"
printf '%s\n' "$LOCALIZER_CHECKPOINT" > "$report_root/LOCALIZER_CHECKPOINT"
sha256sum "$LOCALIZER_CHECKPOINT" | tee "$report_root/LOCALIZER_SHA256"

# Protocol section 2: the rschange commit is recorded and the checkout must
# be unmodified. Untracked files are ignored so placed weights (SeaFormer
# under review_pretrain/) do not count as edits.
rschange_root="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["model"]["repo_root"])' "$STAGE1_CONFIG")"
git -C "$rschange_root" rev-parse HEAD | tee "$report_root/RSCHANGE_COMMIT"
test -z "$(git -C "$rschange_root" status --porcelain --untracked-files=no)" || {
  echo "rschange checkout at $rschange_root has modified tracked files; the protocol forbids editing it" >&2
  exit 1
}

python -m pytest -q \
  tests/test_external_rschange.py \
  tests/test_dual_path_adapter.py \
  tests/test_p21_dualpath_checkpoint.py \
  tests/test_p21_matrix.py \
  tests/test_train_optimizer.py \
  2>&1 | tee "$report_root/pytest.log"

echo "===== 1. SMOKE: IMPORT, FORWARD, BACKWARD, P7 ANCHOR ====="
python -u -m tools.check_p21_detectors \
  --config "$STAGE1_CONFIG" \
  --localizer-checkpoint "$LOCALIZER_CHECKPOINT" \
  --device cuda \
  --output "$report_root/smoke.json" \
  2>&1 | tee "$report_root/smoke.log"
python - "$report_root/smoke.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report["all_ok"]:
    raise SystemExit(
        "Smoke test failed; see smoke.json. Nothing may be trained until "
        "import, forward, the P7 anchor and zero-offset identity all pass."
    )
PY

echo "===== 2. STAGE-1 TRAINING (TRAIN + VAL ONLY) ====="
experiment_name="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["experiment"]["name"])' "$STAGE1_CONFIG")"
marker="$report_root/.before_stage1"
touch "$marker"
python -u -m tools.train \
  --config "$STAGE1_CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --device cuda \
  2>&1 | tee "$report_root/stage1_train.log"
summary="$(
  find "outputs/$experiment_name" -name summary.json -newer "$marker" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
)"
test -n "$summary"
run_dir="$(dirname "$summary")"
DETECTOR_CHECKPOINT="$run_dir/checkpoints/best.pth"
test -f "$DETECTOR_CHECKPOINT"
printf '%s\n' "$run_dir" > "$report_root/STAGE1_RUN_DIR"
printf '%s\n' "$DETECTOR_CHECKPOINT" > "$report_root/DETECTOR_CHECKPOINT"
cp "$summary" "$report_root/stage1_summary.json"

echo "===== 3. INSTRUMENT GUARD ====="
python - "$report_root/stage1_summary.json" "$GUARD_MODE" "$GUARD_KIND" \
  "$GUARD_VALUE" "$GUARD_REFERENCE" "$GUARD_TOLERANCE" "$DETECTOR_SLUG" \
  "$report_root/instrument_guard.json" <<'PY'
import json
import sys
from pathlib import Path

(summary_path, mode, kind, value, reference, tolerance, detector_slug,
 output) = sys.argv[1:]
observed = float(json.loads(Path(summary_path).read_text(encoding="utf-8"))["best_f1"])
tolerance = float(tolerance)


def latest_passed_guard(slug: str) -> tuple[Path, dict] | None:
    passed = []
    for path in sorted(Path(f"reports/p21/cells/{slug}_levir").glob("*/instrument_guard.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("passed"):
            passed.append((path, record))
    return passed[-1] if passed else None


record = {"mode": mode, "kind": kind, "stage1_best_val_f1": observed,
          "tolerance": tolerance}
if mode == "levir" and kind == "published":
    record.update(published_levir_f1=float(value),
                  passed=abs(observed - float(value)) <= tolerance)
elif mode == "levir" and kind == "relative":
    # No published LEVIR-CD value exists for this detector (amendment A,
    # 9.5): only a grossly broken port is caught, and the record says so.
    found = latest_passed_guard(reference)
    if found is None:
        raise SystemExit(
            f"Run the reference detector's ({reference}) LEVIR-CD cell first"
        )
    reference_path, reference_record = found
    minimum = float(reference_record["stage1_best_val_f1"]) - tolerance
    record.update(reference_guard=str(reference_path),
                  reference_f1=reference_record["stage1_best_val_f1"],
                  minimum=minimum, passed=observed >= minimum,
                  strength="weak: relative to a peer, not a published value")
elif mode == "inherit_levir":
    # Published DSIFN/GZ-CD numbers use other splits, so the port is
    # validated once on LEVIR-CD and inherited by the detector's other cells.
    found = latest_passed_guard(detector_slug)
    record.update(inherited_from=str(found[0]) if found else None,
                  passed=found is not None)
else:
    raise SystemExit(f"unknown guard mode/kind {mode!r}/{kind!r}")
Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
if not record["passed"]:
    raise SystemExit(
        "Instrument guard failed: diagnose the port (or run this detector's "
        "LEVIR-CD cell first) before reading any displacement number."
    )
PY

echo "===== 4. RAW AND INPUT-ORACLE ROBUSTNESS ON VAL ====="
python -u -m tools.evaluate_robustness \
  --checkpoint "$DETECTOR_CHECKPOINT" \
  --protocol "$ORACLE_PROTOCOL" \
  --dataset-root "$DATASET_ROOT" \
  --split val \
  --device cuda \
  --paired-shift-support \
  --output-dir "$report_root/raw_val" \
  2>&1 | tee "$report_root/raw_val.log"
python -u -m tools.evaluate_robustness \
  --checkpoint "$DETECTOR_CHECKPOINT" \
  --protocol "$ORACLE_PROTOCOL" \
  --dataset-root "$DATASET_ROOT" \
  --split val \
  --device cuda \
  --paired-shift-support \
  --oracle-inverse-shift \
  --output-dir "$report_root/oracle_val" \
  2>&1 | tee "$report_root/oracle_val.log"
# Reported, not a P21 gate: the cell verdict uses precondition (a) and the
# two hard gates from the classifier run below.
python -u -m tools.summarize_p8_whu_headroom \
  --dataset "$DATASET" \
  --raw "$report_root/raw_val/results.csv" \
  --oracle "$report_root/oracle_val/results.csv" \
  --train-summary "$report_root/stage1_summary.json" \
  --shift8-headroom-min 0.05 \
  --output "$report_root/headroom.json" \
  2>&1 | tee "$report_root/headroom.log"

echo "===== 5. COMPOSE DUAL-PATH CHECKPOINT ====="
python -u -m tools.build_p21_dualpath_checkpoint \
  --localizer-checkpoint "$LOCALIZER_CHECKPOINT" \
  --detector-checkpoint "$DETECTOR_CHECKPOINT" \
  --detector "$DETECTOR" \
  --dataset "$DATASET" \
  --output "$report_root/dual_path.pth" \
  2>&1 | tee "$report_root/compose.log"

echo "===== 6. TASK-BENEFIT ROUTER ====="
python -u -m tools.train_evaluate_task_benefit_router \
  --checkpoint "$report_root/dual_path.pth" \
  --protocol "$ROUTER_PROTOCOL" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" \
  --device cuda \
  --output-dir "$report_root/router_experiment" \
  2>&1 | tee "$report_root/router.log"

echo "===== 7. SAFE-ALIGNMENT CLASSIFIER ====="
python -u -m tools.train_evaluate_safe_alignment_classifier \
  --checkpoint "$report_root/dual_path.pth" \
  --features-csv "$report_root/router_experiment/training_router_features.csv" \
  --task-router-checkpoint "$report_root/router_experiment/task_router.pth" \
  --protocol "$CLASSIFIER_PROTOCOL" \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$MANIFEST" \
  --device cuda \
  --output-dir "$report_root/safe_alignment_experiment" \
  2>&1 | tee "$report_root/classifier.log"

echo "===== 8. CELL SUMMARY AGAINST PREREGISTERED GATES ====="
python -u -m tools.summarize_p21_matrix \
  --scenario-csv "$report_root/safe_alignment_experiment/validation_scenario_metrics.csv" \
  --detector "$DETECTOR" \
  --dataset "$DATASET" \
  --output "$report_root/cell_summary.json" \
  2>&1 | tee "$report_root/cell_summary.log"

cat > "$report_root/TEST_SPLIT_STATUS.txt" <<'EOF'
No test split was opened by this cell. P21 test consumption is deferred by
protocol section 7 until every cell has a frozen validation report.
EOF

sha256sum \
  "$MATRIX" "$PROTOCOL_DOC" "$STAGE1_CONFIG" "$MANIFEST" \
  "$ROUTER_PROTOCOL" "$CLASSIFIER_PROTOCOL" \
  "$LOCALIZER_CHECKPOINT" "$DETECTOR_CHECKPOINT" \
  "$report_root/dual_path.pth" \
  "$report_root/safe_alignment_experiment/validation_scenario_metrics.csv" \
  "$report_root/cell_summary.json" \
  > "$report_root/SHA256SUMS.txt"
rm -f "$marker"
printf '%s\n' "$report_root" > "$cell_root/LATEST"

echo "P21 cell complete: $report_root"
cat "$report_root/cell_summary.json"
