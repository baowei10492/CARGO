#!/usr/bin/env bash
set -euo pipefail

# P22 Gate 1 (docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md section 3).
# Validation tiles with labels, frozen P8 WHU detector and frozen LEVIR
# localizer. Runs only after Gate 0 authorised it; the anchor must reproduce
# P8 Gate 1 before the main measurement starts. The WHU test raster is never
# opened.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p22_whu_physical_shift.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
gate0_root="$(cat reports/p22_whu/gate0/LATEST)"
python - "$gate0_root/gate0_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
if decision["verdict"] != "proceed_to_gate1":
    raise SystemExit(f"Gate 0 does not authorise Gate 1: {decision['verdict']}")
PY

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p22_whu/gate1/$tag"
mkdir -p "$report_root"
printf '%s\n' "$gate0_root" > "$report_root/GATE0_REPORT"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_physical_shift.py 2>&1 | tee "$report_root/pytest.log"

echo "===== GATE 1: ANCHOR THEN MAIN MEASUREMENT ====="
python -u -m tools.run_p22_whu_physical_shift gate1 \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/gate1.log"

cp "$report_root/run/gate1_decision.json" "$report_root/gate1_decision.json"
sha256sum "$CONFIG" docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md \
  "$report_root"/run/*.json "$report_root"/run/*.csv \
  > "$report_root/SHA256SUMS.txt" 2>/dev/null || true
printf '%s\n' "$report_root" > reports/p22_whu/gate1/LATEST

echo "P22 Gate 1 complete: $report_root"
python - "$report_root/gate1_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
print("verdict:", decision["verdict"])
if "f1" not in decision:
    print(json.dumps(decision.get("anchor"), indent=2))
    raise SystemExit(0)
paths = ["raw", "oracle", "estimated", "estimated_gated"]
print(f"{'scenario':24s}" + "".join(f"{path:>17s}" for path in paths))
for key, values in decision["f1"].items():
    print(f"{key:24s}" + "".join(f"{values[path]:17.4f}" for path in paths))
print("recovery:", decision["recovery"])
PY
