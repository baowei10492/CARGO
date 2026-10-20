#!/usr/bin/env bash
set -euo pipefail

# P23 Gate 1 (docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md section 3).
# Validation labels, frozen P8 detector, P7C and the one candidate the screen
# selected. The P7C paths must reproduce the P22 Gate 1 table first. Never
# opens the WHU test raster.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p23_whu_strong_localizer.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
screen_root="$(cat reports/p23_whu/screen/LATEST)"
p22_gate1="$(cat reports/p22_whu/gate1/LATEST)"
python - "$screen_root/screen_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
if decision["verdict"] != "proceed_to_gate1":
    raise SystemExit(f"screen does not authorise Gate 1: {decision['verdict']}")
print("selected:", decision["selected"])
PY
if ls reports/p23_whu/gate1/*/gate1_decision.json >/dev/null 2>&1; then
  echo "a P23 Gate 1 decision already exists; the protocol allows one run" >&2
  exit 1
fi

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p23_whu/gate1/$tag"
mkdir -p "$report_root"
printf '%s\n%s\n' "$screen_root" "$p22_gate1" > "$report_root/REFERENCES"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_strong_localizers.py tests/test_physical_shift.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P23 GATE 1 ====="
python -u -m tools.run_p23_whu_strong_localizer gate1 \
  --config "$CONFIG" \
  --device cuda \
  --screen-dir "$screen_root" \
  --p22-gate1-dir "$p22_gate1" \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/gate1.log"

cp "$report_root/run/gate1_decision.json" "$report_root/gate1_decision.json"
sha256sum "$CONFIG" docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md \
  src/rs_cd/strong_localizers.py tools/run_p23_whu_strong_localizer.py \
  "$report_root"/run/*.json "$report_root"/run/*.csv \
  > "$report_root/SHA256SUMS.txt" 2>/dev/null || true
printf '%s\n' "$report_root" > reports/p23_whu/gate1/LATEST

echo "P23 Gate 1 complete: $report_root"
python - "$report_root/gate1_decision.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("verdict:", d["verdict"], " selected:", d["selected"])
print("anchor passed:", d["anchor"]["passed"], " max |delta|:", d["anchor"]["max_abs_delta"])
if "f1" not in d:
    raise SystemExit(0)
paths = ["raw", "oracle", "p7c_estimated", "p7c_gated", "estimated", "estimated_gated"]
print(f"{'scenario':24s}" + "".join(f"{p:>16s}" for p in paths))
for key, values in d["f1"].items():
    print(f"{key:24s}" + "".join(f"{values[p]:16.4f}" for p in paths))
print("recovery:", d["recovery"])
print("clean change vs raw:", {k: round(v, 4) for k, v in d["clean_change"].items()})
PY
