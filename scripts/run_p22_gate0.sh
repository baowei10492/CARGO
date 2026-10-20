#!/usr/bin/env bash
set -euo pipefail

# P22 Gate 0 (docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md section 2).
# Label-free, train-raster blocks only, frozen LEVIR localizer. Reads no
# validation tile, no label, and never opens the WHU test raster.

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
test -f docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p22_whu/gate0/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_physical_shift.py 2>&1 | tee "$report_root/pytest.log"

echo "===== GATE 0: COARSE LOCALIZATION OF NATIVE SHIFTS ====="
python -u -m tools.run_p22_whu_physical_shift gate0 \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/gate0.log"

cp "$report_root/run/gate0_decision.json" "$report_root/gate0_decision.json"
sha256sum "$CONFIG" docs/P22_WHU_PHYSICAL_SHIFT_PROTOCOL.md \
  "$report_root/run/gate0_offsets.csv" "$report_root/gate0_decision.json" \
  > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p22_whu/gate0/LATEST

echo "P22 Gate 0 complete: $report_root"
python - "$report_root/gate0_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
print("verdict:", decision["verdict"], " clean_passed:", decision["clean_passed"])
for key, entry in decision["scenarios"].items():
    print(
        f"{key:24s} n={entry['n']} native_mean_epe={entry['native_mean_epe']:.3f} "
        f"coarse_pck1={entry.get('coarse_pck1')} native_pck1={entry['native_pck1']} "
        f"q={entry['q_trigger_rate']} clean_gated_1c={entry.get('gated_within_1_coarse_px')}"
    )
PY
