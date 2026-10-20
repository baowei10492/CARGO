#!/usr/bin/env bash
set -euo pipefail

# P26-A step 1 (docs/P26A_DIFFREGCD_INPUT_ORDER_AMENDMENT.md), an amendment of P26.
# DiffRegCD's per-dataset registration head in the P25 step 1 harness, run
# in the DiffRegCD environment. Validation crops only; no detector runs.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p26a_diffregcd_registration.yaml}"

cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
EXTERNAL_ROOT="$("$PROJECT_PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['external']['root'])")"
EXPECTED_COMMIT="$("$PROJECT_PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['external']['commit'])")"
EXTERNAL_PYTHON="$EXTERNAL_ROOT/.venv/bin/python"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
test -f docs/P26A_DIFFREGCD_INPUT_ORDER_AMENDMENT.md
test -x "$EXTERNAL_PYTHON"
actual_commit="$(git -C "$EXTERNAL_ROOT" rev-parse HEAD)"
test "$actual_commit" = "$EXPECTED_COMMIT" || {
  echo "DiffRegCD commit mismatch: $actual_commit" >&2
  exit 1
}
"$PROJECT_PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
root = Path(config["external"]["root"])
paths = [root / config["external"]["backbone"]["path"]]
paths += [root / config["external"]["registration_heads"][name]["path"] for name in config["datasets"]]
for path in paths:
    if not path.is_file():
        raise SystemExit(f"missing weight {path}")
    print("ok", path, path.stat().st_size)
PY

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$EXTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=disabled

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p26a_diffregcd/step1/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS (project environment) ====="
"$PROJECT_PYTHON" -m pytest -q tests/test_p26_diffregcd_registration.py tests/test_affine_registration.py \
  2>&1 | tee "$report_root/pytest.log"

echo "===== P26-A STEP 1 (DiffRegCD environment) ====="
"$EXTERNAL_PYTHON" -u -m tools.run_p26_diffregcd_registration \
  --config "$CONFIG" \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/step1.log"

cp "$report_root/run/step1_decision.json" "$report_root/step1_decision.json"
sha256sum "$CONFIG" configs/p25_affine_protocol.yaml docs/P26A_DIFFREGCD_INPUT_ORDER_AMENDMENT.md \
  tools/run_p26_diffregcd_registration.py src/rs_cd/external_flow_registration.py src/rs_cd/affine_crops.py \
  src/rs_cd/affine_registration.py "$report_root"/run/*.json "$report_root"/run/*.csv \
  > "$report_root/SHA256SUMS.txt" 2>/dev/null || true
printf '%s\n' "$report_root" > reports/p26a_diffregcd/step1/LATEST

echo "P26-A step 1 complete: $report_root"
"$PROJECT_PYTHON" - "$report_root/step1_decision.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("verdicts:", d["verdicts"])

def fmt(value, digits=3):
    return "None" if value is None else f"{value:.{digits}f}"

for name, entry in d["datasets"].items():
    calibration = entry["calibration"]
    descriptive = entry.get("descriptive_verdict")
    print(f"\n[{name}] verdict {entry['verdict']}" + (f"  (descriptive: {descriptive})" if descriptive else ""))
    print(f"  calibration: selected {calibration['selected']} worst {fmt(calibration['worst_error'], 2)} "
          f"rival {fmt(calibration['best_rival_error'], 2)} passed {calibration['passed']}")
    print("  errors:", {k: (None if v is None else round(v, 2)) for k, v in calibration["errors"].items()})
    if "levels" not in entry:
        continue
    print(f"  crops {entry['crops']} from {entry['tiles']} tiles; clean trigger {fmt(entry['clean_trigger_rate'])}")
    print("  level  success  converged  trigger  perturb_px  ate_raw  ate_diff  sec")
    for level, v in entry["levels"].items():
        print(f"  {level:>5}  {fmt(v.get('success_rate')):>7}  {fmt(v['converged_rate']):>9}  "
              f"{fmt(v['trigger_rate']):>7}  {fmt(v['median_perturbation_px'], 1):>10}  "
              f"{fmt(v['median_ate_raw'], 2):>7}  {fmt(v['median_ate_diff'], 2):>8}  {fmt(v['median_seconds'], 2)}")
    print(f"  clean median displacement px: {fmt(entry['levels']['0'].get('median_clean_displacement_px'), 2)}")
    print("  change bins (crops / top-level success / clean trigger):",
          " | ".join(f"{b['range'][0]:.2f}-{b['range'][1]:.2f}: {b['crops']} / {fmt(b['top_level_success'], 2)} / "
                     f"{fmt(b['clean_trigger_rate'], 2)}" for b in entry["change_bins"]))
PY
