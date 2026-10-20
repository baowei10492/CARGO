#!/usr/bin/env bash
set -euo pipefail

# P25 step 1 (docs/P25_AFFINE_PERTURBATION_PROTOCOL.md section 3).
# Label-free: RoMa + similarity fit against the DiffRegCD affine perturbation
# on validation crops of LEVIR-CD, DSIFN-CD and WHU-CD. No detector runs; no
# test split is read.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p25_affine_protocol.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
test -f docs/P25_AFFINE_PERTURBATION_PROTOCOL.md
python - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for spec in config["datasets"]:
    for key in ("dataset_root", "manifest"):
        if not Path(spec[key]).exists():
            raise SystemExit(f"{spec['name']}: {key} {spec[key]} does not exist")
    print(spec["name"], "ok", spec["dataset_root"], spec["manifest"])
PY

echo "===== LIBRARIES ====="
python - <<'PY'
import importlib.metadata as metadata

import cv2
import local_corr
import romatch
import torch

print("torch", torch.__version__, "cv2", cv2.__version__, "romatch", metadata.version("romatch"),
      "fused-local-corr", metadata.version("fused-local-corr"))
PY

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p25_affine/step1/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_affine_registration.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P25 STEP 1 ====="
python -u -m tools.run_p25_affine_step1 \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/step1.log"

cp "$report_root/run/step1_decision.json" "$report_root/step1_decision.json"
sha256sum "$CONFIG" docs/P25_AFFINE_PERTURBATION_PROTOCOL.md \
  src/rs_cd/affine_registration.py tools/run_p25_affine_step1.py \
  "$report_root"/run/*.json "$report_root"/run/*.csv \
  > "$report_root/SHA256SUMS.txt" 2>/dev/null || true
printf '%s\n' "$report_root" > reports/p25_affine/step1/LATEST

echo "P25 step 1 complete: $report_root"
python - "$report_root/step1_decision.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("verdicts:", d["verdicts"])

def fmt(value, digits=3):
    return "None" if value is None else f"{value:.{digits}f}"

for name, entry in d["datasets"].items():
    print(f"\n[{name}] verdict {entry['verdict']}  selfcheck {entry['selfcheck']}")
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
