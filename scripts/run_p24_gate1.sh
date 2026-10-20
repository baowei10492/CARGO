#!/usr/bin/env bash
set -euo pipefail

# P24 Gate 1 (docs/P24_WHU_ROMA_GATE1_PROTOCOL.md).
# RoMa named by the protocol after the P23 screen stopped. Validation labels,
# frozen P8 detector, P7C paths reproduce P22 Gate 1 first. One run only.
# Never opens the WHU test raster.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p24_whu_roma_gate1.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
test -f docs/P24_WHU_ROMA_GATE1_PROTOCOL.md
screen_run="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['p23_screen_run'])")"
screen_root="reports/p23_whu/screen/$screen_run"
p22_gate1="$(cat reports/p22_whu/gate1/LATEST)"
test -f "$screen_root/screen_decision.json"
test -f "$p22_gate1/gate1_decision.json"
if ls reports/p24_whu/gate1/*/gate1_decision.json >/dev/null 2>&1; then
  echo "a P24 Gate 1 decision already exists; the protocol allows one run" >&2
  exit 1
fi

echo "===== LIBRARIES ====="
python - <<'PY'
import importlib.metadata as metadata

import local_corr
import romatch
import torch

print("torch", torch.__version__, "romatch", metadata.version("romatch"),
      "fused-local-corr", metadata.version("fused-local-corr"))
PY

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p24_whu/gate1/$tag"
mkdir -p "$report_root"
printf '%s\n%s\n' "$screen_root" "$p22_gate1" > "$report_root/REFERENCES"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_strong_localizers.py tests/test_physical_shift.py \
  tests/test_p24_gate1.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P24 GATE 1 ====="
python -u -m tools.run_p23_whu_strong_localizer p24-gate1 \
  --config "$CONFIG" \
  --device cuda \
  --screen-dir "$screen_root" \
  --p22-gate1-dir "$p22_gate1" \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/gate1.log"

cp "$report_root/run/gate1_decision.json" "$report_root/gate1_decision.json"
sha256sum "$CONFIG" configs/p23_whu_strong_localizer.yaml docs/P24_WHU_ROMA_GATE1_PROTOCOL.md \
  src/rs_cd/strong_localizers.py tools/run_p23_whu_strong_localizer.py \
  "$report_root"/run/*.json "$report_root"/run/*.csv \
  > "$report_root/SHA256SUMS.txt" 2>/dev/null || true
printf '%s\n' "$report_root" > reports/p24_whu/gate1/LATEST

echo "P24 Gate 1 complete: $report_root"
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
print("recovery (RoMa estimated):", {k: None if v is None else round(v, 3) for k, v in d["recovery"].items()})
print("clean change vs raw:", {k: round(v, 4) for k, v in d["clean_change"].items()})
print("\n[row bands] tiles / RoMa recovery B1,B2,B3 / P7C recovery B1,B2,B3 / clean est, gated vs raw")
for label, band in d["row_groups"].items():
    rec = [None if v is None else round(v, 3) for v in band["recovery"].values()]
    p7c = [None if v is None else round(v, 3) for v in band["p7c_recovery"].values()]
    clean = band["f1"]["clean:s0:common"]
    print(f"  {label}: {band['tiles']} / {rec} / {p7c} / "
          f"{clean['estimated'] - clean['raw']:+.4f}, {clean['estimated_gated'] - clean['raw']:+.4f}")
PY
