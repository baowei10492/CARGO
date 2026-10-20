#!/usr/bin/env bash
set -euo pipefail

# P23 localizer screen (docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md section 2).
# Label-free verdict: self-check, P7C offsets anchor against P22, then every
# candidate on the P22 train blocks and validation tiles. Never opens the WHU
# test raster.

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
test -f docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md

p22_gate0="$(cat reports/p22_whu/gate0/LATEST)"
p22_gate1="$(cat reports/p22_whu/gate1/LATEST)"
test -f "$p22_gate0/run/gate0_offsets.csv"
test -f "$p22_gate1/run/gate1_offsets.csv"
python - "$p22_gate1/gate1_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
if decision.get("verdict") != "recovery_insufficient" or "f1" not in decision:
    raise SystemExit(f"unexpected P22 Gate 1 reference: {decision.get('verdict')}")
PY

echo "===== LIBRARIES ====="
python - <<'PY'
import importlib.metadata as metadata

import cv2
import kornia
import local_corr
import romatch
import torch

print("torch", torch.__version__, "kornia", kornia.__version__, "cv2", cv2.__version__,
      "romatch", metadata.version("romatch"),
      "fused-local-corr", metadata.version("fused-local-corr"))
PY

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p23_whu/screen/$tag"
mkdir -p "$report_root"
printf '%s\n%s\n' "$p22_gate0" "$p22_gate1" > "$report_root/P22_REFERENCES"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_strong_localizers.py tests/test_physical_shift.py 2>&1 | tee "$report_root/pytest.log"

echo "===== P23 SCREEN ====="
python -u -m tools.run_p23_whu_strong_localizer screen \
  --config "$CONFIG" \
  --device cuda \
  --p22-gate0-dir "$p22_gate0" \
  --p22-gate1-dir "$p22_gate1" \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/screen.log"

cp "$report_root/run/screen_decision.json" "$report_root/screen_decision.json"
sha256sum "$CONFIG" docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md \
  src/rs_cd/strong_localizers.py tools/run_p23_whu_strong_localizer.py \
  "$report_root"/run/*.json "$report_root"/run/*.csv \
  > "$report_root/SHA256SUMS.txt" 2>/dev/null || true
printf '%s\n' "$report_root" > reports/p23_whu/screen/LATEST

echo "P23 screen complete: $report_root"
python - "$report_root/screen_decision.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("verdict:", d["verdict"], " selected:", d.get("selected"))
if "candidates" not in d:
    print(json.dumps({k: d.get(k) for k in ("selfcheck", "anchor")}, indent=2))
    raise SystemExit(0)
print("anchor passed:", d["anchor"]["passed"], " selfcheck passed:", d["selfcheck"]["passed"])
keys = ["clean:s0:common"] + [f"shift:s{s}:t2" for s in (1, 2, 3)] + [f"physical_shift:s{s}:t2" for s in (1, 2, 3)]
short = {"clean:s0:common": "clean", **{f"shift:s{s}:t2": f"A{s}" for s in (1, 2, 3)},
         **{f"physical_shift:s{s}:t2": f"B{s}" for s in (1, 2, 3)}}
for set_name in ("val_tiles", "train_blocks"):
    print(f"\n[{set_name}] per scenario: pck2 / mean_epe / trigger / nonconv / sec")
    for name, entry in d["candidates"].items():
        cells = []
        for key in keys:
            m = entry[set_name][key]
            cells.append(f"{short[key]} {m['native_pck2']:.2f}/{m['native_mean_epe']:.1f}/"
                         f"{m['trigger_rate']:.2f}/{m['nonconverged_rate']:.2f}/{m['median_seconds']:.2f}")
        print(f"  {name:18s} " + "  ".join(cells))
print("\n[val verdicts] tracks / mean B pck2 / clean trigger")
for name, v in d["val_verdicts"].items():
    print(f"  {name:18s} {v['tracks']!s:5s} {v['mean_physical_pck2']:.3f} {v['clean_trigger_rate']:.3f}")
ca = d["clean_agreement"]
print(f"\nclean agreement among {ca['candidates']}: {ca['agreeing_nonzero']}/{ca['clean_pairs']}")
print("\n[val change bins] tiles / B3 pck2 / clean trigger")
for name, entry in d["candidates"].items():
    cells = [f"{b['range'][0]:.2f}-{b['range'][1]:.2f}: {b['tiles']} / {b['physical_s3_pck2']} / {b['clean_trigger_rate']}"
             for b in entry["change_bins"]]
    print(f"  {name:18s} " + " | ".join(cells))
PY
