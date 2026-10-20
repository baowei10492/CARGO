#!/usr/bin/env bash
set -euo pipefail

# P22 diagnostics (docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md).
# A1: train-region change-density dose-response, frozen LEVIR localizer,
#     train labels used only to stratify windows. Validation strip and test
#     raster are never read.
# A2: cross-dataset association on existing P21 per-sample decisions; no
#     model is run.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p22_change_density_probe.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
test -f docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p22_whu/change_density/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_change_density.py tests/test_physical_shift.py 2>&1 | tee "$report_root/pytest.log"

echo "===== A2: CROSS-DATASET ASSOCIATION (NO MODEL) ====="
python -u -m tools.summarize_localizer_change_density \
  --config "$CONFIG" \
  --output "$report_root/a2_decision.json" \
  2>&1 | tee "$report_root/a2.log"

echo "===== A1: TRAIN-REGION DOSE-RESPONSE ====="
python -u -m tools.run_p22_change_density_probe \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/a1" \
  2>&1 | tee "$report_root/a1.log"

sha256sum "$CONFIG" docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md \
  "$report_root/a2_decision.json" "$report_root/a1/probe_decision.json" \
  > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p22_whu/change_density/LATEST

echo "P22 change-density diagnostics complete: $report_root"
python - "$report_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
a2 = json.loads((root / "a2_decision.json").read_text(encoding="utf-8"))
print("A2 overall:", a2["overall"])
for name, cell in a2["cells"].items():
    s = cell["association_s3"]
    print(f"  {name:9s} n={cell['n']} rho={s['spearman']} low_n={s['low_bin_n']} high_n={s['high_bin_n']} "
          f"low_epe={s['low_bin_mean_error']} high_epe={s['high_bin_mean_error']} verdict={cell['verdict']}")
a1 = json.loads((root / "a1" / "probe_decision.json").read_text(encoding="utf-8"))
print("A1 verdict:", a1["verdict"], " merged_high_bin:", a1["decision"]["merged_high_bin"])
print("  candidates per bin:", a1["candidates_per_bin"])
for b in a1["bins"]:
    print(f"  bin {b['range']} n={b['n']} S2 PCK1={b['s2_pck1']} S3 PCK1={b['s3_pck1']} clean_q={b['clean_q']} "
          f"S2 err={b['mean_s2_error']} S3 err={b['mean_s3_error']}")
print("  spearman:", a1["spearman"])
PY
