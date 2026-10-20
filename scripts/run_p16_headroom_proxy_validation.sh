#!/usr/bin/env bash
set -euo pipefail

# P16 step 1: does label morphology alone predict recoverable headroom?
#
# Labels only. No model is loaded, nothing is trained, no test split is
# read. The measured headroom values are quoted from runs that already
# happened; this script recomputes the label side and checks whether the
# cheap signal orders datasets the same way the expensive one does.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/rs_cd_work/data}"
LEVIR_ROOT="${LEVIR_ROOT:-${DATA_ROOT}/LEVIR-CD}"
WHU_ROOT="${WHU_ROOT:-${DATA_ROOT}/WHU-CD}"
S2_ROOT="${S2_ROOT:-${DATA_ROOT}/S2Looking}"
DSIFN_ROOT="${DSIFN_ROOT:-${DATA_ROOT}/DSIFN-CD/DSIFN}"
DSIFN_MANIFEST="${DSIFN_MANIFEST:-manifests/datasets/dsifn_cd_source_split.csv}"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p16_headroom_proxy/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_threshold_sensitivity.py

run_audit() {
  local name="$1" root="$2" manifest="$3" threshold="$4"
  if [[ ! -f "${manifest}" ]]; then
    echo "SKIP ${name}: manifest not found (${manifest})" >&2
    return 0
  fi
  echo "===== MORPHOLOGY: ${name} ====="
  python -m tools.audit_change_target_scale \
    --dataset-name "${name}" \
    --dataset-root "${root}" \
    --manifest "${manifest}" \
    --split val \
    --mask-threshold "${threshold}" \
    --output "${report_root}/morphology_${name}.json"
  # An all-zero audit means the threshold does not match the label
  # encoding. Fail here instead of propagating empty statistics.
  python - "${report_root}/morphology_${name}.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(audit["positive_samples"]) == 0 or audit["change_fraction"] == 0.0:
    raise SystemExit(
        f"{audit['dataset']}: no change pixels found at mask_threshold "
        f"{audit['mask_threshold']} while label values are "
        f"{audit['pixel_values']}. The threshold does not match the "
        "label encoding."
    )
PY
}

# Mask encodings differ by dataset and the threshold must match the one
# each frozen pipeline used: LEVIR/WHU/S2Looking store 0/255 labels,
# DSIFN stores 0/1. Using 127 on DSIFN silently zeroes every change
# pixel, which is the failure recorded in main record 113.
run_audit LEVIR-CD "${LEVIR_ROOT}" manifests/datasets/levir_cd.csv 127
run_audit WHU-CD "${WHU_ROOT}" manifests/datasets/whu_cd.csv 127
run_audit S2Looking "${S2_ROOT}" manifests/datasets/s2looking.csv 127
run_audit DSIFN-CD "${DSIFN_ROOT}" "${DSIFN_MANIFEST}" 0

echo "===== ASSEMBLE PROXY VERSUS MEASURED HEADROOM ====="
python - "${report_root}" <<'PY'
import json
import sys
from pathlib import Path

# Measured Shift-8 input-oracle headroom from the runs already recorded.
# These are quoted, not recomputed; the label side is what this step
# measures.
MEASURED = {
    "LEVIR-CD": 0.2141,
    "WHU-CD": 0.10880,
    "S2Looking": 0.0330,
    "DSIFN-CD": 0.0227,
}

root = Path(sys.argv[1])
records = []
for name, headroom in MEASURED.items():
    path = root / f"morphology_{name}.json"
    if not path.is_file():
        print(f"missing morphology for {name}, skipping")
        continue
    audit = json.loads(path.read_text(encoding="utf-8"))
    records.append(
        {
            "dataset": name,
            "mask_self_iou_shift8": float(
                audit["mask_self_iou_on_paired_support"]["8"]
            ),
            "boundary_fraction_shift8": float(
                audit["boundary_band_fraction_of_change"]["8"]
            ),
            "change_fraction": float(audit["change_fraction"]),
            "area_weighted_equivalent_diameter": float(
                audit["components"]["area_weighted_equivalent_diameter"][
                    "median"
                ]
            ),
            "measured_shift8_headroom": headroom,
        }
    )
(root / "records.json").write_text(
    json.dumps(records, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(f"assembled {len(records)} datasets")
PY

python -m tools.validate_headroom_proxy \
  --records "${report_root}/records.json" \
  --output-json "${report_root}/proxy_validation.json"

printf '%s\n' "${report_root}" > reports/p16_headroom_proxy/LATEST
sha256sum "${report_root}"/*.json > "${report_root}/SHA256SUMS.txt"
echo "P16 proxy validation complete: ${report_root}"
