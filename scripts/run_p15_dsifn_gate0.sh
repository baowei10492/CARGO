#!/usr/bin/env bash
set -euo pipefail

# P15 Gate 0: label-free applicability screening for DSIFN-CD.
# Runs the frozen LEVIR registration stack zero-shot on DSIFN and on a
# geometry-matched LEVIR control. Reads no change label, trains nothing,
# and never touches any test split.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "${DSIFN_ROOT:-}" && -f reports/p15_dsifn_gate_minus1/DATASET_ROOT ]]; then
  DSIFN_ROOT="$(cat reports/p15_dsifn_gate_minus1/DATASET_ROOT)"
fi
DSIFN_ROOT="${DSIFN_ROOT:-/root/autodl-tmp/rs_cd_work/data/DSIFN-CD}"
# Prefer the leakage-free source-crop split once it exists: Gate -1 found
# augmented copies of the same crop in both official splits, so the
# official validation set must not be used to calibrate tau.
if [[ -z "${DSIFN_MANIFEST:-}" ]]; then
  if [[ -f manifests/datasets/dsifn_cd_source_split.csv ]]; then
    DSIFN_MANIFEST="manifests/datasets/dsifn_cd_source_split.csv"
  else
    DSIFN_MANIFEST="manifests/datasets/dsifn_cd.csv"
  fi
fi
echo "Using DSIFN manifest: ${DSIFN_MANIFEST}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
LEVIR_MANIFEST="${LEVIR_MANIFEST:-manifests/datasets/levir_cd.csv}"
CONFIG="${CONFIG:-configs/p15_dsifn_screening.yaml}"

for required in "${DSIFN_MANIFEST}" "${LEVIR_MANIFEST}" "${CONFIG}"; do
  test -f "${required}" || {
    echo "Missing Gate 0 input: ${required}" >&2
    echo "Run scripts/run_p15_dsifn_gate_minus1.sh first." >&2
    exit 1
  }
done

freeze_root="$(cat reports/p11_structure_freeze/LATEST)"
test -f "${freeze_root}/STRUCTURE_FROZEN"
P7_CHECKPOINT="$(
  python - "${freeze_root}/freeze_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["sources"]["p7_checkpoint"])
PY
)"
test -f "${P7_CHECKPOINT}"
echo "Frozen P7 checkpoint: ${P7_CHECKPOINT}"

TILE_SIZE="$(python -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG}',encoding='utf-8'))['geometry']['tile_size'])")"
CONTROL_CROP="$(python -c "import yaml,sys;print(yaml.safe_load(open('${CONFIG}',encoding='utf-8'))['geometry']['control_center_crop'])")"
SEVERITIES="$(python -c "import yaml;print(' '.join(str(v) for v in yaml.safe_load(open('${CONFIG}',encoding='utf-8'))['geometry']['severities']))")"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p15_dsifn_gate0/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p15_dsifn_gates.py

echo "===== CANDIDATE: DSIFN val (native 512, tile ${TILE_SIZE}) ====="
python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "${P7_CHECKPOINT}" \
  --dataset-root "${DSIFN_ROOT}" \
  --manifest "${DSIFN_MANIFEST}" \
  --split val \
  --label dsifn_val_native512 \
  --tile-size "${TILE_SIZE}" \
  --severities ${SEVERITIES} \
  --output-dir "${report_root}/candidate_dsifn" \
  2>&1 | tee "${report_root}/candidate.log"

echo "===== CONTROL: LEVIR val centre-cropped to ${CONTROL_CROP} ====="
python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "${P7_CHECKPOINT}" \
  --dataset-root "${LEVIR_ROOT}" \
  --manifest "${LEVIR_MANIFEST}" \
  --split val \
  --label levir_val_matched512 \
  --tile-size "${TILE_SIZE}" \
  --center-crop "${CONTROL_CROP}" \
  --severities ${SEVERITIES} \
  --output-dir "${report_root}/control_levir" \
  2>&1 | tee "${report_root}/control.log"

echo "===== GATE 0 ROUTING ====="
python -m tools.summarize_p15_gate0 \
  --candidate-hygiene "${report_root}/candidate_dsifn/hygiene_summary.json" \
  --control-hygiene "${report_root}/control_levir/hygiene_summary.json" \
  --config "${CONFIG}" \
  --output-json "${report_root}/gate0_decision.json"

printf '%s\n' "${report_root}" > reports/p15_dsifn_gate0/LATEST

sha256sum \
  "${report_root}/candidate_dsifn/hygiene_summary.json" \
  "${report_root}/control_levir/hygiene_summary.json" \
  "${report_root}/gate0_decision.json" \
  > "${report_root}/SHA256SUMS.txt"

echo "Gate 0 complete: ${report_root}"
python - "${report_root}/gate0_decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
signals = decision["signals"]
print("branch :", decision["branch"])
print("action :", decision["action"])
print(
    "dispersion: candidate {:.3f} control {:.3f} excess {:+.3f}".format(
        signals["mean_injected_dispersion"]["candidate"],
        signals["mean_injected_dispersion"]["control"],
        signals["mean_injected_dispersion"]["excess"],
    )
)
print(
    "clean q   : candidate {:.4f} control {:.4f}".format(
        signals["clean_q_trigger_rate"]["candidate"],
        signals["clean_q_trigger_rate"]["control"],
    )
)
PY
