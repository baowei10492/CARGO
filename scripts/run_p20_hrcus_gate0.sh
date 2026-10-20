#!/usr/bin/env bash
set -euo pipefail

# P20 Gate 0: label-free applicability screening for HRCUS-CD.
# Runs the frozen LEVIR registration stack zero-shot on HRCUS-CD and on
# two LEVIR controls. Reads no change label, trains nothing, and never
# touches any test split.
#
# Why two controls. HRCUS-CD ships as 256 px tiles and cannot be re-tiled
# larger, so its geometry differs from every dataset screened so far. If
# the candidate scores badly we must be able to say whether that is the
# imagery or the window size. The reference control also reproduces a
# published LEVIR number and is treated as an instrument check: if it
# fails, nothing downstream is read.
#
#   reference : LEVIR centre-crop 512, tile 256  -> must reproduce clean q 0.0625
#   matched   : LEVIR centre-crop 256, tile 128  -> same geometry as HRCUS
#   candidate : HRCUS-CD val, native 256, tile 128

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "${HRCUS_ROOT:-}" && -f reports/p20_hrcus_gate_minus1/DATASET_ROOT ]]; then
  HRCUS_ROOT="$(cat reports/p20_hrcus_gate_minus1/DATASET_ROOT)"
fi
HRCUS_ROOT="${HRCUS_ROOT:-/root/autodl-tmp/rs_cd_work/data/HRCUS-CD}"
HRCUS_MANIFEST="${HRCUS_MANIFEST:-manifests/datasets/hrcus_cd.csv}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
LEVIR_MANIFEST="${LEVIR_MANIFEST:-manifests/datasets/levir_cd.csv}"
CONFIG="${CONFIG:-configs/p20_hrcus_screening.yaml}"

for required in "${HRCUS_MANIFEST}" "${LEVIR_MANIFEST}" "${CONFIG}"; do
  test -f "${required}" || {
    echo "Missing Gate 0 input: ${required}" >&2
    echo "Run scripts/run_p20_hrcus_gate_minus1.sh first." >&2
    exit 1
  }
done

# Gate -1 must have passed. A leaking official split cannot be screened.
decision="$(cat reports/p20_hrcus_gate_minus1/LATEST)/gate_minus1_decision.json"
test -f "${decision}" || {
  echo "Missing ${decision}; run Gate -1 first." >&2
  exit 1
}
python - "${decision}" <<'PY'
import json
import sys

verdict = json.load(open(sys.argv[1], encoding="utf-8"))["verdict"]
if verdict not in ("passed", "passed_after_recorded_cleaning"):
    raise SystemExit(
        f"Gate -1 verdict is {verdict!r}. Resolve the split before screening."
    )
print(f"Gate -1 verdict: {verdict}")
PY

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

read_cfg() {
  python -c "import yaml;print(yaml.safe_load(open('${CONFIG}',encoding='utf-8'))${1})"
}
TILE_SIZE="$(read_cfg "['geometry']['tile_size']")"
CONTROL_CROP="$(read_cfg "['geometry']['control_center_crop']")"
REF_CROP="$(read_cfg "['geometry']['reference_center_crop']")"
REF_TILE="$(read_cfg "['geometry']['reference_tile_size']")"
SEVERITIES="$(python -c "import yaml;print(' '.join(str(v) for v in yaml.safe_load(open('${CONFIG}',encoding='utf-8'))['geometry']['severities']))")"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p20_hrcus_gate0/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p15_dsifn_gates.py

echo "===== REFERENCE: LEVIR val centre-crop ${REF_CROP}, tile ${REF_TILE} ====="
python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "${P7_CHECKPOINT}" \
  --dataset-root "${LEVIR_ROOT}" \
  --manifest "${LEVIR_MANIFEST}" \
  --split val \
  --label levir_val_reference512 \
  --tile-size "${REF_TILE}" \
  --center-crop "${REF_CROP}" \
  --severities ${SEVERITIES} \
  --output-dir "${report_root}/reference_levir" \
  2>&1 | tee "${report_root}/reference.log"

echo "===== INSTRUMENT CHECK BEFORE READING ANYTHING ELSE ====="
python - "${report_root}/reference_levir/hygiene_summary.json" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

# Use the summarizer's own accessor rather than reimplementing the path
# into the summary, so this check cannot drift from the gate itself.
from tools.summarize_p15_gate0 import clean_q_trigger_rate

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
geometry = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))["geometry"]
expected = geometry["reference_clean_q_expected"]
tolerance = geometry["reference_clean_q_tolerance"]
measured = float(clean_q_trigger_rate(summary))
print(f"LEVIR reference clean q: {measured:.4f} (published {expected:.4f})")
if abs(measured - expected) > tolerance:
    raise SystemExit(
        f"Reference control does not reproduce the published LEVIR clean q "
        f"({measured:.4f} vs {expected:.4f}, tolerance {tolerance}). The "
        "instrument is not in a known state; do not read the candidate."
    )
print("instrument check passed")
PY

echo "===== MATCHED CONTROL: LEVIR val centre-crop ${CONTROL_CROP}, tile ${TILE_SIZE} ====="
python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "${P7_CHECKPOINT}" \
  --dataset-root "${LEVIR_ROOT}" \
  --manifest "${LEVIR_MANIFEST}" \
  --split val \
  --label levir_val_matched256 \
  --tile-size "${TILE_SIZE}" \
  --center-crop "${CONTROL_CROP}" \
  --severities ${SEVERITIES} \
  --output-dir "${report_root}/control_levir" \
  2>&1 | tee "${report_root}/control.log"

echo "===== CANDIDATE: HRCUS-CD val (native 256, tile ${TILE_SIZE}) ====="
python -u -m tools.diagnose_registration_hygiene \
  --checkpoint "${P7_CHECKPOINT}" \
  --dataset-root "${HRCUS_ROOT}" \
  --manifest "${HRCUS_MANIFEST}" \
  --split val \
  --label hrcus_val_native256 \
  --tile-size "${TILE_SIZE}" \
  --severities ${SEVERITIES} \
  --output-dir "${report_root}/candidate_hrcus" \
  2>&1 | tee "${report_root}/candidate.log"

echo "===== GATE 0 ROUTING ====="
python -m tools.summarize_p15_gate0 \
  --candidate-hygiene "${report_root}/candidate_hrcus/hygiene_summary.json" \
  --control-hygiene "${report_root}/control_levir/hygiene_summary.json" \
  --config "${CONFIG}" \
  --output-json "${report_root}/gate0_decision.json"

echo "===== WINDOW-SIZE ATTRIBUTION ====="
# Preregistered reading, written before the run: separate "HRCUS imagery
# is hard" from "a 256 px window is too small for this localizer".
python - "${report_root}" "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from tools.summarize_p15_gate0 import clean_q_trigger_rate

root = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
green_max = config["gate0"]["clean_q_green_max"]


def load(name):
    return json.loads(
        (root / name / "hygiene_summary.json").read_text(encoding="utf-8")
    )


reference = load("reference_levir")
control = load("control_levir")
candidate = load("candidate_hrcus")

rq = float(clean_q_trigger_rate(reference))
cq = float(clean_q_trigger_rate(control))
hq = float(clean_q_trigger_rate(candidate))

print(f"clean q  LEVIR@512/256 {rq:.4f} -> LEVIR@256/128 {cq:.4f} "
      f"-> HRCUS@256/128 {hq:.4f}")

window_cost = cq - rq
candidate_excess = hq - cq

# Order matters and the first version of this block had it wrong. The
# calibration-range caveat below fires whenever the matched control
# crosses the green threshold, which it can do by a hair, and as an
# elif-chain head it preempted a far stronger attribution signal. The
# magnitude comparison comes first; the caveat is an annotation on top
# of it, never a substitute for it.
if candidate_excess > window_cost:
    attribution = "candidate_dominates"
    note = (
        f"The candidate's excess over the matched control ({candidate_excess:+.4f}) "
        f"exceeds what the window itself costs ({window_cost:+.4f}), so the "
        "degradation is mostly a property of the candidate imagery, not of "
        "the analysis window."
    )
elif candidate_excess > 0.5 * window_cost:
    attribution = "mixed"
    note = "Window size and imagery contribute comparably; report both."
else:
    attribution = "window_size_dominates"
    note = (
        "The window costs more than the candidate's excess over it, so the "
        "candidate number cannot be attributed to its imagery."
    )

if cq > green_max:
    note += (
        f" Caveat: the matched control ({cq:.4f}) is itself above the green "
        f"threshold ({green_max}), so the absolute clean q thresholds have "
        "left the scale they were calibrated on. This qualifies how the "
        "absolute number should be read; it does not by itself overturn the "
        "attribution above, and it must not be used to reread a branch that "
        "is red by a wide margin on both signals."
    )

print(f"window cost {window_cost:+.4f}, candidate excess {candidate_excess:+.4f}")
print(f"attribution: {attribution}")
print(note)

(root / "window_attribution.json").write_text(
    json.dumps(
        {
            "status": "completed",
            "preregistered": True,
            "clean_q": {
                "levir_reference_512_256": rq,
                "levir_matched_256_128": cq,
                "hrcus_candidate_256_128": hq,
            },
            "window_cost": window_cost,
            "candidate_excess_over_matched_control": candidate_excess,
            "attribution": attribution,
            "note": note,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "${report_root}" > reports/p20_hrcus_gate0/LATEST

sha256sum \
  "${report_root}/reference_levir/hygiene_summary.json" \
  "${report_root}/control_levir/hygiene_summary.json" \
  "${report_root}/candidate_hrcus/hygiene_summary.json" \
  "${report_root}/gate0_decision.json" \
  "${report_root}/window_attribution.json" \
  > "${report_root}/SHA256SUMS.txt"

echo "P20 Gate 0 complete: ${report_root}"
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
