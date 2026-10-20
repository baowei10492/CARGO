#!/usr/bin/env bash
set -euo pipefail

# P15 Gate -1: DSIFN-CD data hygiene audit.
# Reads images only. Trains nothing, evaluates nothing, and computes no
# performance metric on any split.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/rs_cd_work/data/DSIFN-CD}"
MANIFEST="${MANIFEST:-manifests/datasets/dsifn_cd.csv}"

test -d "${DATASET_ROOT}" || {
  echo "Dataset root not found: ${DATASET_ROOT}" >&2
  exit 1
}

echo "===== DISCOVERED LAYOUT ====="
find "${DATASET_ROOT}" -maxdepth 3 -type d | sort | head -40
echo

# Resolve the split root and the T1/T2/mask folder names. Public DSIFN
# mirrors use t1/t2/mask, A/B/label or T1/T2/GT, sometimes one wrapper
# directory deep, so detect instead of guessing.
layout="$(
  DATASET_ROOT="${DATASET_ROOT}" \
  T1_DIR="${T1_DIR:-}" T2_DIR="${T2_DIR:-}" MASK_DIR="${MASK_DIR:-}" \
  python - <<'PY'
import os
import sys
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"]).expanduser().resolve()
splits = ("train", "val", "test")
aliases = {
    "t1": ("t1", "a", "im1", "image1", "before", "t1_images"),
    "t2": ("t2", "b", "im2", "image2", "after", "t2_images"),
    "mask": ("mask", "label", "labels", "gt", "masks", "out", "change"),
}
override = {
    role: os.environ.get(f"{role.upper()}_DIR", "").strip()
    for role in aliases
}


def split_root(candidate: Path) -> Path | None:
    if all((candidate / name).is_dir() for name in splits):
        return candidate
    return None


found = split_root(root)
if found is None:
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        found = split_root(child)
        if found is not None:
            break
if found is None:
    print("UNRESOLVED", file=sys.stderr)
    raise SystemExit(2)

names = {p.name for p in (found / "train").iterdir() if p.is_dir()}
lowered = {name.lower(): name for name in names}
resolved = {}
for role, options in aliases.items():
    if override[role]:
        resolved[role] = override[role]
        continue
    for option in options:
        if option in lowered:
            resolved[role] = lowered[option]
            break
    else:
        print(f"UNRESOLVED role={role} candidates={sorted(names)}",
              file=sys.stderr)
        raise SystemExit(3)

print(f"{found}|{resolved['t1']}|{resolved['t2']}|{resolved['mask']}")
PY
)" || {
  echo >&2
  echo "Could not resolve the DSIFN layout under ${DATASET_ROOT}." >&2
  echo "Expected train/ val/ test/ each holding T1, T2 and mask folders." >&2
  echo "If the archive is not extracted yet, extract it first; otherwise" >&2
  echo "set them explicitly, e.g.:" >&2
  echo "  T1_DIR=t1 T2_DIR=t2 MASK_DIR=mask bash $0" >&2
  exit 2
}

DATASET_ROOT="${layout%%|*}"
rest="${layout#*|}"
T1_DIR="${rest%%|*}"
rest="${rest#*|}"
T2_DIR="${rest%%|*}"
MASK_DIR="${rest##*|}"

echo "Resolved dataset root: ${DATASET_ROOT}"
echo "Resolved folders: T1=${T1_DIR} T2=${T2_DIR} MASK=${MASK_DIR}"
for split in train val test; do
  count="$(find "${DATASET_ROOT}/${split}/${T1_DIR}" -type f | wc -l)"
  echo "  ${split}: ${count} T1 files"
done
echo

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p15_dsifn_gate_minus1/${tag}"
mkdir -p "${report_root}"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p15_dsifn_gates.py tests/test_dataset.py

echo "===== BUILD MANIFEST ====="
# split_stem is mandatory here: DSIFN restarts filenames inside every
# split, so bare stems would give three different pairs the same id.
python -m tools.build_manifest \
  --dataset-root "${DATASET_ROOT}" \
  --output "${MANIFEST}" \
  --splits train val test \
  --t1-dir "${T1_DIR}" \
  --t2-dir "${T2_DIR}" \
  --mask-dir "${MASK_DIR}" \
  --sample-id-mode split_stem \
  2>&1 | tee "${report_root}/build_manifest.log"

echo "===== DATASET AUDIT (A1/A2) ====="
python -m tools.audit_dataset \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-json "${report_root}/data_audit_DSIFN-CD.json" \
  --output-md "${report_root}/data_audit_DSIFN-CD.md" \
  2>&1 | tee "${report_root}/audit.log"

echo "===== AUGMENTATION LEAKAGE AUDIT (A3) ====="
# Not --fail-on-leakage: leakage is an expected finding for DSIFN and
# routes to a re-split, it is not a reason to abort the audit.
python -m tools.audit_augmentation_leakage \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --splits train val test \
  --output-json "${report_root}/augmentation_leakage.json" \
  2>&1 | tee "${report_root}/leakage.log"

printf '%s\n' "${report_root}" > reports/p15_dsifn_gate_minus1/LATEST
# Gate 0 must read images through the same root the manifest was built
# against, which may be one wrapper directory below DSIFN-CD.
printf '%s\n' "${DATASET_ROOT}" \
  > reports/p15_dsifn_gate_minus1/DATASET_ROOT
printf '%s\n' "${DATASET_ROOT}" > "${report_root}/DATASET_ROOT"

echo "===== GATE -1 SUMMARY ====="
python - "${report_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
audit = json.loads(
    (root / "data_audit_DSIFN-CD.json").read_text(encoding="utf-8")
)
leak = json.loads(
    (root / "augmentation_leakage.json").read_text(encoding="utf-8")
)
print("samples per split      :", leak["samples_per_split"])
print("cross-split exact groups:", leak["cross_split_exact_duplicate_groups"])
print("cross-split near pairs  :", leak["cross_split_near_duplicate_pairs"])
print("official split usable   :", leak["decision"]["official_split_usable"])
print("required action         :", leak["decision"]["required_action"])
print("dataset audit status    :", audit.get("status", "see json"))
PY

echo "Gate -1 complete: ${report_root}"
