#!/usr/bin/env bash
set -euo pipefail

# P20 Gate -1: build the HRCUS-CD manifest and audit data hygiene.
# Follows docs/P20_HRCUS_SECOND_POSITIVE_PROTOCOL.md, frozen before any
# HRCUS training. No model is loaded and no threshold is evaluated here.
#
# Two things differ from the GZ-CD Gate -1 and both are deliberate:
#   1. HRCUS-CD ships pre-tiled with an official split, so the split is
#      NOT constructed by us and spatial disjointness has no structural
#      guarantee. The leakage audit is therefore the decisive check of
#      this gate, not a formality. DSIFN failed exactly here.
#   2. The test split was never uploaded, so the audit can only compare
#      train against val. A test-inclusive audit is required later,
#      before Gate 3, and is enforced by the config.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/rs_cd_work/data/HRCUS-CD}"
MANIFEST="${MANIFEST:-manifests/datasets/hrcus_cd.csv}"
CONFIG="${CONFIG:-configs/p20_hrcus_screening.yaml}"
# Written by tools/build_hrcus_clean_split.py. Its presence means the
# manifest has already been cleaned per protocol section 3.3 and must not
# be regenerated from the raw directory tree.
CLEAN_REPORT="${CLEAN_REPORT:-manifests/datasets/hrcus_clean_split.json}"

for split in train val; do
  for sub in A B label; do
    test -d "${DATASET_ROOT}/${split}/${sub}" || {
      echo "Missing ${DATASET_ROOT}/${split}/${sub}" >&2
      echo "Unzip HRCUS_trainval.zip so that train/ and val/ sit" >&2
      echo "directly under DATASET_ROOT." >&2
      exit 1
    }
  done
done

if [[ -d "${DATASET_ROOT}/test" ]]; then
  echo "REFUSING TO RUN: ${DATASET_ROOT}/test exists." >&2
  echo "The test split is meant to be absent until Gate 2 passes." >&2
  exit 1
fi

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p20_hrcus_gate_minus1/${tag}"
mkdir -p "${report_root}" manifests/datasets

echo "===== TEST SUITE ====="
python -m pytest -q

if [[ -f "${CLEAN_REPORT}" ]]; then
  echo "===== USING PRE-CLEANED MANIFEST ====="
  echo "Found ${CLEAN_REPORT}; not regenerating ${MANIFEST}."
  test -f "${MANIFEST}" || {
    echo "Cleaning report present but ${MANIFEST} is missing." >&2
    exit 1
  }
  cp "${CLEAN_REPORT}" "${report_root}/clean_split.json"
  python - "${MANIFEST}" "${CLEAN_REPORT}" <<'PY'
import csv
import json
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
report = json.load(open(sys.argv[2], encoding="utf-8"))
counts = Counter(row["split"] for row in rows)
expected = {k: v for k, v in report["samples_after"].items() if k != "test"}
print("manifest counts:", dict(counts))
print("cleaning report :", expected)
if dict(counts) != expected:
    raise SystemExit(
        "manifest does not match the cleaning report; upload the manifest "
        "produced by tools/build_hrcus_clean_split.py"
    )
print(f"degenerate tiles removed  : {report['degenerate_tiles_removed']}")
print(f"training duplicates dropped: {report['training_samples_dropped']}")
print(f"hash false-positive rate  : {report['hash_false_positive_rate']:.1%}")
PY
else
  echo "===== BUILD MANIFEST ====="
  # split_stem is mandatory here: every filename repeats across splits
  # (val/label/00000.png and train/label/00000.png both exist), so plain
  # stem ids would collide exactly as they did on DSIFN-CD (record 110.3).
  python -m tools.build_manifest \
    --dataset-root "${DATASET_ROOT}" \
    --output "${MANIFEST}" \
    --splits train val \
    --t1-dir A --t2-dir B --mask-dir label \
    --sample-id-mode split_stem \
    2>&1 | tee "${report_root}/build_manifest.log"
fi

python - "${MANIFEST}" <<'PY'
import csv
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
per_split = Counter(row["split"] for row in rows)
ids = [row["sample_id"] for row in rows]
print("rows per split:", dict(per_split))
if len(set(ids)) != len(ids):
    duplicated = [i for i, n in Counter(ids).items() if n > 1][:5]
    raise SystemExit(f"sample_id collision, e.g. {duplicated}")
if "test" in per_split:
    raise SystemExit("manifest contains a test split; it must not be present")
print("sample_id unique:", len(set(ids)))
PY

echo "===== DATASET AUDIT ====="
python -m tools.audit_dataset \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --output-json "${report_root}/data_audit_HRCUS-CD.json" \
  --output-md "${report_root}/data_audit_HRCUS-CD.md" \
  2>&1 | tail -20

echo "===== CROSS-SPLIT DUPLICATE AND AUGMENTATION LEAKAGE ====="
# This is the gate. The official split is not ours, so a non-zero result
# means the official partition is unusable and the line must move to a
# source-crop resplit before anything is trained.
python -m tools.audit_augmentation_leakage \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --splits train val \
  --output-json "${report_root}/leakage_audit.json" \
  2>&1 | tail -25

echo "===== LABEL MORPHOLOGY (ordering only, not a gate) ====="
python -m tools.audit_change_target_scale \
  --dataset-name HRCUS-CD \
  --dataset-root "${DATASET_ROOT}" \
  --manifest "${MANIFEST}" \
  --split val \
  --mask-threshold 127 \
  --output "${report_root}/morphology_HRCUS-CD.json" \
  2>&1 | tail -30

echo "===== GATE -1 SUMMARY ====="
python - "${report_root}" "${CONFIG}" "${CLEAN_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
expect = config["preflight_expectations"]

leak = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
morph = json.loads(
    (root / "morphology_HRCUS-CD.json").read_text(encoding="utf-8")
)

self_iou = float(morph["mask_self_iou_on_paired_support"]["8"])
diameter = float(
    morph["components"]["area_weighted_equivalent_diameter"]["median"]
)
change_fraction = float(morph["change_fraction"])

exact = leak["cross_split_exact_duplicate_groups"]
near = leak["cross_split_near_duplicate_pairs"]

print("cross-split exact duplicates :", exact)
print("cross-split near duplicates  :", near)
print(f"change fraction              : {change_fraction:.4f}")
print(
    f"self-IoU at 8 px             : {self_iou:.4f}"
    "  (LEVIR 0.520, S2Looking 0.610, WHU 0.749, GZ-CD 0.803, DSIFN 0.888)"
)
print(
    f"area-weighted diameter       : {diameter:.1f} px"
    "  (LEVIR 37.5, WHU 91.6, DSIFN 286.1)"
)

# Instrument check before interpretation: the server must reproduce what
# was measured locally from the same labels (main record 85).
failures = []
if abs(self_iou - expect["val_self_iou_at_8"]) > expect["val_self_iou_tolerance"]:
    failures.append(
        f"self-IoU@8 {self_iou:.4f} does not reproduce the preflight "
        f"{expect['val_self_iou_at_8']:.4f}"
    )
if (
    abs(diameter - expect["val_area_weighted_median_diameter_px"])
    > expect["val_diameter_tolerance_px"]
):
    failures.append(
        f"area-weighted diameter {diameter:.1f} px does not reproduce the "
        f"preflight {expect['val_area_weighted_median_diameter_px']:.1f} px; "
        "the scipy/cv2 connectivity equivalence assumed locally is wrong"
    )
if morph["positive_samples"] == 0:
    failures.append("morphology audit found no change pixels")
if failures:
    raise SystemExit("GATE -1 INSTRUMENT FAILURE:\n  " + "\n  ".join(failures))

clean_path = Path(sys.argv[3])
cleaned = clean_path.is_file()
if cleaned:
    # The hash-based near-duplicate count is NOT decisive on this dataset.
    # Pixel verification showed a 94.8% false-positive rate: a 64-bit
    # difference hash saturates on low-texture urban tiles (record 135).
    # The decisive checks are the cryptographic exact-duplicate count and
    # the recorded cleaning, which already removed every pixel-verified
    # cross-split duplicate from the training side.
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    print(
        f"\nhash near-duplicate count above is NOT decisive "
        f"({clean['hash_false_positive_rate']:.1%} false positives, record 135)."
    )
    print(
        f"pixel-verified cross-split pairs : "
        f"{clean['pixel_verified_cross_split_pairs']}"
    )
    print(
        f"training duplicates dropped      : {clean['training_samples_dropped']}"
    )
    verdict = "passed_after_recorded_cleaning" if exact == 0 else "unresolved_exact_duplicates"
else:
    verdict = "passed" if (exact == 0 and near == 0) else "official_split_unusable"

print("\nverdict :", verdict)
if verdict == "official_split_unusable":
    print(
        "The official partition leaks. Do not train on it. Move to a "
        "source-crop resplit per protocol section 3.2, or clean it per "
        "section 3.3, and record which was done."
    )
elif verdict == "unresolved_exact_duplicates":
    print(
        "Exact cross-split duplicates survive the cleaning. That is a "
        "cryptographic match, not a hash collision; do not proceed."
    )
print(
    "\nNOTE: only train and val were compared. The test split is not on "
    "this machine by design. A leakage audit including test is required "
    "before Gate 3."
)
(root / "gate_minus1_decision.json").write_text(
    json.dumps(
        {
            "status": "completed",
            "dataset": "HRCUS-CD",
            "verdict": verdict,
            "official_split_cleaned": cleaned,
            "cross_split_exact_duplicate_groups": exact,
            "cross_split_near_duplicate_pairs_hash_only": near,
            "hash_near_duplicate_count_is_decisive": not cleaned,
            "splits_compared": ["train", "val"],
            "test_split_present": False,
            "leakage_audit_including_test_required_before_gate3": True,
            "self_iou_at_8": self_iou,
            "area_weighted_median_diameter_px": diameter,
            "change_fraction": change_fraction,
            "reproduces_preflight": True,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "${report_root}" > reports/p20_hrcus_gate_minus1/LATEST
printf '%s\n' "${DATASET_ROOT}" > reports/p20_hrcus_gate_minus1/DATASET_ROOT
sha256sum "${report_root}"/*.json "${MANIFEST}" \
  > "${report_root}/SHA256SUMS.txt"
echo "P20 Gate -1 complete: ${report_root}"
