#!/usr/bin/env bash
set -euo pipefail

# P15 Gate 1: train the frozen P3b recipe on the leakage-free GZ-CD
# source-crop split, then measure input-level oracle headroom on val.
# This script never loads the GZ-CD test split.

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

GZCD_ROOT="${GZCD_ROOT:-/root/autodl-tmp/rs_cd_work/data/GZ-CD-tiles}"
MANIFEST="${MANIFEST:-manifests/datasets/gzcd.csv}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/p17_gzcd_p3b.yaml}"
ORACLE_CONFIG="${ORACLE_CONFIG:-configs/p17_gzcd_oracle_t2_shift.yaml}"

for required in \
  "${MANIFEST}" \
  "${TRAIN_CONFIG}" \
  "${ORACLE_CONFIG}" \
  reports/p17_gzcd_gate0/LATEST \
  reports/p17_gzcd_gate_minus1/LATEST; do
  test -f "${required}" || {
    echo "Missing Gate 1 input: ${required}" >&2
    exit 1
  }
done

gate0_root="$(cat reports/p17_gzcd_gate0/LATEST)"
build_root="$(cat reports/p17_gzcd_gate_minus1/LATEST)"
gate0_decision="${gate0_root}/gate0_decision.json"
candidate_hygiene="${gate0_root}/candidate_gzcd/hygiene_summary.json"
build_report="${build_root}/build.json"

for required in \
  "${gate0_decision}" \
  "${candidate_hygiene}" \
  "${build_report}"; do
  test -f "${required}" || {
    echo "Missing prerequisite result: ${required}" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p17_gzcd_gate1/${tag}"
mkdir -p "${report_root}"

echo "===== PREREGISTERED ROUTE AND SPLIT GUARDS ====="
python - \
  "${gate0_decision}" \
  "${build_report}" \
  "${MANIFEST}" \
  "${report_root}/prerequisite_guard.json" <<'PY'
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

gate0_path, build_path, manifest_path, output_path = map(Path, sys.argv[1:])
gate0 = json.loads(gate0_path.read_text(encoding="utf-8"))
build = json.loads(build_path.read_text(encoding="utf-8"))
if gate0.get("action") != "proceed_to_gate1":
    raise SystemExit(f"Gate 0 does not authorize Gate 1: {gate0.get('action')}")
if not gate0.get("geometry_match", {}).get("passed"):
    raise SystemExit("Gate 0 geometry-matched control did not pass")
if gate0.get("labels_read") or gate0.get("test_split_touched"):
    raise SystemExit("Gate 0 provenance guard failed")
if not all(build.get("geometric_matches_protocol", {}).values()):
    raise SystemExit("Tile counts disagree with the frozen protocol")

counts = Counter()
clusters = defaultdict(set)
with manifest_path.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        counts[row["split"]] += 1
        clusters[row["parent_id"]].add(row["split"])
spanning = {
    parent: sorted(splits)
    for parent, splits in clusters.items()
    if len(splits) > 1
}
if spanning:
    raise SystemExit(f"Raster spans splits: {list(spanning)[:5]}")
expected = {"train": 511, "val": 109, "test": 109}
if dict(counts) != expected:
    raise SystemExit(f"Unexpected manifest split counts: {dict(counts)}")

result = {
    "status": "passed",
    "gate0_branch": gate0["branch"],
    "gate0_action": gate0["action"],
    "manifest": str(manifest_path),
    "split_counts": dict(sorted(counts.items())),
    "source_rasters": len(clusters),
    "rasters_spanning_splits": 0,
    "evaluation_split": "val",
    "test_split_touched": False,
}
output_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

echo "===== GATE 0 ESTIMATOR-SUPPORT DIAGNOSTIC (NO GATE) ====="
python -m tools.summarize_p15_gate0_instrument \
  --hygiene "${candidate_hygiene}" \
  --search-radius-pixels 8 \
  --output "${report_root}/gate0_instrument_diagnosis.json" \
  2>&1 | tee "${report_root}/gate0_instrument_diagnosis.log"

echo "===== TRAIN/VAL MASK ENCODING GUARD (TEST UNTOUCHED) ====="
python - \
  "${GZCD_ROOT}" \
  "${MANIFEST}" \
  "${TRAIN_CONFIG}" \
  "${report_root}/mask_encoding_guard.json" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

root, manifest_path, config_path, output_path = map(Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
threshold = int(config["data"]["mask_threshold"])
stats = defaultdict(lambda: {
    "samples": 0,
    "positive_samples": 0,
    "positive_pixels": 0,
    "total_pixels": 0,
    "pixel_values": set(),
})
with manifest_path.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        split = row["split"]
        if split not in {"train", "val"}:
            continue
        mask = np.asarray(Image.open(root / row["mask_path"]))
        positive = mask > threshold
        item = stats[split]
        item["samples"] += 1
        item["positive_samples"] += int(bool(positive.any()))
        item["positive_pixels"] += int(positive.sum())
        item["total_pixels"] += int(positive.size)
        item["pixel_values"].update(int(v) for v in np.unique(mask))

result = {
    "status": "passed",
    "mask_threshold": threshold,
    "test_split_touched": False,
    "splits": {},
}
for split in ("train", "val"):
    item = stats[split]
    fraction = item["positive_pixels"] / max(1, item["total_pixels"])
    if item["samples"] == 0 or item["positive_samples"] == 0:
        raise SystemExit(
            f"{split}: no positive masks after threshold {threshold}"
        )
    if not 0.0 < fraction < 1.0:
        raise SystemExit(
            f"{split}: invalid positive fraction {fraction} "
            f"after threshold {threshold}"
        )
    result["splits"][split] = {
        "samples": item["samples"],
        "positive_samples": item["positive_samples"],
        "positive_fraction": fraction,
        "pixel_values": sorted(item["pixel_values"]),
    }
output_path.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, ensure_ascii=False))
PY

echo "===== TARGETED TESTS ====="
python -m pytest -q \
  tests/test_p15_gate1.py \
  tests/test_p15_dsifn_gates.py \
  tests/test_gzcd_tiles.py \
  tests/test_dataset.py \
  2>&1 | tee "${report_root}/pytest.log"

echo "===== GPU SMOKE ====="
python -m tools.gpu_smoke \
  --config "${TRAIN_CONFIG}" \
  --batch-size 32 \
  --image-size 256 \
  --steps 5 \
  --output "${report_root}/gpu_smoke_bs32.json" \
  2>&1 | tee "${report_root}/gpu_smoke_bs32.log"

echo "===== TRAIN GZ-CD P3B (TRAIN + VAL ONLY) ====="
python -u -m tools.train \
  --config "${TRAIN_CONFIG}" \
  --dataset-root "${GZCD_ROOT}" \
  --device cuda \
  2>&1 | tee "${report_root}/train.log"

summary="$(
  find outputs/p17_gzcd_dinov3_partial_ft \
    -name summary.json \
    -printf '%T@ %p\n' |
  sort -nr |
  head -n 1 |
  cut -d' ' -f2-
)"
test -n "${summary}"
run_dir="$(dirname "${summary}")"
checkpoint="${run_dir}/checkpoints/best.pth"
test -f "${checkpoint}"
printf '%s\n' "${run_dir}" > "${report_root}/RUN_DIR"
printf '%s\n' "${checkpoint}" > "${report_root}/CHECKPOINT"
cp "${summary}" "${report_root}/train_summary.json"

echo "===== RAW SHIFT ROBUSTNESS ON VAL ====="
python -u -m tools.evaluate_robustness \
  --checkpoint "${checkpoint}" \
  --protocol "${ORACLE_CONFIG}" \
  --dataset-root "${GZCD_ROOT}" \
  --split val \
  --device cuda \
  --paired-shift-support \
  --output-dir "${report_root}/raw_val" \
  2>&1 | tee "${report_root}/raw_val.log"

echo "===== INPUT-ORACLE INVERSE SHIFT ON VAL ====="
python -u -m tools.evaluate_robustness \
  --checkpoint "${checkpoint}" \
  --protocol "${ORACLE_CONFIG}" \
  --dataset-root "${GZCD_ROOT}" \
  --split val \
  --device cuda \
  --paired-shift-support \
  --output-dir "${report_root}/oracle_val" \
  --oracle-inverse-shift \
  2>&1 | tee "${report_root}/oracle_val.log"

echo "===== PREREGISTERED GATE 1 DECISION ====="
python -u -m tools.summarize_p8_whu_headroom \
  --dataset GZ-CD \
  --raw "${report_root}/raw_val/results.csv" \
  --oracle "${report_root}/oracle_val/results.csv" \
  --train-summary "${report_root}/train_summary.json" \
  --shift8-headroom-min 0.05 \
  --output "${report_root}/gate1_decision.json" \
  2>&1 | tee "${report_root}/gate1_decision.log"

sha256sum \
  "${MANIFEST}" \
  "${gate0_decision}" \
  "${candidate_hygiene}" \
  "${TRAIN_CONFIG}" \
  "${ORACLE_CONFIG}" \
  "${checkpoint}" \
  "${report_root}/raw_val/results.csv" \
  "${report_root}/oracle_val/results.csv" \
  "${report_root}/gate1_decision.json" \
  > "${report_root}/SHA256SUMS.txt"

python - "${report_root}/TEST_SPLIT_STATUS.txt" <<'PY'
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    "GZ-CD test split was not loaded or evaluated.\n"
    "Gate 1 used only the source-crop train and validation splits.\n"
    "Test remains sealed until every preregistered Gate 2 criterion "
    "passes.\n",
    encoding="utf-8",
)
PY

printf '%s\n' "${report_root}" > reports/p17_gzcd_gate1/LATEST
echo "P15 GZ-CD Gate 1 complete: ${report_root}"
cat "${report_root}/gate1_decision.json"
