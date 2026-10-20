#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ARCHIVE="${ARCHIVE:-p13_diffregcd_final_evidence.tar.gz}"

if [[ -e "${ARCHIVE}" && "${FORCE:-0}" != "1" ]]; then
  echo "Archive already exists: ${ARCHIVE}" >&2
  echo "Re-run with FORCE=1 to replace it." >&2
  exit 1
fi

echo "===== TARGETED TESTS ====="
python -m pytest -q \
  tests/test_build_p13_final_evidence.py \
  tests/test_build_final_paper_evidence.py

echo "===== FROZEN MARKER PRECHECK ====="
python - <<'PY'
import json
from pathlib import Path

for path, required in (
    (
        Path("reports/p12_final_test/LEVIR_TEST_CONSUMED.json"),
        {"status": "completed", "allowed_reruns": 0},
    ),
    (
        Path(
            "reports/p13_diffregcd_final_test/"
            "DIFFREGCD_LEVIR_TEST_CONSUMED.json"
        ),
        {"status": "completed", "allowed_reruns": 0},
    ),
):
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value in required.items():
        if record.get(key) != value:
            raise SystemExit(f"{path}: {key}={record.get(key)!r}")
    print(f"{path}: completed, allowed_reruns=0")
PY

tag="$(date -u +%Y%m%dT%H%M%SZ)"
output="reports/p13_final_evidence/${tag}"

echo "===== BUILD P13 EVIDENCE PACKAGE (packaging only) ====="
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"
python -m tools.build_p13_final_evidence \
  --project-root . \
  --output-dir "${output}" \
  --archive-path "${ARCHIVE}"

mkdir -p reports/p13_final_evidence
printf '%s\n' "${output}" > reports/p13_final_evidence/LATEST

echo "===== PACKAGE SUMMARY ====="
python - "${output}/verification.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
gates = (
    "markers",
    "external_freeze",
    "external_validation",
    "recorded_number_reproduction",
    "recomputed_comparison_matches_recorded",
)
print(f"status = {record['status']}")
for gate in gates:
    print(f"{gate} = {record[gate]['passed']}")
print(f"missing_protocol_files = {record['missing_protocol_files']}")
if record["status"] != "verified" or not all(
    record[gate]["passed"] for gate in gates
):
    raise SystemExit("P13 evidence verification did not pass")
PY

cat "${output}/paper_assets/table_external_comparison.md"
ls -lh "${ARCHIVE}"
cat "${ARCHIVE}.sha256"

echo "P13 evidence package complete:"
echo "  directory: ${output}"
echo "  archive:   ${ARCHIVE}"
