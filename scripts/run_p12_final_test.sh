#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"
SOURCE_P3B="${SOURCE_P3B:-outputs/p3b_dinov3_convnext_tiny_partial_ft/20260724T151633Z_seed3407/checkpoints/best.pth}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

freeze_root="$(cat reports/p11_structure_freeze/LATEST)"
second_root="$(cat reports/p11_second_detector_seed/LATEST)"
test -f "$freeze_root/STRUCTURE_FROZEN"
sha256sum -c "$freeze_root/FROZEN_SHA256SUMS.txt"

python - "$second_root/pipeline_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary["status"] != "completed":
    raise SystemExit("Second detector replication is incomplete")
if summary["p7_3_decision"]["status"] != "passed":
    raise SystemExit("Second detector P7-3 replication did not pass")
if not summary["test_split_untouched"]:
    raise SystemExit("Second detector pipeline touched test")
PY

P7_CHECKPOINT="$(
  python - "$freeze_root/freeze_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["sources"]["p7_checkpoint"])
PY
)"
TASK_ROUTER="$(
  python - "$freeze_root/freeze_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["sources"]["task_router"])
PY
)"
REGISTRATION_ROUTER="$(
  python - "$freeze_root/freeze_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["sources"]["registration_router"])
PY
)"

for required in \
  "$P7_CHECKPOINT" \
  "$SOURCE_P3B" \
  "$TASK_ROUTER" \
  "$REGISTRATION_ROUTER" \
  manifests/datasets/levir_cd.csv \
  configs/p12_levir_final_test.yaml; do
  test -f "$required" || {
    echo "Missing final-test input: $required" >&2
    exit 1
  }
done

mkdir -p reports/p12_final_test
consumed="reports/p12_final_test/LEVIR_TEST_CONSUMED.json"
if [[ -e "$consumed" ]]; then
  echo "LEVIR final test has already been started or consumed:" >&2
  cat "$consumed" >&2
  exit 1
fi

python -m pytest -q \
  tests/test_p12_final_test.py \
  tests/test_audit_repairs.py

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p12_final_test/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" > reports/p12_final_test/LATEST

python - "$consumed" "$report_root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "started_irreversible",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "report_root": sys.argv[2],
            "allowed_reruns": 0,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

python -u -m tools.evaluate_final_dual_path_test \
  --p7-checkpoint "$P7_CHECKPOINT" \
  --detector-checkpoint "$SOURCE_P3B" \
  --classifier-registration "$REGISTRATION_ROUTER" \
  --classifier-task "$TASK_ROUTER" \
  --protocol configs/p12_levir_final_test.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --device cuda \
  --output-dir "$report_root/levir" \
  2>&1 | tee "$report_root/levir.log"

python - "$consumed" "$report_root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
record["status"] = "completed"
record["completed_utc"] = datetime.now(timezone.utc).isoformat()
record["summary"] = str(
    Path(sys.argv[2]) / "levir" / "final_test_summary.json"
)
path.write_text(
    json.dumps(record, indent=2) + "\n",
    encoding="utf-8",
)
PY

cat > "$report_root/S2LOOKING_TEST_NOT_CONSUMED.txt" <<'EOF'
S2Looking labeled test was not consumed.
Reason: the preregistered Stage C validation unlabeled applicability gate
failed (Shift-8 conditional PCK@1 0.486 < 0.50). The protocol forbids
bypassing that gate merely to obtain labeled test numbers.
EOF

sha256sum \
  "$report_root/levir/final_test_summary.json" \
  "$report_root/levir/results.json" \
  "$consumed" \
  > "$report_root/SHA256SUMS.txt"

echo "P12 final test complete: $report_root"
cat "$report_root/levir/final_test_summary.json"
cat "$report_root/S2LOOKING_TEST_NOT_CONSUMED.txt"
