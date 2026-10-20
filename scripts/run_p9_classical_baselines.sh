#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
LEVIR_ROOT="${LEVIR_ROOT:-/root/autodl-tmp/rs_cd_work/data/LEVIR-CD}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=8

if [[ -z "${P7_CHECKPOINT:-}" ]]; then
  p7c_report="$(cat reports/p7c_fullres_calibration_levir/LATEST)"
  P7_CHECKPOINT="$(cat "$p7c_report/CHECKPOINT")"
fi

python - <<'PY'
try:
    import cv2
except ImportError:
    raise SystemExit(
        "opencv is required for ECC and SIFT: "
        "pip install opencv-python-headless"
    )
if not hasattr(cv2, "SIFT_create"):
    raise SystemExit("This OpenCV build has no SIFT_create")
print(f"opencv={cv2.__version__} sift=ok")
PY

for required in \
  "$P7_CHECKPOINT" \
  manifests/datasets/levir_cd.csv \
  configs/p9_classical_baselines.yaml; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p9_classical_baselines_levir/$tag"
mkdir -p "$report_root"
printf '%s\n' "$report_root" \
  > reports/p9_classical_baselines_levir/LATEST
printf '%s\n' "$P7_CHECKPOINT" > "$report_root/SOURCE_CHECKPOINT"

python -m pytest -q 2>&1 | tee "$report_root/pytest.log"

python -u -m tools.evaluate_classical_baselines \
  --checkpoint "$P7_CHECKPOINT" \
  --protocol configs/p9_classical_baselines.yaml \
  --dataset-root "$LEVIR_ROOT" \
  --manifest manifests/datasets/levir_cd.csv \
  --device cuda \
  --output-dir "$report_root/baselines" \
  2>&1 | tee "$report_root/baselines.log"

cp "$report_root/baselines/decision.json" "$report_root/decision.json"
echo "P9 classical baselines complete: $report_root"
