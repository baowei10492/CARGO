#!/usr/bin/env bash
set -euo pipefail

# P27 development check (configs/p27_rotation_check.yaml): untrained coarse
# matching on frozen DINOv3 tokens under rotation. Validation images only, no
# labels, no training. Decides only whether P27 version 1 handles rotation.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_rotation_check.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || {
  echo "$CONFIG is not marked preregistered" >&2
  exit 1
}
checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['backbone']['checkpoint_dir'])")"
test -d "$checkpoint_dir" || {
  echo "missing DINOv3 checkpoint directory $checkpoint_dir" >&2
  exit 1
}

tag="$(date -u +%Y%m%dT%H%M%SZ)"
report_root="reports/p27_localizer/rotation_check/$tag"
mkdir -p "$report_root"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_coarse_matching.py 2>&1 | tee "$report_root/pytest.log"

echo "===== ROTATION CHECK ====="
python -u -m tools.check_p27_rotation_feasibility \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$report_root/run" \
  2>&1 | tee "$report_root/check.log"

sha256sum "$CONFIG" src/rs_cd/coarse_matching.py tools/check_p27_rotation_feasibility.py \
  "$report_root"/run/* > "$report_root/SHA256SUMS.txt"
printf '%s\n' "$report_root" > reports/p27_localizer/rotation_check/LATEST
echo "P27 rotation check complete: $report_root"
