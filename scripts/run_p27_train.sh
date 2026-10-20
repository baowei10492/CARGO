#!/usr/bin/env bash
set -euo pipefail

# Train the P27 version 1 localizer (docs/P27_OWN_LOCALIZER_DESIGN.md).
# Training splits only; development decisions use the hashed 5% hold-out.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p27_localizer_v1.yaml}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for spec in config["sources"]:
    for key in ("dataset_root", "manifest"):
        if not Path(spec[key]).exists():
            raise SystemExit(f"{spec['name']}: {key} {spec[key]} does not exist")
if not Path(config["model"]["checkpoint_dir"]).is_dir():
    raise SystemExit(f"missing DINOv3 directory {config['model']['checkpoint_dir']}")
print("inputs ok")
PY

tag="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="outputs/p27_localizer/$tag"
mkdir -p outputs/p27_localizer

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p27_localizer.py tests/test_p27_training.py tests/test_affine_registration.py

echo "===== TRAINING ($run_root) ====="
python -u -m tools.train_similarity_localizer \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$run_root" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"}

printf '%s\n' "$run_root" > outputs/p27_localizer/LATEST
sha256sum "$CONFIG" src/rs_cd/models/change_aware_similarity_localizer.py src/rs_cd/similarity_geometry.py \
  src/rs_cd/datasets/localizer_pairs.py src/rs_cd/localizer_training.py tools/train_similarity_localizer.py \
  "$run_root"/checkpoints/*.pth > "$run_root/SHA256SUMS.txt"
echo "P27 training complete: $run_root"
cat "$run_root/train_summary.json"
