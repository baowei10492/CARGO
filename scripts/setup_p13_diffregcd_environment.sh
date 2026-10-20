#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_ROOT="${1:-/root/autodl-tmp/rs_cd_work/DiffRegCD}"
PYTHON="${EXTERNAL_ROOT}/.venv/bin/python"

test -x "${PYTHON}" || {
  echo "Missing external Python environment: ${PYTHON}" >&2
  exit 1
}

"${PYTHON}" -m pip install \
  einops \
  kornia \
  "numpy<2" \
  "opencv-python-headless>=4.8,<4.12" \
  thop

"${PYTHON}" - <<'PY'
import importlib

modules = [
    "cv2",
    "einops",
    "kornia",
    "numpy",
    "thop",
    "torch",
    "torchvision",
    "yaml",
]
for name in modules:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "n/a")
    print(f"{name}: {version}")
PY

echo "P13 DiffRegCD environment is ready."
