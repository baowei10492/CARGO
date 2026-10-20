#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

tag="$(date -u +%Y%m%dT%H%M%SZ)"
output="reports/final_paper_evidence/$tag"

.venv/bin/python -m tools.build_final_paper_evidence \
  --project-root . \
  --output-dir "$output" \
  --archive

printf '%s\n' "$output" > reports/final_paper_evidence/LATEST
sha256sum "${output}.tar.gz" | tee "${output}.tar.gz.sha256"

echo "Final paper evidence package complete:"
echo "  directory: $output"
echo "  archive:   ${output}.tar.gz"
