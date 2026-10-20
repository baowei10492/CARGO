#!/usr/bin/env bash
set -euo pipefail

python -m tools.make_synthetic_dataset \
  --output-root data/synthetic \
  --manifest manifests/datasets/synthetic.csv \
  --overwrite

python -m tools.audit_dataset \
  --dataset-root data/synthetic \
  --manifest manifests/datasets/synthetic.csv \
  --output-json reports/data_audit_synthetic.json \
  --output-md reports/data_audit_synthetic.md

pytest -q

python -m tools.train \
  --config configs/smoke_synthetic.yaml \
  --device auto \
  --overfit-samples 4

