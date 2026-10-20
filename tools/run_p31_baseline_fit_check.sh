#!/usr/bin/env bash
# Matching baselines in the registration module's transformation family (reviewer check, 2026-10-08).
#
# RoMa and SIFT fitted with translation + isotropic scale (3 DoF): RoMa by RANSAC at 3 px and by certainty-weighted
# least squares with Huber reweighting, SIFT by RANSAC at 3 px. Same test crops, draws, tile-based perturbation and
# DINOv3 detectors as the P31 localizer ablation. Run from the project root:
#
#     nohup bash tools/run_p31_baseline_fit_check.sh > reports/p31/baseline_fit_check.out 2>&1 &
#
# About 4 hours on one RTX 4090 (two RoMa passes over 4792 crops x 4 levels; the second RoMa fit reuses the matches).
# Everything to bring back is copied into reports/p31/baseline_fit_check/<stamp>/.
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH=src
VARIANTS=(roma_ts_ransac roma_ts_huber sift_ts_ransac)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=reports/p31/baseline_fit_check/$STAMP
ERRORS=reports/p31/overnight/run/errors
ABLATION=$(cat reports/p31/localizer_ablation/LATEST)
mkdir -p "$OUT"
echo "ablation run: $ABLATION" | tee "$OUT/run.log"

python -m pytest tests/test_translation_scale_baselines.py -q 2>&1 | tee "$OUT/pytest.log"

# 1. detection F1 when the registration result is adopted for every pair (Table 7); added to the ablation run
python -m tools.p31_localizer_ablation --output-dir "$ABLATION" --variants "${VARIANTS[@]}" 2>&1 | tee "$OUT/ablation.log"

# 2. per-crop registration errors (Figure 4, Sections 1 and 4.7.2), anchored to the success rates of step 1
python -m tools.p31_registration_errors --output-dir "$ERRORS" --variants "${VARIANTS[@]}" 2>&1 | tee "$OUT/errors.log"

# 3. the numbers the paper needs, next to the original 4-DoF baselines
python -m tools.p31_baseline_fit_check --output-dir "$OUT" 2>&1 | tee "$OUT/report.log"

for variant in "${VARIANTS[@]}"; do
    cp "$ABLATION/$variant.json" "$OUT/ablation_$variant.json"
    cp "$ERRORS/errors_$variant.csv" "$ERRORS/$variant.json" "$OUT/"
done
(cd "$OUT" && sha256sum * > SHA256SUMS.txt)
echo "done: $OUT" | tee -a "$OUT/run.log"
