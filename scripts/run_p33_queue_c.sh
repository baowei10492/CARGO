#!/usr/bin/env bash
set -uo pipefail

# P33 queue C (docs/P33_REVIEWER_EXPERIMENTS.md): the light items, alongside queues A and B.
#   1. statistics from the frozen test records (CPU): CARGO minus always correcting, 18-cell means, mixed test
#   2. classical global registration in the 3-DoF family (Fourier-Mellin phase correlation, ECC): Table 7 rows and
#      Figure 4 errors
#   3. the registration module on identical pairs and known level-1 transforms (+ RoMa on identical pairs)
#   4. the interpolation control of the paper's CARGO on all eighteen combinations
#   5. phase correlation + the paper's selector recipe on Table 6's twelve combinations
# Finished parts are skipped.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3
mkdir -p reports/p33/logs

status=0
step() { echo "===== $1 ($(date -u +%FT%TZ)) ====="; }

step "STATISTICS"
if [ ! -f reports/p33/statistics/statistics.json ]; then
  python -u -m tools.p33_statistics || status=1
fi

step "CLASSICAL BASELINES: ALWAYS-REGISTERING ROWS AND ERRORS"
ABLATION="$(cat reports/p31/localizer_ablation/LATEST)"
python -u -m tools.p31_localizer_ablation --output-dir "$ABLATION" --variants phasecorr_fm ecc_ts || status=1
python -u -m tools.p31_registration_errors --output-dir reports/p31/overnight/run/errors \
  --variants phasecorr_fm ecc_ts || status=1

step "IDENTITY CHECK"
if [ ! -f reports/p33/identity/identity.json ]; then
  python -u -m tools.p33_selector_runs identity || status=1
fi

step "INTERPOLATION CONTROL (CARGO, 18 combinations)"
python -u -m tools.p33_selector_runs interp --rows dinov3 bit cdlamba cdxlstm snunet bifa || status=1

step "PHASE CORRELATION + SELECTOR (12 combinations)"
python -u -m tools.p33_selector_runs pipeline --localizer phasecorr_fm --rows dinov3 bit cdlamba cdxlstm || status=1

# reports/p33/identity was run without RoMa; this run adds RoMa on the identical pairs of the first 300 crops and
# repeats the registration module's numbers, which must equal reports/p33/identity
step "ROMA ON IDENTICAL PAIRS (300 crops per dataset)"
if [ ! -f reports/p33/identity_roma/identity/identity.json ]; then
  python -u -m tools.p33_selector_runs identity --roma-crops 300 --output-root reports/p33/identity_roma || status=1
fi

echo "P33 queue C finished with status $status ($(date -u +%FT%TZ))"
exit "$status"
