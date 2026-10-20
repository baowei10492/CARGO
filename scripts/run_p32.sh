#!/usr/bin/env bash
set -uo pipefail

# P32 (docs/P32_PLUGIN_EXTENSION.md): two added detectors, SNUNet and BiFA.
#
#   bash scripts/run_p32.sh              # smoke -> train six cells -> both plug-in rows
#   CELLS="bifa_levir" bash scripts/run_p32.sh
#   STAGE=smoke bash scripts/run_p32.sh  # only the pre-flight forwards
#
# Every step is skipped when its result already exists, so the script can be
# restarted after an interruption. A cell whose training fails is reported and
# the next one still runs.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/rs_cd_work/rs_cd_project}"
CONFIG="${CONFIG:-configs/p32_plugin_extension.yaml}"
STAGE="${STAGE:-all}"

cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

grep -q "^status: preregistered" "$CONFIG" || { echo "$CONFIG is not marked preregistered" >&2; exit 1; }
test -f docs/P32_PLUGIN_EXTENSION.md
test -f outputs/p30_localizer/LATEST || { echo "localizer v2 missing (outputs/p30_localizer/LATEST)" >&2; exit 1; }
CELLS="${CELLS:-snunet_levir snunet_dsifn snunet_whu bifa_levir bifa_dsifn bifa_whu}"
case "$CELLS" in
  *bifa*) test -f external/mit_b0_imagenet.pth || {
            echo "external/mit_b0_imagenet.pth missing; run:" >&2
            echo "  python -m tools.p32_convert_mit_b0 --download --repo-root <BiFA checkout>" >&2
            exit 1
          } ;;
esac
mkdir -p reports/p32

echo "===== PRE-FLIGHT FORWARDS ====="
for cell in $CELLS; do
  python -u -m tools.p32_smoke --config "configs/p21/${cell}.yaml" --device cuda \
    2>&1 | tee -a reports/p32/smoke.log
  test "${PIPESTATUS[0]}" -eq 0 || { echo "$cell: smoke failed" >&2; exit 1; }
done
[ "$STAGE" = "smoke" ] && exit 0

train_one() {
  local cell="$1"
  local config="configs/p21/${cell}.yaml"
  local pointer_dir="outputs/p32_detectors/$cell"
  if [ -f "$pointer_dir/CHECKPOINT" ]; then
    echo "$cell: already trained ($(cat "$pointer_dir/CHECKPOINT"))"
    return 0
  fi
  mkdir -p "$pointer_dir"
  local marker="$pointer_dir/.before_training"
  touch "$marker"
  echo "===== TRAIN $cell ====="
  python -u -m tools.train --config "$config" --device cuda 2>&1 | tee "$pointer_dir/train.log" || return 1
  local summary
  summary="$(find "outputs/p32_${cell}" -name summary.json -newer "$marker" | sort | tail -n 1)"
  test -n "$summary" || { echo "$cell: no summary.json" >&2; return 1; }
  python - "$summary" "$pointer_dir" "$PROJECT_ROOT" <<'PY' || return 1
import json
import math
import sys
from pathlib import Path

summary_path, pointer_dir, project_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
best = summary.get("best_f1")
if summary.get("status") != "completed" or best is None or not math.isfinite(float(best)) or float(best) < 0.5:
    raise SystemExit(f"training did not produce a usable detector: status {summary.get('status')}, best F1 {best}")
checkpoint = summary_path.parent / "checkpoints" / "best.pth"
if not checkpoint.is_file():
    raise SystemExit(f"missing {checkpoint}")
relative = checkpoint.resolve().relative_to(project_root)
(pointer_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(pointer_dir / "CHECKPOINT").write_text(f"{relative.as_posix()}\n", encoding="utf-8")
print(f"best validation F1 {float(best):.4f} at epoch {summary.get('best_epoch')}; pointer -> {relative}")
PY
  sha256sum "$config" "$(cat "$pointer_dir/CHECKPOINT")" > "$pointer_dir/SHA256SUMS.txt"
}

failed=0
for cell in $CELLS; do
  echo "########## $cell ##########"
  train_one "$cell" || { echo "########## $cell: FAILED ##########"; failed=1; }
done
[ "$STAGE" = "train" ] && exit "$failed"

echo "===== TARGETED TESTS ====="
python -m pytest -q tests/test_p31.py tests/test_p30_evaluation.py tests/test_external_bifa.py \
  2>&1 | tee -a reports/p32/pytest.log
test "${PIPESTATUS[0]}" -eq 0 || { echo "tests failed" >&2; exit 1; }

for row in ${ROWS:-snunet bifa}; do
  echo "===== P32 ROW $row ====="
  if python -u -m tools.run_p31_plugins all --row "$row" --config "$CONFIG" 2>&1 | tee -a "reports/p32/$row.log"; then
    echo "########## $row: done ##########"
  else
    echo "########## $row: FAILED ##########"
    failed=1
  fi
done
echo "P32 finished with status $failed"
exit "$failed"
