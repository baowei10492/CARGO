#!/usr/bin/env bash
set -uo pipefail

# Smoke of the P33 segments the first smoke (2026-10-08) could not reach because their inputs did not exist yet:
#   1. always-registering rows and registration errors of a retrained module (the conf_only training's current
#      best.pth behind a smoke pointer); the two must agree on the same crops
#   2. the selector recipe on that module (DINOv3, BIT; DSIFN-CD subset)
#   3. the S2Looking detector training (tiny: 1 epoch on 8 samples) and queue B's pointer step
#   4. the S2Looking selector pipeline with that detector (subset)
#   5. tools.p33_summary on these outputs (+ small interp and identity runs)
#   6. scripts/p33_gate_identity.sh with dummy processes
# Everything is written under reports/p33_smoke2; the real pointers are replaced only inside the smoke processes.

cd /root/autodl-tmp/rs_cd_work/rs_cd_project
source .venv/bin/activate
export PYTHONPATH=src OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
S=reports/p33_smoke2
rm -rf "$S"; mkdir -p "$S"
ok() { echo "SMOKE2 $1: $2"; }
PATCH='from rs_cd import p33_protocol
p33_protocol.RETRAINED_LOCALIZERS["v2r_conf_only"] = "reports/p33_smoke2/fake_conf_only/LATEST"
from tools import p33_selector_runs
p33_selector_runs.S2LOOKING_DETECTOR_POINTER = "reports/p33_smoke2/s2looking_detector/CHECKPOINT"'

# ------------------------------------------------------------------ 1. retrained module: ablation rows + errors
R=$(ls -d outputs/p33_localizer_conf_only/2* | tail -n 1)
mkdir -p "$S/fake_conf_only/run/checkpoints"
cp "$R/checkpoints/best.pth" "$S/fake_conf_only/run/checkpoints/best.pth"
cp "$R/config.yaml" "$S/fake_conf_only/run/config.yaml"
echo "$S/fake_conf_only/run" > "$S/fake_conf_only/LATEST"
timeout 900 python - <<PY > "$S/1_retrained.log" 2>&1
$PATCH
import torch
from pathlib import Path
from tools import p31_localizer_ablation as abl, run_p30
from tools.p31_registration_errors import crop_errors, success_by_level
from tools.run_p31_plugins import load
from tools.run_p27_selector import with_split

def flags(obj, depth=0):
    found = [bool(obj.use_change_weights)] if hasattr(obj, "use_change_weights") else []
    if depth < 3:
        for value in getattr(obj, "__dict__", {}).values():
            if hasattr(value, "__dict__") and not isinstance(value, torch.Tensor):
                found += flags(value, depth + 1)
    return found

config, p30, evaluation, _ = load(Path("configs/p31_border_free.yaml"))
evaluation = with_split(evaluation, config["ablation"]["split"])
device = torch.device("cuda")
localizer = abl.build("v2r_conf_only", p30, evaluation, device)
print("use_change_weights flags:", flags(localizer))
assert flags(localizer) and not any(flags(localizer)), "the retrained module is not run confidence-only"
spec = [s for s in evaluation["datasets"] if s["name"] == "DSIFN-CD"][0]
detector = run_p30.open_detector(p30, "original", "DSIFN-CD", device)
margin = int(p30["crop_margin"])
result = abl.evaluate(spec, evaluation, localizer, detector, device, margin=margin, max_crops=24)
print("ablation success", result["localization_success"], "L0 change", -result["clean_drop"]["estimated"],
      "L3 gain", result["gain"]["3"]["estimated"])
torch.manual_seed(int(evaluation["seed"]))
success = success_by_level(crop_errors(spec, evaluation, localizer, margin=margin, max_crops=24))
print("errors success", success)
assert all(abs(success[k] - result["localization_success"][k]) < 1e-9 for k in success), "errors and ablation disagree"
print("OK")
PY
ok "1 retrained module: confidence-only, ablation rows = errors" "exit $? ($(tail -n 1 $S/1_retrained.log))"

# ------------------------------------------------------------------ 2. selector recipe on the retrained module
timeout 1500 python - <<PY > "$S/2_retrained_selector.log" 2>&1
$PATCH
import sys
sys.argv = ["p33_selector_runs", "pipeline", "--localizer", "v2r_conf_only", "--rows", "dinov3", "bit",
            "--datasets", "DSIFN-CD", "--max-crops", "120", "--output-root", "reports/p33_smoke2"]
p33_selector_runs.main()
PY
ok "2 selector on the retrained module (120 crops)" "exit $?"; grep -E "^\[v2r|never|always|selector" "$S/2_retrained_selector.log" | tail -n 8

# ------------------------------------------------------------------ 3. S2Looking detector training + pointer step
python - <<'PY'
import yaml
config = yaml.safe_load(open("configs/p33/dinov3_s2looking.yaml"))
config["experiment"].update(name="p33_smoke2_s2looking", output_root="reports/p33_smoke2/outputs")
config["data"].update(cache_mode="none", num_workers=2, prefetch_factor=2)
config["training"].update(epochs=1, samples_per_epoch=16, batch_size=2, checkpoint_interval=1)
config["evaluation"].update(validation_interval=1, batch_size=1)     # full 1024 tiles next to the running jobs
yaml.safe_dump(config, open("reports/p33_smoke2/s2looking_smoke.yaml", "w"), sort_keys=False)
PY
pointer_dir="$S/s2looking_detector"; mkdir -p "$pointer_dir"; touch "$pointer_dir/.before_training"
timeout 1200 python -u -m tools.train --config "$S/s2looking_smoke.yaml" --device cuda --overfit-samples 8 > "$S/3_s2looking_train.log" 2>&1
ok "3a S2Looking detector training (1 epoch, 8 samples)" "exit $?"; tail -n 2 "$S/3_s2looking_train.log" | cut -c1-200
summary="$(find $S/outputs/p33_smoke2_s2looking -name summary.json -newer "$pointer_dir/.before_training" | sort | tail -n 1)"
# queue B's pointer step, verbatim
python - "$summary" "$pointer_dir" "$PWD" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path, pointer_dir, project_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
best = summary.get("best_f1")
if summary.get("status") != "completed" or best is None or not math.isfinite(float(best)):
    raise SystemExit(f"training did not produce a usable detector: status {summary.get('status')}, best F1 {best}")
checkpoint = summary_path.parent / "checkpoints" / "best.pth"
relative = checkpoint.resolve().relative_to(project_root)
(pointer_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(pointer_dir / "CHECKPOINT").write_text(f"{relative.as_posix()}\n", encoding="utf-8")
print(f"S2Looking detector: best validation F1 {float(best):.4f} at epoch {summary.get('best_epoch')} -> {relative}")
PY
ok "3b pointer step" "exit $? ($(cat $pointer_dir/CHECKPOINT 2>/dev/null))"
sha256sum configs/p33/dinov3_s2looking.yaml "$(cat "$pointer_dir/CHECKPOINT")" > /dev/null && ok "3c checksum step" "ok"

# ------------------------------------------------------------------ 4. S2Looking selector pipeline
timeout 1500 python - <<PY > "$S/4_s2looking_selector.log" 2>&1
$PATCH
import sys
sys.argv = ["p33_selector_runs", "pipeline", "--localizer", "v2", "--s2looking", "--max-crops", "120",
            "--output-root", "reports/p33_smoke2"]
p33_selector_runs.main()
PY
ok "4 S2Looking selector pipeline (120 crops)" "exit $?"; grep -E "^\[v2|never|always|selector" "$S/4_s2looking_selector.log" | tail -n 4

# ------------------------------------------------------------------ 5. read-out on the smoke outputs
timeout 900 python -u -m tools.p33_selector_runs interp --rows dinov3 --datasets DSIFN-CD --max-crops 10 --output-root "$S" > "$S/5_interp.log" 2>&1
ok "5a interp (10 crops)" "exit $?"
timeout 900 python -u -m tools.p33_selector_runs identity --max-crops 4 --roma-crops 0 --output-root "$S" > "$S/5_identity.log" 2>&1
timeout 900 python -u -m tools.p33_selector_runs identity --max-crops 4 --roma-crops 0 --output-root "$S/identity_roma" >> "$S/5_identity.log" 2>&1
ok "5b identity twice (4 crops)" "exit $?"
timeout 900 python - <<PY > "$S/5_summary.log" 2>&1
$PATCH
import sys
from pathlib import Path
from tools import p33_summary
p33_summary.ROOT = Path("reports/p33_smoke2")
sys.argv = ["p33_summary", "--output-dir", "reports/p33_smoke2/summary", "--repeats", "200"]
p33_summary.main()
PY
ok "5c read-out" "exit $?"; grep -E "^## 2. Retrained module v2r_conf_only|v2r_conf_only|^## 3|crops, .* tiles|^## 4|dinov3/DSIFN-CD|^## 6|module repeat" "$S/5_summary.log" | head -n 12

# ------------------------------------------------------------------ 6. identity gate with dummy processes
sed -e 's|^STEP=.*|STEP="^gate-smoke-step"|' -e 's|^TRAINING=.*|TRAINING="^gate-smoke-training"|' -e 's|sleep 5|sleep 1|' \
  scripts/p33_gate_identity.sh > "$S/gate_smoke.sh"
(exec -a gate-smoke-training sleep 12) &
sleep 1
bash "$S/gate_smoke.sh" > "$S/6_gate.log" 2>&1 &
sleep 2
(exec -a gate-smoke-step sleep 40) &
step=$!
sleep 4
during=$(ps -o stat= -p "$step" | cut -c1)
sleep 10
after=$(ps -o stat= -p "$step" | cut -c1)
kill "$step" 2>/dev/null
ok "6 gate" "state while training runs: $during (T = paused), after it ends: $after (S = running)"; cat "$S/6_gate.log"

nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "SMOKE2 done"
