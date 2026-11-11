"""Read the original train/test JSONs to identify all 18 calibrated thresholds."""
from pathlib import Path
import csv
import hashlib
import json

BASE = Path.cwd()
OUT = BASE / "reports/round1_20261103/E8_original_metadata"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for model in ("dinov3", "bit", "cdlamba", "cdxlstm", "snunet", "bifa"):
    if model == "dinov3":
        test = BASE / "reports/p30/test/20260918T134224Z/p30_test.json"
        summary = BASE / "reports/p30/selector/20260918T104133Z/bundles/train_summary.json"
    else:
        root = BASE / ("reports/p32" if model in ("snunet", "bifa") else "reports/p31") / model
        test = root / "test/p31_test.json"
        summary = root / "bundles/train_summary.json"
    data = json.loads(test.read_text(encoding="utf-8"))
    train = json.loads(summary.read_text(encoding="utf-8"))
    train = train.get("datasets", train)
    for name, entry in data["datasets"].items():
        entry = entry["proposed"] if model == "dinov3" else entry
        tau = entry["tau"]
        if train[name]["tau"] != tau:
            raise ValueError(f"training/test threshold mismatch: {model}/{name}")
        rows.append({"model": model, "dataset": name, "tau": tau, "best_epoch": train[name]["best_epoch"],
                     "source": str(test), "source_sha256": hashlib.sha256(test.read_bytes()).hexdigest()})
assert len(rows) == 18
with (OUT / "selector_thresholds_18.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / "selector_thresholds_18.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps({"combinations": len(rows), "above_050": [r for r in rows if r["tau"] > 0.5]}, indent=2))
