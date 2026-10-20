"""Check a WHU-CD detector trained elsewhere against the P21-recipe config, and install it.

The checkpoint's embedded training config is compared with
configs/p21/<detector>_whu.yaml. Machine-specific keys are ignored; any other
difference is listed. With --install and no difference (or with
--accept-differences, which records them), the checkpoint is copied under
outputs/p27_whu_detectors/<detector>/ and the pointer file the plug-in configs
read is written, so scripts/run_p27_whu_detectors.sh skips training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

IGNORED = {
    ("experiment", "output_root"), ("experiment", "name"),
    ("data", "dataset_root"), ("data", "num_workers"), ("data", "prefetch_factor"), ("data", "cache_mode"),
    ("model", "repo_root"),
    ("training", "log_interval"), ("training", "checkpoint_interval"),
    ("evaluation", "save_predictions"),
}
BLOCKS = ("experiment", "data", "model", "loss", "training", "evaluation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def differences(expected: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for block in BLOCKS:
        want, have = expected.get(block, {}), actual.get(block, {})
        for key in sorted(set(want) | set(have)):
            if (block, key) in IGNORED:
                continue
            if want.get(key, "<missing>") != have.get(key, "<missing>"):
                found.append({"key": f"{block}.{key}", "recipe": want.get(key, "<missing>"),
                              "checkpoint": have.get(key, "<missing>")})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, choices=("bit", "cdlamba", "cdxlstm"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest-sha256", default=None,
                        help="sha256 of manifests/datasets/whu_cd.csv on the machine that trained it")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--accept-differences", action="store_true")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    recipe_path = Path("configs/p21") / f"{args.detector}_whu.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in payload or "model" not in payload:
        raise SystemExit(f"{checkpoint} is not a tools.train checkpoint (no embedded config)")
    found = differences(recipe, payload["config"])
    local_manifest = Path(recipe["data"]["manifest"])
    manifest = {"local_sha256": sha256(local_manifest), "training_machine_sha256": args.manifest_sha256}
    manifest["same"] = args.manifest_sha256 is not None and args.manifest_sha256 == manifest["local_sha256"]
    report = {
        "detector": args.detector,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": payload.get("epoch"),
        "val_metrics": payload.get("val_metrics"),
        "recipe": str(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "differences": found,
        "manifest": manifest,
    }
    print(json.dumps(report, indent=2, default=str))
    if not args.install:
        return
    if not manifest["same"]:
        raise SystemExit("the WHU split is not proven identical (pass --manifest-sha256 from the training machine)")
    if found and not args.accept_differences:
        raise SystemExit(f"{len(found)} recipe differences; re-run with --accept-differences to use it anyway")
    target_dir = Path("outputs/p27_whu_detectors") / args.detector
    target_dir.mkdir(parents=True, exist_ok=True)
    if (target_dir / "CHECKPOINT").exists():
        raise SystemExit(f"{target_dir / 'CHECKPOINT'} already exists")
    target = target_dir / "external_best.pth"
    if checkpoint.resolve() != target.resolve():
        shutil.copy2(checkpoint, target)
    if sha256(target) != report["checkpoint_sha256"]:
        raise SystemExit("copy does not match the checked checkpoint")
    report["installed_as"] = target.as_posix()
    report["accepted_differences"] = bool(found)
    (target_dir / "external_check.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (target_dir / "CHECKPOINT").write_text(target.as_posix() + "\n", encoding="utf-8")
    print(f"installed: {target_dir / 'CHECKPOINT'} -> {target}")


if __name__ == "__main__":
    main()
