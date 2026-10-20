"""Check a machine before the P29 augmentation baseline (scripts/run_p29_aug_baseline.sh).

Reads only. For each dataset of configs/p27_eval_perturbation.yaml: the manifest
hash against the one the P27 rows used, and that every file of every split is
present. Then the DINOv3 backbone weights, the P27 localizer run, the chained
config hashes and the free GPU memory. Exits with status 1 if anything a run
needs is missing or differs.
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

# sha256 of the manifests behind the P27 rows (LEVIR: this repository; DSIFN: the
# paper supplement copy; WHU: both servers, main record 192.17).
MANIFESTS = {
    "LEVIR-CD": "0391b00382382393fb1b5e23a69eebe4998aa5e19f24b0f0ade1b680b251501f",
    "DSIFN-CD": "03e86e3aabd35fb95b004caeaabd7c394eab9e1ba9358e945467c86d419d7c52",
    "WHU-CD": "7fd30ffcc5f14c658364b6c1cb56a598b0fb0ac4893a8612f467c46ca3d1f10e",
}
EVALUATION = Path("configs/p27_eval_perturbation.yaml")
TRAINING = {"LEVIR-CD": "configs/p29/dinov3_aug_levir.yaml", "DSIFN-CD": "configs/p29/dinov3_aug_dsifn.yaml",
            "WHU-CD": "configs/p29/dinov3_aug_whu.yaml"}
SELECTOR = Path("configs/p27_plugins/dinov3aug_selector.yaml")
MIN_FREE_GPU_MIB = 16000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dataset(name: str, root: Path, manifest: Path) -> list[str]:
    problems = []
    if not manifest.is_file():
        return [f"{name}: manifest {manifest} is missing"]
    digest = sha256(manifest)
    if digest != MANIFESTS[name]:
        problems.append(f"{name}: manifest sha256 {digest} is not {MANIFESTS[name]} (the split differs)")
    if not root.is_dir():
        return problems + [f"{name}: dataset root {root} is missing"]
    counts, missing = {}, []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["split"]] = counts.get(row["split"], 0) + 1
            missing += [row[key] for key in ("t1_path", "t2_path", "mask_path") if not (root / row[key]).is_file()]
    print(f"  {name}: {root}  rows {counts}  missing files {len(missing)}")
    if missing:
        problems.append(f"{name}: {len(missing)} files missing under {root}, e.g. {missing[:3]}")
    return problems


def free_gpu_mib() -> list[int]:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [int(line) for line in output.split() if line.strip().isdigit()]


def main() -> int:
    problems: list[str] = []
    evaluation = yaml.safe_load(EVALUATION.read_text(encoding="utf-8"))
    print("datasets (P27 evaluation paths):")
    for spec in evaluation["datasets"]:
        problems += check_dataset(spec["name"], Path(spec["dataset_root"]), Path(spec["manifest"]))
        training = yaml.safe_load(Path(TRAINING[spec["name"]]).read_text(encoding="utf-8"))["data"]
        if (training["dataset_root"], training["manifest"]) != (spec["dataset_root"], spec["manifest"]):
            problems.append(f"{spec['name']}: training and evaluation read different data")

    backbones = set()
    for path in TRAINING.values():
        backbones.add(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["model"]["checkpoint_path"])
    for backbone in sorted(backbones):
        present = Path(backbone).exists()
        print(f"DINOv3 backbone weights: {backbone}  {'present' if present else 'MISSING'}")
        if not present:
            problems.append(f"DINOv3 backbone weights {backbone} are missing")

    selector = yaml.safe_load(SELECTOR.read_text(encoding="utf-8"))
    run = Path(selector["localizer_run"])
    for path in (run / "checkpoints" / selector["localizer_checkpoint"], run / "config.yaml"):
        if path.is_file():
            print(f"P27 localizer: {path}  sha256 {sha256(path)}")
        else:
            problems.append(f"P27 localizer file {path} is missing (copy it from the machine that trained it)")

    f1 = yaml.safe_load(Path(selector["f1_config"]).read_text(encoding="utf-8"))
    chained = [(Path(selector["f1_config"]), selector["f1_config_sha256"]),
               (EVALUATION, f1["evaluation_config_sha256"])]
    for path, expected in chained:
        if sha256(path) != expected:
            problems.append(f"{path} was changed (sha256 {sha256(path)}, the configs expect {expected})")
    for dataset in ("levir", "dsifn", "whu"):
        pointer = Path("outputs/p29_aug_detectors") / dataset / "CHECKPOINT"
        print(f"augmented detector {dataset}: {'trained -> ' + pointer.read_text().strip() if pointer.is_file() else 'not trained yet'}")

    free = free_gpu_mib()
    print(f"free GPU memory (MiB): {free or 'unknown'}")
    if free and max(free) < MIN_FREE_GPU_MIB:
        print(f"  note: less than {MIN_FREE_GPU_MIB} MiB free; wait for the running job instead of changing the batch size")

    print("\nRESULT: " + ("ready" if not problems else "NOT READY"))
    for problem in problems:
        print(f"  - {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
