"""R2.5: two new WHU detector seeds, original training/selection recipes.

Manual server entry: .venv/bin/python -u tools/round1_revision_20261103/round1_whu_seeds.py run
No SSH, downloads, deletion, test-dependent seed selection, or global pointer edits.
Completed stages are reused. Interrupted detector training restarts from the
original pretrained backbone; the original trainer has no exact-resume support.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


SEEDS = (42, 2026)
BASE_SEED = 3407
REPORT = Path("reports/round1_20261103/WHU_detector_seeds")
OUTPUT = Path("outputs/round1_whu_detector_seeds_20261103")
ORIGINAL = Path("outputs/p8_whu_dinov3_partial_ft/20260726T145452Z_seed3407/checkpoints/best.pth")
ORIGINAL_SHA = "3805359459ade6665ba7488b1096556c09e82925ec1c40d398fdb79f61786da3"
LOCALIZER = Path("outputs/p30_localizer/20260918T070115Z/checkpoints/best.pth")
LOCALIZER_SHA = "329e7956570e3fb4950940195ba9d9700309f5cb994cdd36acb40cd598018289"
MANIFEST = Path("manifests/datasets/whu_cd.csv")
MANIFEST_SHA = "7fd30ffcc5f14c658364b6c1cb56a598b0fb0ac4893a8612f467c46ca3d1f10e"
EVALUATION = Path("configs/p27_eval_perturbation.yaml")
EVALUATION_SHA = "64fd0aed61580f1dd0b35d3f491a0db19335c98cf94735347c8e9a42cf7e99fa"
SELECTOR = Path("configs/p27_selector.yaml")
SELECTOR_SHA = "cd37272732448ebe88d42b6feb3d28b7e45f0f19db72dd28763e0d3d80dab509"
OLD_RESULT = Path("reports/p30/test/20260918T134224Z/p30_test.json")
OLD_RECORDS = OLD_RESULT.parent / "records_proposed_WHU-CD.csv"
OLD_SELECTOR = Path("reports/p30/selector/20260918T104133Z/bundles/selector_WHU-CD.pth")
OLD_SELECTOR_SHA = "eabeccad392072e728866e2ae23be2bd12fc9a1021d17dee017c9fcea7ac44dc"


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_dir(seed):
    return REPORT / f"seed{seed}"


def model_dir(seed):
    return OUTPUT / f"whu_dinov3_seed{seed}"


def device_info():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip()


def training_config(seed):
    import torch
    config = copy.deepcopy(torch.load(ORIGINAL, map_location="cpu", weights_only=False)["config"])
    expected = {"epochs": 40, "samples_per_epoch": 8192, "batch_size": 32,
                "learning_rate": 0.0003, "backbone_learning_rate": 0.00002,
                "weight_decay": 0.0001, "amp": True, "initial_checkpoint": None}
    for key, value in expected.items():
        if config["training"].get(key) != value:
            raise RuntimeError(f"original WHU recipe mismatch: training.{key}")
    if config["model"]["name"] != "dinov3_convnext_cd" or config["model"].get("trainable_stages") != 2:
        raise RuntimeError("unexpected original detector architecture")
    config["experiment"].update(name=f"whu_dinov3_seed{seed}", seed=seed, output_root=str(OUTPUT))
    # All training/data/model/loss/evaluation fields stay exactly as recorded.
    return config


def same_recipe(config, seed):
    expected = training_config(seed)
    return all(config[key] == expected[key] for key in ("data", "model", "loss", "training", "evaluation", "experiment"))


def detector_marker(seed):
    return seed_dir(seed) / "detector.json"


def completed_detector(seed):
    marker = detector_marker(seed)
    if marker.is_file():
        data = read_json(marker)
        if not Path(data["checkpoint"]).is_file() or digest(data["checkpoint"]) != data["sha256"]:
            raise RuntimeError(f"saved new-seed checkpoint identity changed: seed {seed}")
        return data
    # Recover a fully completed training if the coordinator stopped just after it.
    import torch
    for run in sorted(model_dir(seed).glob(f"*_seed{seed}"), reverse=True):
        summary = run / "summary.json"
        checkpoint = run / "checkpoints/best.pth"
        if not summary.is_file() or not checkpoint.is_file() or read_json(summary).get("status") != "completed":
            continue
        config = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        if not same_recipe(config, seed):
            raise RuntimeError(f"completed run has another recipe: {run}")
        data = {"seed": seed, "run_dir": str(run), "checkpoint": str(checkpoint),
                "sha256": digest(checkpoint), "summary": read_json(summary), "recorded_utc": stamp()}
        save_json(marker, data)
        return data
    return None


def preflight():
    import torch
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback training")
    assets = ((ORIGINAL, ORIGINAL_SHA), (LOCALIZER, LOCALIZER_SHA), (MANIFEST, MANIFEST_SHA),
              (EVALUATION, EVALUATION_SHA), (SELECTOR, SELECTOR_SHA), (OLD_SELECTOR, OLD_SELECTOR_SHA))
    for path, expected in assets:
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"missing or changed original asset: {path}")
    for path in (OLD_RECORDS, OLD_RESULT, LOCALIZER.parent.parent / "config.yaml"):
        if not path.is_file():
            raise FileNotFoundError(f"required original evidence missing: {path}")
    config = training_config(SEEDS[0])
    pretrained = Path(config["model"]["checkpoint_path"])
    if not pretrained.exists():
        raise FileNotFoundError(pretrained)
    gpu = device_info()
    if len(gpu.splitlines()) != 1:
        raise RuntimeError("expected exactly one visible GPU")
    fields = [field.strip() for field in gpu.split(",")]
    if int(fields[2]) > 2048:
        raise RuntimeError(f"GPU already in use; refusing concurrent training: {gpu}")
    free = shutil.disk_usage(Path.cwd()).free
    if free < 8 * 1024**3:
        raise RuntimeError("less than 8 GiB free; no files will be deleted automatically")
    settings = yaml.safe_load(SELECTOR.read_text(encoding="utf-8"))
    if settings["seed"] != BASE_SEED:
        raise RuntimeError("selector seed changed")
    source_paths = [Path(p) for p in ("tools/train.py", "src/rs_cd/engine.py", "src/rs_cd/runtime.py",
                    "src/rs_cd/optimization.py", "tools/run_p30.py", "tools/run_p27_selector.py",
                    "tools/train_evaluate_safe_alignment_classifier.py", "src/rs_cd/p27_selector.py",
                    "src/rs_cd/p30_geometry.py", "src/rs_cd/affine_crops.py",
                    "src/rs_cd/affine_registration.py", "src/rs_cd/localizer_training.py")]
    source_hashes = {str(p): digest(p) for p in source_paths}
    original_system = read_json(OLD_RESULT)["datasets"]["WHU-CD"]["proposed"]
    old_tau = math.inf if original_system["tau"] is None else float(original_system["tau"])
    original_scores, _, _ = read_counts(OLD_RECORDS, old_tau)
    for level in ("0", "1", "2", "3"):
        for rule, recorded in (("raw", "raw"), ("always", "estimated"), ("cargo", "selector")):
            if abs(original_scores[level][rule] - 100 * original_system["f1"][level][recorded]) > 1e-6:
                raise RuntimeError("original test record/F1 anchors do not agree")
    identity = {"protocol": "round1_WHUr2_5_seeds_v1", "new_detector_seeds": list(SEEDS),
                "original_detector_seed": BASE_SEED, "selector_seed": BASE_SEED,
                "controlled_fields": ["data", "model", "loss", "training", "evaluation"],
                "training_recipe": {key: config[key] for key in ("data", "model", "loss", "training", "evaluation")},
                "assets": {str(p): digest(p) for p, _ in assets},
                "original_records_sha256": digest(OLD_RECORDS),
                "original_result_sha256": digest(OLD_RESULT), "script_sha256": digest(__file__),
                "source_hashes": source_hashes,
                "localizer_training_config_sha256": digest(LOCALIZER.parent.parent / "config.yaml")}
    pin = REPORT / "protocol.json"
    if pin.exists() and read_json(pin) != identity:
        raise RuntimeError("protocol/script changed since this queue began; do not mix runs")
    save_json(pin, identity)
    save_json(REPORT / "environment.json", {"utc": stamp(), "gpu": gpu, "torch": torch.__version__,
              "cuda": torch.version.cuda, "free_disk_gib": free / 1024**3,
              "source_hashes": source_hashes})
    print("PREFLIGHT_OK", gpu, f"free_disk_GiB={free/1024**3:.1f}", flush=True)


def latest_metrics(seed):
    for run in sorted(model_dir(seed).glob(f"*_seed{seed}"), reverse=True):
        path = run / "metrics.jsonl"
        if path.is_file():
            records = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            if records:
                last = records[-1]
                seconds = [r["train"]["seconds"] for r in records[-3:]]
                remaining = (40 - last["epoch"]) * sum(seconds) / len(seconds) / 60
                return (f"epoch={last['epoch']}/40 samples_per_second={last['train']['samples_per_second']:.1f} "
                        f"remaining_train_min={remaining:.1f} (excludes validation, selector and test)")
    return "initializing / loading data; first epoch not yet finished"


def child(args, label, seed=None):
    log = REPORT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print("STAGE_START", label, stamp(), "log=" + str(log), flush=True)
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\nSTART {stamp()} {json.dumps(args)}\n")
        handle.flush()
        process = subprocess.Popen([sys.executable, "-u", *args], stdout=handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(20)
            message = latest_metrics(seed) if seed is not None else "running; details in stage log"
            print("PROGRESS", label, f"elapsed_min={(time.monotonic()-started)/60:.1f}", message, flush=True)
        if process.returncode:
            raise RuntimeError(f"stage {label} exited {process.returncode}; inspect {log}")
    print("STAGE_DONE", label, stamp(), f"elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)


def train(seed):
    import yaml
    if completed_detector(seed) is not None:
        print("KEEP_COMPLETED_TRAINING", seed, flush=True)
        return
    config_path = seed_dir(seed) / "detector_training.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(training_config(seed), sort_keys=False), encoding="utf-8")
    if list(model_dir(seed).glob(f"*_seed{seed}")):
        print("RESTART_INCOMPLETE_TRAINING_FROM_PRETRAINED", seed, "previous files retained", flush=True)
    child(["-m", "tools.train", "--config", str(config_path), "--device", "cuda"], f"seed{seed}_detector_train", seed)
    if completed_detector(seed) is None:
        raise RuntimeError(f"training did not produce a complete summary for seed {seed}")


def context(seed, gpu=True):
    import torch
    import yaml
    from tools import run_p30
    from tools.run_p27_selector import harness
    torch.set_num_threads(4)
    settings = yaml.safe_load(SELECTOR.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(EVALUATION.read_text(encoding="utf-8"))
    spec = next(s for s in evaluation["datasets"] if s["name"] == "WHU-CD")
    detector = read_json(detector_marker(seed))
    h = harness(evaluation)
    device = torch.device("cuda" if gpu else "cpu")
    return settings, spec, detector, h, device


def extract(seed, split):
    import torch
    import yaml
    from rs_cd.affine_crops import TileCache
    from rs_cd.localizer_training import TrainedLocalizer, load_localizer
    from rs_cd.p27_selector import crop_records
    from rs_cd.p30_geometry import BorderFreeGeometry
    from tools import run_p30
    from tools.run_p27_f1 import open_detectors
    from tools.run_p27_selector import open_dataset, write_csv
    settings, spec, det, h, device = context(seed)
    localizer = TrainedLocalizer(load_localizer(str(LOCALIZER), device), device)
    detector = open_detectors({"detectors": {"WHU-CD": {"checkpoint": det["checkpoint"], "sha256": det["sha256"]}}}, device)["WHU-CD"]
    dataset = open_dataset(spec, "train" if split == "features" else "test")
    if split == "features":
        held = yaml.safe_load((LOCALIZER.parent.parent / "config.yaml").read_text(encoding="utf-8"))
        items = run_p30.training_items(dataset, spec, settings, held, h["size"], 64)
        counts = {name: 4 * sum(s == name for _, s in items) for name in ("fit", "early_stop", "risk_calibration")}
        if counts != {"fit": 10272, "early_stop": 1728, "risk_calibration": 672}:
            raise RuntimeError(f"selector split differs from original WHU records: {counts}")
    else:
        crops = run_p30.margin_crops(dataset, spec, h["size"], 64)
        if len(crops) != 2552:
            raise RuntimeError("WHU test crop count changed")
        items = [(crop, "test") for crop in crops]
    cache = TileCache(dataset, int(spec["tile_size"]))
    root = seed_dir(seed) / f"{split}_crop_records"
    root.mkdir(parents=True, exist_ok=True)
    identity = {"detector_sha256": det["sha256"], "localizer_sha256": LOCALIZER_SHA,
                "selector_config_sha256": SELECTOR_SHA, "evaluation_config_sha256": EVALUATION_SHA,
                "stage": split, "items": [{**item, "split": s} for item, s in items],
                "script_sha256": digest(__file__)}
    if (root / "identity.json").exists() and read_json(root / "identity.json") != identity:
        raise RuntimeError("per-crop resume identity mismatch")
    save_json(root / "identity.json", identity)
    records = []
    started = time.monotonic()
    for index, (item, assignment) in enumerate(items, 1):
        path = root / (hashlib.sha256(item["crop_id"].encode()).hexdigest() + ".json")
        if path.exists():
            crop = read_json(path)
        else:
            t1, t2, mask = cache.crop(item, h["size"])
            geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], h["size"])
            _, crop = crop_records(localizer, detector["model"], threshold=detector["threshold"],
                loss_weights=detector["loss"], dataset="WHU-CD", item=item, t1=t1, t2=t2, mask=mask,
                mask_threshold=int(spec["mask_threshold"]), ranges=h["ranges"], levels=h["levels"],
                seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
                trigger_probability=h["trigger_probability"], device=device, geometry=geometry)
            for row in crop:
                row["split"] = assignment
            save_json(path, crop)
        if len(crop) != 4 or any(r["crop_id"] != item["crop_id"] for r in crop):
            raise RuntimeError("invalid saved crop record")
        records.extend(crop)
        if index % 100 == 0 or index == len(items):
            elapsed = time.monotonic() - started
            eta = elapsed / index * (len(items) - index) / 60
            print(f"PROGRESS {split} seed={seed} crops={index}/{len(items)} elapsed_min={elapsed/60:.1f} eta_min={eta:.1f}", flush=True)
    if split == "features":
        write_csv(seed_dir(seed) / "features_WHU-CD.csv", records)
        save_json(seed_dir(seed) / "features_COMPLETE.json", {"records": len(records), "detector_sha256": det["sha256"], "utc": stamp()})
    else:
        bundle = torch.load(seed_dir(seed) / "selector_WHU-CD.pth", map_location="cpu", weights_only=False)
        if bundle["detector_sha256"] != det["sha256"]:
            raise RuntimeError("selector belongs to another detector")
        system = run_p30.system_table(records, bundle)
        write_csv(seed_dir(seed) / "records_WHU-CD.csv", records)
        save_json(seed_dir(seed) / "test_WHU-CD.json", {"seed": seed, "dataset": "WHU-CD", "crops": len(items),
                   "detector_sha256": det["sha256"], "localizer_sha256": LOCALIZER_SHA, **system})
        print("TEST_COMPLETE", seed, json.dumps(system["f1"]), flush=True)


def selector(seed):
    import torch
    from tools.run_p27_selector import read_records, train_dataset
    settings, spec, det, h, device = context(seed, gpu=False)
    bundle, summary = train_dataset("WHU-CD", read_records(seed_dir(seed) / "features_WHU-CD.csv"), settings)
    bundle.update(protocol="round1_detector_seed_repeat", detector_seed=seed, detector_sha256=det["sha256"],
                  localizer_sha256=LOCALIZER_SHA, config_sha256=SELECTOR_SHA, selector_seed=BASE_SEED)
    temporary = seed_dir(seed) / "selector_WHU-CD.partial.pth"
    torch.save(bundle, temporary)
    temporary.replace(seed_dir(seed) / "selector_WHU-CD.pth")
    save_json(seed_dir(seed) / "selector_train.json", summary)
    print("SELECTOR_COMPLETE", seed, "tau", bundle["tau"], "best_epoch", bundle["best_epoch"], flush=True)


def read_counts(path, tau):
    import numpy as np
    from rs_cd.robustness import _stable_seed
    totals = {str(level): np.zeros((3, 4), dtype=np.int64) for level in range(4)}
    totals["mixed"] = np.zeros((3, 4), dtype=np.int64)
    identity, adopted = {}, {str(k): 0 for k in range(4)}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            level = int(row["level"])
            key = (row["crop_id"], level)
            if key in identity:
                raise RuntimeError("duplicate test record")
            raw, estimated = [np.array([int(row[f"{p}_{k}"]) for k in ("tp", "fp", "fn", "tn")], dtype=np.int64)
                              for p in ("raw", "estimated")]
            if raw.sum() != estimated.sum():
                raise RuntimeError("different scoring regions across paths")
            identity[key] = (row["tile_id"], int(raw.sum()))
            take = float(row["selector_probability"]) >= tau
            counts = np.stack((raw, estimated, estimated if take else raw))
            totals[str(level)] += counts
            adopted[str(level)] += int(take)
            assigned = (0, 1, 3)[_stable_seed(BASE_SEED, row["crop_id"], "p32-mixed", 0, "state") % 3]
            if level == assigned:
                totals["mixed"] += counts
    if len(identity) != 4 * 2552:
        raise RuntimeError("incomplete test records")
    def f1(c):
        denominator = 2 * c[:, 0] + c[:, 1] + c[:, 2]
        return np.divide(2 * c[:, 0], denominator, out=np.zeros(3, dtype=float), where=denominator != 0)
    scores = {level: dict(zip(("raw", "always", "cargo"), (100 * f1(value)).tolist())) for level, value in totals.items()}
    for value in scores.values():
        value["cargo_minus_raw_pp"] = value["cargo"] - value["raw"]
        value["cargo_minus_always_pp"] = value["cargo"] - value["always"]
    return scores, identity, adopted


def summarize():
    import numpy as np
    old = read_json(OLD_RESULT)["datasets"]["WHU-CD"]["proposed"]
    sources = [(BASE_SEED, OLD_RECORDS, old)]
    sources += [(seed, seed_dir(seed) / "records_WHU-CD.csv", read_json(seed_dir(seed) / "test_WHU-CD.json")) for seed in SEEDS]
    rows, reference = [], None
    for seed, path, anchor in sources:
        tau = math.inf if anchor["tau"] is None else float(anchor["tau"])
        scores, identity, adopted = read_counts(path, tau)
        if reference is not None and reference != identity:
            raise RuntimeError("test crop identities/scored pixels differ across detector seeds")
        reference = identity
        for level in ("0", "1", "2", "3"):
            for rule, original in (("raw", "raw"), ("always", "estimated"), ("cargo", "selector")):
                if abs(scores[level][rule] - 100 * anchor["f1"][level][original]) > 1e-6:
                    raise RuntimeError(f"F1 anchor mismatch: seed {seed} / {level} / {rule}")
        rows.append({"detector_seed": seed, "selector_seed": BASE_SEED,
                     "tau": None if math.isinf(tau) else tau, "f1_percent": scores,
                     "adoption_percent": {k: 100 * n / 2552 for k, n in adopted.items()},
                     "records_sha256": digest(path)})
    aggregates = {level: {metric: {"mean": float(np.mean([r["f1_percent"][level][metric] for r in rows])),
                                  "sample_std": float(np.std([r["f1_percent"][level][metric] for r in rows], ddof=1))}
                          for metric in rows[0]["f1_percent"][level]} for level in rows[0]["f1_percent"]}
    save_json(REPORT / "summary_3seeds.json", {"dataset": "WHU-CD", "detector": "DINOv3", "seeds": rows,
              "mean_and_sample_std": aggregates, "n_training_seeds": 3,
              "scope": "three detector-training seeds for one combination; fixed registration checkpoint; selector refitted per detector with fixed seed 3407; same test crops and perturbations; seed SD is separate from tile-bootstrap CI"})
    lines = ["DINOv3 + WHU-CD detector-training seed repeat; F1 %, gains pp.", "seed  level  raw  always  CARGO  gain  tau"]
    for row in rows:
        for level in ("0", "3", "mixed"):
            v = row["f1_percent"][level]
            lines.append(f"{row['detector_seed']} {level} {v['raw']:.3f} {v['always']:.3f} {v['cargo']:.3f} {v['cargo_minus_raw_pp']:+.3f} {row['tau']}")
    for level in ("0", "3", "mixed"):
        v = aggregates[level]["cargo_minus_raw_pp"]
        lines.append(f"Mean gain {level}: {v['mean']:+.3f} +/- {v['sample_std']:.3f} pp (sample SD, n=3)")
    (REPORT / "summary_3seeds.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def run():
    import fcntl
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("this seed queue is already running")
        print("QUEUE_START", stamp(), "checking saved assets and recipe; no smoke run", flush=True)
        save_json(REPORT / "queue_status.json", {"status": "running", "utc": stamp(), "pid": os.getpid()})
        try:
            preflight()
            for seed in SEEDS:
                train(seed)
                for stage, marker in (("features", "features_COMPLETE.json"), ("selector", "selector_train.json"), ("test", "test_WHU-CD.json")):
                    if (seed_dir(seed) / marker).exists():
                        print("KEEP_COMPLETED_STAGE", seed, stage, flush=True)
                        continue
                    child([str(Path(__file__).resolve()), stage, "--seed", str(seed)], f"seed{seed}_{stage}")
            child([str(Path(__file__).resolve()), "summarize"], "summary_3seeds")
            save_json(REPORT / "COMPLETE.json", {"utc": stamp(), "seeds": [BASE_SEED, *SEEDS], "status": "completed"})
            save_json(REPORT / "queue_status.json", {"status": "completed", "utc": stamp()})
            print("QUEUE_DONE", stamp(), REPORT, flush=True)
        except Exception as error:
            save_json(REPORT / "queue_status.json", {"status": "failed", "utc": stamp(), "error": str(error)})
            print("QUEUE_FAILED", str(error), flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "features", "selector", "test", "summarize"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "run":
        run()
    elif args.command == "summarize":
        summarize()
    elif args.seed is None:
        parser.error("--seed is required for per-seed stages")
    elif args.command == "selector":
        selector(args.seed)
    else:
        extract(args.seed, args.command)


if __name__ == "__main__":
    main()
