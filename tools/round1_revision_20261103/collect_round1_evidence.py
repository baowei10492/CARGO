"""CPU-only evidence collection for round-1 revision; no training or image inference.

Run manually in the server's project directory. Original assets are read only;
new metadata is written under the round-1 output root and packed with results.
Model checkpoints and complete datasets are never added to the archive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def copy_evidence(path, root):
    path = Path(path)
    if not path.is_file():
        return None
    destination = root / "source_files" / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def manifests(root):
    result = {}
    for name, path in (
        ("LEVIR-CD", "manifests/datasets/levir_cd.csv"),
        ("DSIFN-CD", "manifests/datasets/dsifn_cd_source_split.csv"),
        ("WHU-CD", "manifests/datasets/whu_cd.csv"),
        ("S2Looking", "manifests/datasets/s2looking.csv"),
        ("GZ-CD", "manifests/datasets/gzcd.csv"),
    ):
        path = Path(path)
        if not path.is_file():
            result[name] = {"missing": str(path)}
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        groups = defaultdict(set)
        for row in rows:
            groups[row["split"]].add(row.get("parent_id") or row["sample_id"])
        overlaps = {f"{a}__{b}": len(groups[a] & groups[b]) for a, b in
                    (("train", "val"), ("train", "test"), ("val", "test"))}
        train = [r for r in rows if r["split"] == "train"]
        def held(sample_id):
            key = f"3407|{sample_id}|p27-holdout|0|split".encode()
            value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**31)
            return value / (2**31) < 0.05
        result[name] = {"manifest": copy_evidence(path, root),
                        "rows_by_split": dict(Counter(r["split"] for r in rows)),
                        "unique_parents_or_sample_ids": {s: len(ids) for s, ids in groups.items()},
                        "parent_or_sample_overlaps": overlaps,
                        "localizer_train_rows_held_by_original_sample_id_rule": sum(held(r["sample_id"]) for r in train),
                        "localizer_train_rows_after_holdout": sum(not held(r["sample_id"]) for r in train)}
    save(root / "manifest_counts.json", result)
    for name, entry in result.items():
        print("DATA", name, entry.get("rows_by_split", entry), flush=True)
    return result


def selector_thresholds(root):
    output = []
    for row in ("dinov3", "bit", "cdlamba", "cdxlstm", "snunet", "bifa"):
        if row == "dinov3":
            training = Path("reports/p30/selector/20260918T104133Z")
            test = Path("reports/p30/test/20260918T134224Z/p30_test.json")
        else:
            training = next((p for p in (Path("reports/p31") / row, Path("reports/p32") / row)
                             if (p / "test/p31_test.json").is_file()), None)
            test = None if training is None else training / "test/p31_test.json"
        if training is None or test is None or not test.is_file():
            output.append({"row": row, "missing": "original test result"})
            continue
        data = json.loads(test.read_text(encoding="utf-8"))
        summary_path = training / "bundles/train_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        entries = summary.get("datasets", summary)
        copy_evidence(test, root)
        copy_evidence(summary_path, root)
        for name, test_entry in data["datasets"].items():
            entry = test_entry["proposed"] if row == "dinov3" else test_entry
            trained = entries.get(name, {})
            threshold = entry["tau"]
            if "tau" in trained and trained["tau"] != threshold:
                raise ValueError(f"training/test threshold mismatch: {row}/{name}")
            output.append({"row": row, "dataset": name, "tau": threshold,
                           "best_epoch": trained.get("best_epoch"), "records": trained.get("records"),
                           "source": str(test), "source_sha256": sha(test)})
    save(root / "selector_thresholds.json", output)
    print("THRESHOLDS_ABOVE_050", json.dumps([r for r in output if r.get("tau") is not None and r["tau"] > 0.5]), flush=True)


def model_capacity(protocol, root):
    # Restricts this process to CPU even if the instance still has a GPU attached.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from rs_cd.runtime import build_model
    from rs_cd.localizer_training import load_localizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = {}
    def counts(model):
        return {"total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters_in_saved_training_configuration": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "note": "Detector weights are frozen during CARGO evaluation; trainable count describes original model construction."}
    for name, spec in protocol["datasets"].items():
        path = Path(spec["detector"])
        try:
            actual_sha = sha(path)
            if actual_sha != spec["detector_sha256"]:
                raise ValueError("checkpoint hash mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config = payload["config"]
            save(root / "detector_configs" / f"{name}.json", config)
            model = build_model(config).cpu().eval()
            model.load_state_dict(payload["model"], strict=True)
            output[name] = {**counts(model), "checkpoint": str(path), "sha256": actual_sha,
                            "training": config.get("training", {})}
            # Some historical checkpoints store the training block under train.
            if not output[name]["training"]:
                output[name]["training"] = config.get("train", {})
            del model, payload
            gc.collect()
            print("MODEL_CAPACITY", name, output[name]["total_parameters"], flush=True)
        except Exception as error:
            output[name] = {"error": repr(error), "checkpoint": str(path)}
            print("MODEL_METADATA_WARNING", name, repr(error), flush=True)
    dsifn = Path("outputs/p15_dsifn_dinov3_partial_ft/20260728T062235Z_seed3407/checkpoints/best.pth")
    if dsifn.is_file():
        payload = torch.load(dsifn, map_location="cpu", weights_only=False)
        actual = sha(dsifn)
        expected = "c9b54ac938401de6689c082538edc2d938a7662fb00b8ebe1822bb4c627eab9e"
        if actual == expected:
            save(root / "detector_configs/DSIFN-CD.json", payload["config"])
            output["DSIFN-CD"] = {"checkpoint": str(dsifn), "sha256": actual,
                                  "training": payload["config"].get("training", payload["config"].get("train", {})),
                                  "model_config": payload["config"]["model"]}
        else:
            output["DSIFN-CD"] = {"error": "checkpoint hash mismatch", "sha256": actual}
        del payload
        gc.collect()
    try:
        model = load_localizer(protocol["localizer"]["checkpoint"], torch.device("cpu"))
        output["CARGO_registration"] = counts(model)
        del model
        gc.collect()
    except Exception as error:
        output["CARGO_registration"] = {"error": repr(error)}
    save(root / "model_capacity.json", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    project = args.project.resolve()
    os.chdir(project)
    sys.path[:0] = [str(project / "src"), str(project)]
    result_root = project / "reports/round1_20261103"
    for stage in ("E3", "E2", "E1", "E7"):
        if not (result_root / stage / "COMPLETE.json").is_file():
            raise RuntimeError(f"stage is not complete: {stage}")
    protocol = json.loads((result_root / "E3/protocol.json").read_text(encoding="utf-8"))
    evidence = result_root / "cpu_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for stage in ("E3", "E2"):
        data = json.loads((result_root / stage / "summary.json").read_text(encoding="utf-8"))
        for name, entry in data["conditions"].items():
            delta = entry["differences"]["cargo_minus_raw"]
            print(f"CI {stage} {name}: {100*delta['estimate']:+.3f} pp "
                  f"[{100*delta['ci95'][0]:+.3f}, {100*delta['ci95'][1]:+.3f}]", flush=True)
    manifests(evidence)
    selector_thresholds(evidence)
    for name in ("configs/p27_selector.yaml", "configs/p30_localizer_v2.yaml", "configs/p27_eval_perturbation.yaml",
                 "tools/build_source_crop_split.py", "tools/audit_augmentation_leakage.py",
                 "src/rs_cd/p27_selector.py", "src/rs_cd/datasets/localizer_pairs.py", "src/rs_cd/robustness.py",
                 "reports/p30/selector/20260918T104133Z/train.log", "reports/p30/selector/20260918T104133Z/features.log",
                 "outputs/p30_localizer/20260918T070115Z/config.yaml",
                 "tools/round1_revision_20261103/collect_round1_evidence.py"):
        copy_evidence(Path(name), evidence)
    model_capacity(protocol, evidence)
    archive = project / "reports/round1_revision_materials_20261103.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".partial")
    with tarfile.open(partial, "w:gz", compresslevel=3) as handle:
        handle.add(result_root, arcname=result_root.name)
    partial.replace(archive)
    checksum = sha(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(checksum + "  " + archive.name + "\n", encoding="ascii")
    print("ARCHIVE", str(archive), flush=True)
    print("ARCHIVE_BYTES", archive.stat().st_size, flush=True)
    print("ARCHIVE_SHA256", checksum, flush=True)
    print("COLLECTION_DONE: original results, metadata, and exported cases; no checkpoints or full datasets", flush=True)


if __name__ == "__main__":
    main()
