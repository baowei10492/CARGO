"""P31 plug-in rows under the border-free protocol (docs/P31_BORDER_FREE_REMAINDER.md, configs/p31_border_free.yaml).

    features  --row R                  selector-training records: border-free training crops, localizer v2, R's detectors
    train     --row R                  R's selectors (configs/p27_selector.yaml)
    evaluate  --row R --split val|test without correction vs localizer + selector; the test once, after validation
    all       --row R                  the four steps in order; finished steps are skipped

The P30 machinery (tools/run_p30.py) does the work; only the detectors differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from tools import run_p30
from tools.run_p27_f1 import open_detectors
from tools.run_p27_selector import SPLITS, check_sources, harness, load_bundles, open_dataset, with_split, write_csv
from tools.run_p27_step1 import sha256

ROOT = Path("reports/p31")
LEVELS = ("0", "1", "2", "3")


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(P31 config, P30 config, evaluation config, selector recipe), every chained hash checked."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("the P31 config is not marked preregistered")
    if sha256(Path(config["p30_config"])) != config["p30_config_sha256"]:
        raise SystemExit(f"{config['p30_config']} was changed")
    for file, digest in config["config_sha256"].items():
        if sha256(Path(file)) != digest:
            raise SystemExit(f"{file} was changed")
    p30, evaluation, selector = run_p30.load(Path(config["p30_config"]))
    config["_sha256"] = sha256(path)
    return config, p30, evaluation, selector


def row_detector(config: dict[str, Any], row: str, name: str, device: torch.device) -> dict[str, Any]:
    """The frozen P27 plug-in detector of ``row`` for dataset ``name`` (P21 cells or checkpoint pointers)."""
    plugin = yaml.safe_load(Path(config["rows"][row][name]).read_text(encoding="utf-8"))
    wrapper: dict[str, Any] = {"detectors": {name: plugin["detectors"][name]}}
    if plugin.get("p21_freeze_manifest"):
        wrapper["p21_freeze_manifest"] = plugin["p21_freeze_manifest"]
    return open_detectors(wrapper, device)[name]


def release(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_features(config: dict[str, Any], p30: dict[str, Any], evaluation: dict[str, Any], selector: dict[str, Any],
                 row: str, device: torch.device, output_dir: Path, *, max_crops: int | None = None) -> dict[str, Any]:
    localizer, localizer_config = run_p30.open_localizer(p30, "p30", device)
    check_sources(evaluation, localizer_config)
    h = harness(evaluation)
    margin = int(p30["crop_margin"])
    summary: dict[str, Any] = {"row": row, "localizer_sha256": sha256(run_p30.localizer_checkpoint(p30, "p30")),
                               "selector_config_sha256": p30["selector_config_sha256"], "crop_margin": margin,
                               "datasets": {}}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        dataset = open_dataset(spec, selector["training_crops"]["split"])
        items = run_p30.training_items(dataset, spec, selector, localizer_config, h["size"], margin)
        items = items[:max_crops] if max_crops is not None else items
        detector = row_detector(config, row, name, device)
        records = run_p30.records_for(dataset, spec, items, localizer, detector, h, device, f"{row} features")
        write_csv(output_dir / f"features_{name}.csv", records)
        summary["datasets"][name] = {"tiles": len(dataset.rows),
                                     "crops": {split: sum(1 for _, s in items if s == split) for split in SPLITS},
                                     "records": len(records), "detector_sha256": detector["sha256"]}
        del detector
        release(device)
    (output_dir / "features_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def components(config: dict[str, Any], p30: dict[str, Any], row: str, names: list[str], bundle_dir: Path) -> dict[str, Any]:
    return {"p31_config_sha256": config["_sha256"], "row": row,
            "localizer": sha256(run_p30.localizer_checkpoint(p30, "p30")),
            "selectors": {name: sha256(bundle_dir / f"selector_{name}.pth") for name in names},
            "detector_configs": {name: config["config_sha256"][config["rows"][row][name]] for name in names}}


def run_evaluate(config: dict[str, Any], p30: dict[str, Any], evaluation: dict[str, Any], row: str, split: str,
                 bundle_dir: Path, device: torch.device, output_dir: Path, *, max_crops: int | None = None) -> dict[str, Any]:
    evaluation = with_split(evaluation, split)
    h = harness(evaluation)
    margin = int(p30["crop_margin"])
    names = [spec["name"] for spec in evaluation["datasets"]]
    localizer, _ = run_p30.open_localizer(p30, "p30", device)
    localizer_sha = sha256(run_p30.localizer_checkpoint(p30, "p30"))
    bundles, _ = load_bundles(bundle_dir, names)
    for name, bundle in bundles.items():
        if bundle.get("protocol") != "p30" or bundle["localizer_sha256"] != localizer_sha:
            raise SystemExit(f"{row} {name}: the selector was trained for another localizer or protocol")
    decision: dict[str, Any] = {"row": row, "split": split, "crop_margin": margin, "datasets": {}}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        dataset = open_dataset(spec, split)
        crops = run_p30.margin_crops(dataset, spec, h["size"], margin)
        crops = crops[:max_crops] if max_crops is not None else crops
        detector = row_detector(config, row, name, device)
        records = run_p30.records_for(dataset, spec, [(crop, split) for crop in crops], localizer, detector, h, device,
                                      f"{row} {split}")
        system = run_p30.system_table(records, bundles[name])
        entry = {"crops": len(crops), "detector_sha256": detector["sha256"], **system,
                 "without_correction": {level: values["raw"] for level, values in system["f1"].items()},
                 "with_selector": {level: values["selector"] for level, values in system["f1"].items()}}
        decision["datasets"][name] = entry
        write_csv(output_dir / f"records_{name}.csv", records)
        print(render(row, name, entry), flush=True)
        del detector
        release(device)
    return decision


def render(row: str, name: str, entry: dict[str, Any]) -> str:
    success = entry["localization_success"]
    return "\n".join([
        f"[{row} {name}] {entry['crops']} crops; F1 % at clean / L1 / L2 / L3",
        "  without correction    " + " / ".join(f"{entry['without_correction'][l] * 100:5.1f}" for l in LEVELS),
        "  localizer + selector  " + " / ".join(f"{entry['with_selector'][l] * 100:5.1f}" for l in LEVELS),
        f"  tau {entry['tau']}; localization success L1-L3 " + " / ".join(f"{success[l] * 100:.1f}" for l in LEVELS[1:])
        + f"; corrected L0/L3 {entry['selection_rate']['0']['selector']:.3f}/{entry['selection_rate']['3']['selector']:.3f}"])


def step_features(args, config, p30, evaluation, selector, device, root: Path) -> None:
    out = root / "features"
    if (out / "features_summary.json").is_file():
        print(f"{args.row}: features exist ({out})")
        return
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_features(config, p30, evaluation, selector, args.row, device, out), indent=2))


def step_train(args, config, p30, evaluation, selector, root: Path) -> None:
    out = root / "bundles"
    if (out / "train_summary.json").is_file():
        print(f"{args.row}: selectors exist ({out})")
        return
    out.mkdir(parents=True, exist_ok=True)
    run_p30.run_train(p30, evaluation, selector, root / "features", out)


def step_evaluate(args, split: str, config, p30, evaluation, device, root: Path) -> None:
    out = root / split
    if (out / f"p31_{split}.json").is_file():
        print(f"{args.row}: {split} exists ({out})")
        return
    bundle_dir = root / "bundles"
    names = [spec["name"] for spec in evaluation["datasets"]]
    identity = components(config, p30, args.row, names, bundle_dir)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    marker = run_p30.claim(split, digest, {"split": split, "output_dir": str(out), **identity},
                           marker_dir=root / "markers")
    decision = run_evaluate(config, p30, evaluation, args.row, split, bundle_dir, device, out)
    decision.update(components=identity, marker=str(marker), test_split_touched=split == "test")
    (out / f"p31_{split}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "train", "evaluate", "all"))
    parser.add_argument("--row", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config", default="configs/p31_border_free.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config, p30, evaluation, selector = load(Path(args.config))
    if args.row not in config["rows"]:
        raise SystemExit(f"unknown row {args.row}; rows: {sorted(config['rows'])}")
    device = torch.device(args.device)
    # P32 rows keep their own directory; the frozen P31 configs name none.
    root = Path(config.get("report_root", ROOT)) / args.row
    if args.command in ("features", "all"):
        step_features(args, config, p30, evaluation, selector, device, root)
    if args.command in ("train", "all"):
        step_train(args, config, p30, evaluation, selector, root)
    if args.command == "evaluate":
        step_evaluate(args, args.split, config, p30, evaluation, device, root)
    if args.command == "all":
        for split in ("val", "test"):   # both reported; nothing is chosen on validation
            step_evaluate(args, split, config, p30, evaluation, device, root)


if __name__ == "__main__":
    main()
