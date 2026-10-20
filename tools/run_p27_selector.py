"""P27 selector step (docs/P27_SELECTOR_AMENDMENT.md).

features  training splits only: one record per crop and level for every dataset,
          from the frozen P27 localizer and the frozen DINOv3 detectors
train     one P7-3 classifier per dataset (fit / early stop on non-hold-out
          tiles), threshold from the localizer's hold-out tiles; prints the
          calibration table. No validation data.
evaluate  validation crops, once: the selector next to raw / oracle / gated;
          the four F1-step paths must reproduce the F1-step table first.
test      test crops, once, with the selectors that went through evaluate
          (configs/p27_test.yaml); reporting only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from rs_cd.affine_crops import TileCache, select_evenly
from rs_cd.affine_f1 import f1_decision
from rs_cd.affine_registration import AffineRanges, grid_points
from rs_cd.datasets import PairedChangeDataset
from rs_cd.datasets.localizer_pairs import in_holdout
from rs_cd.localizer_training import TrainedLocalizer, load_localizer
from rs_cd.p27_selector import (
    COUNT_KEYS,
    FEATURE_NAMES,
    RULES,
    calibrate_tau,
    crop_records,
    f1_table,
    feature_matrix,
    sample_weights,
    selection_rates,
    selector_probability,
    selector_split,
)
from tools.run_p27_f1 import PLUGIN_MARKER_ROOT, all_crops, marker_dir, open_detectors
from tools.run_p27_f1 import load_config as load_f1_config
from tools.run_p27_step1 import claim, sha256

SELECTOR_MARKER_DIR = Path("reports/p27_localizer/selector/frozen")
TEST_MARKER_DIR = Path("reports/p27_localizer/test/frozen")
SPLITS = ("fit", "early_stop", "risk_calibration")
ANCHOR_PATHS = {"raw": "raw", "oracle": "oracle", "estimated": "estimated", "gated": "estimated_gated"}
LOSS_KEYS = ("bce_weight", "dice_weight", "pos_weight")


# ---------------------------------------------------------------- setup
def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") != "preregistered":
        raise SystemExit("P27 selector config is not marked preregistered")
    f1_path = Path(config["f1_config"])
    digest = sha256(f1_path)
    if digest != config["f1_config_sha256"]:
        raise SystemExit(f"{f1_path} has sha256 {digest}, the selector config expects {config['f1_config_sha256']}")
    f1_config, evaluation = load_f1_config(f1_path)
    return config, f1_config, evaluation


def localizer_files(config: dict[str, Any]) -> tuple[Path, Path]:
    run = Path(config["localizer_run"])
    return run / "checkpoints" / config["localizer_checkpoint"], run / "config.yaml"


def load_detectors(f1_config: dict[str, Any], device: torch.device) -> dict[str, dict[str, Any]]:
    return open_detectors(f1_config, device)


def step_marker_dirs(f1_config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """(F1-step markers, selector markers, test markers) for this detector row."""

    namespace = f1_config.get("marker_namespace")
    if namespace is None:
        return marker_dir(f1_config), SELECTOR_MARKER_DIR, TEST_MARKER_DIR
    root = PLUGIN_MARKER_ROOT / str(namespace)
    return marker_dir(f1_config), root / "selector_frozen", root / "test_frozen"


def reference_path(value: str) -> Path:
    """The F1 decision file, directly or through the LATEST pointer of an F1 report root."""

    path = Path(value)
    if path.name == "LATEST":
        return Path(path.read_text(encoding="utf-8").strip()) / "run" / "f1_decision.json"
    return path


def check_reference(reference: dict[str, Any], localizer_sha: str, detectors: dict[str, dict[str, Any]]) -> None:
    """The F1 run this step follows used the same localizer and detectors."""

    if reference["frozen"]["checkpoint_sha256"] != localizer_sha:
        raise SystemExit("the F1 reference run used a different localizer")
    for name, detector in detectors.items():
        if reference["datasets"][name]["detector"]["sha256"] != detector["sha256"]:
            raise SystemExit(f"{name}: the F1 reference run used a different detector")


def harness(evaluation: dict[str, Any]) -> dict[str, Any]:
    perturbation, settings = evaluation["perturbation"], evaluation["step1"]
    size = int(perturbation["crop_size"])
    levels = [int(level) for level in perturbation["levels"]]
    low, high = perturbation["scale_range"]
    return {
        "size": size,
        "levels": levels,
        "ranges": AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels)),
        "points": grid_points(size, size, int(settings["grid_points"])),
        "seed": int(evaluation["seed"]),
        "tolerance": float(settings["success_ate_px"]),
        "trigger_probability": float(evaluation["p27"]["trigger_probability"]),
    }


def open_dataset(spec: dict[str, Any], split: str) -> PairedChangeDataset:
    return PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=split, normalization="unit",
                               mask_threshold=int(spec["mask_threshold"]))


def records_for(dataset: PairedChangeDataset, spec: dict[str, Any], items: Sequence[tuple[dict[str, Any], str]],
                localizer: TrainedLocalizer, detector: dict[str, Any], h: dict[str, Any],
                device: torch.device, label: str) -> list[dict[str, Any]]:
    cache = TileCache(dataset, int(spec["tile_size"]))
    records = []
    for index, (item, split) in enumerate(items, start=1):
        t1, t2, mask = cache.crop(item, h["size"])
        _, crop = crop_records(
            localizer, detector["model"], threshold=detector["threshold"], loss_weights=detector["loss"],
            dataset=spec["name"], item=item, t1=t1, t2=t2, mask=mask, mask_threshold=int(spec["mask_threshold"]),
            ranges=h["ranges"], levels=h["levels"], seed=h["seed"], points=h["points"], tolerance=h["tolerance"],
            trigger_probability=h["trigger_probability"], device=device,
        )
        for record in crop:
            record["split"] = split
        records.extend(crop)
        if index % 200 == 0 or index == len(items):
            print(f"[p27 selector {label} {spec['name']} {index}/{len(items)}]", flush=True)
    return records


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_records(path: Path) -> list[dict[str, Any]]:
    integers = {"level", "valid_pixels", *(f"{p}_{k}" for p in ("raw", "oracle", "estimated") for k in COUNT_KEYS)}
    floats = {*FEATURE_NAMES, "change_fraction", "raw_loss", "estimated_loss", "advantage"}
    records = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record: dict[str, Any] = dict(row)
            for key in integers:
                record[key] = int(row[key])
            for key in floats:
                record[key] = float(row[key])
            record["trigger"] = row["trigger"] == "True"
            records.append(record)
    return records


# ------------------------------------------------------------- features
def training_items(dataset: PairedChangeDataset, spec: dict[str, Any], config: dict[str, Any],
                   localizer_config: dict[str, Any], size: int) -> list[tuple[dict[str, Any], str]]:
    fraction, seed = float(localizer_config["holdout"]["fraction"]), int(localizer_config["seed"])
    held_rows = [in_holdout(row.sample_id, fraction, seed) for row in dataset.rows]
    crops = all_crops(dataset, int(spec["tile_size"]), size)
    held = [crop for crop in crops if held_rows[crop["row"]]]
    rest = [crop for crop in crops if not held_rows[crop["row"]]]
    rest = select_evenly(rest, int(config["training_crops"]["max_fit_crops_per_dataset"])) if rest else []
    fraction_early = float(config["training_crops"]["early_stop_fraction"])
    return [
        (crop, selector_split(crop["sample_id"], holdout=held_rows[crop["row"]], seed=int(config["seed"]),
                              early_stop_fraction=fraction_early))
        for crop in held + rest
    ]


def check_sources(evaluation: dict[str, Any], localizer_config: dict[str, Any]) -> None:
    sources = {source["name"]: source for source in localizer_config["sources"]}
    for spec in evaluation["datasets"]:
        source = sources.get(spec["name"])
        if source is None or any(str(source[key]) != str(spec[key])
                                 for key in ("dataset_root", "manifest", "tile_size", "mask_threshold")):
            raise SystemExit(f"{spec['name']}: the localizer was trained on a different source definition")


def run_features(config: dict[str, Any], evaluation: dict[str, Any], localizer: TrainedLocalizer,
                 detectors: dict[str, dict[str, Any]], localizer_config: dict[str, Any], device: torch.device,
                 output_dir: Path, *, max_crops: int | None = None) -> dict[str, Any]:
    check_sources(evaluation, localizer_config)
    h = harness(evaluation)
    summary = {}
    for spec in evaluation["datasets"]:
        dataset = open_dataset(spec, config["training_crops"]["split"])
        items = training_items(dataset, spec, config, localizer_config, h["size"])
        if max_crops is not None:
            items = items[:max_crops]
        records = records_for(dataset, spec, items, localizer, detectors[spec["name"]], h, device, "features")
        write_csv(output_dir / f"features_{spec['name']}.csv", records)
        summary[spec["name"]] = {
            "tiles": len(dataset.rows),
            "crops": {split: sum(1 for _, s in items if s == split) for split in SPLITS},
            "records": len(records),
            "detector_sha256": detectors[spec["name"]]["sha256"],
        }
    return summary


# ---------------------------------------------------------------- train
def train_dataset(name: str, records: list[dict[str, Any]], config: dict[str, Any],
                  feature_names: Sequence[str] = FEATURE_NAMES) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.train_evaluate_safe_alignment_classifier import train_safe_alignment_classifier

    advantage = torch.tensor([r["advantage"] for r in records], dtype=torch.float32)
    clean = torch.tensor([r["level"] == 0 for r in records], dtype=torch.bool)
    safe, weight = sample_weights(advantage, clean, config)
    splits = [r["split"] for r in records]
    masks = {split: torch.tensor([s == split for s in splits], dtype=torch.bool) for split in SPLITS}
    empty = [split for split, mask in masks.items() if not bool(mask.any())]
    if empty:
        raise SystemExit(f"{name}: empty split(s) {empty}")
    prepared = {
        "features": feature_matrix(records, feature_names), "benefit": advantage, "clean": clean,
        "q_gate": torch.tensor([r["trigger"] for r in records], dtype=torch.bool),
        "safe": safe, "weight": weight, "splits": splits, "split_masks": masks,
    }
    protocol = {"seed": int(config["seed"]), "classifier": config["classifier"], "risk": config["risk"]}
    result = train_safe_alignment_classifier(
        f"selector_{name}", [{"scenario": r["scenario"]} for r in records], prepared, tuple(feature_names),
        protocol=protocol, output_dir=None, feature_universe=tuple(feature_names),
    )
    probability = [float(p) for p in result["probability"]]
    calibration = [(r, p) for r, p in zip(records, probability) if r["split"] == "risk_calibration"]
    cal_records, cal_probability = [r for r, _ in calibration], [p for _, p in calibration]
    spec = config["calibration"]
    tau, tau_table = calibrate_tau(cal_records, cal_probability, target=float(spec["clean_drop_target"]),
                                   start=float(spec["tau_grid_start"]), stop=float(spec["tau_grid_stop"]),
                                   step=float(spec["tau_grid_step"]))
    table = f1_table(cal_records, cal_probability, tau)
    bundle = {
        "dataset": name,
        "feature_names": list(feature_names),
        "feature_mean": result["feature_mean"],
        "feature_std": result["feature_std"],
        "hidden_dims": [int(d) for d in result["hidden_dims"]],
        "dropout": float(result["dropout"]),
        "model_state": result["model_state"],
        "tau": None if math.isinf(tau) else tau,
        "best_epoch": int(result["best_epoch"]),
    }
    levels = [key for key in table if key != "0"]
    summary = {
        "records": {split: int(mask.sum()) for split, mask in masks.items()},
        "safe_rate": {split: float(safe[mask].float().mean()) for split, mask in masks.items()},
        "best_epoch": bundle["best_epoch"],
        "tau": bundle["tau"],
        "never_correct": math.isinf(tau),
        "p7_rule_tau": float(result["probability_threshold"]),
        "calibration_f1": table,
        "calibration_selection": selection_rates(cal_records, cal_probability, tau),
        "calibration_clean_drop": {rule: table["0"]["raw"] - table["0"][rule] for rule in RULES},
        "calibration_gain": {level: {rule: table[level][rule] - table[level]["raw"] for rule in RULES}
                             for level in levels},
        "tau_table": tau_table,
        "fit_summary": result["summary"]["fit"],
        "early_stop_summary": result["summary"]["early_stop"],
    }
    summary["selector_gains_at_levels_2_3"] = all(summary["calibration_gain"][level]["selector"] > 0
                                                  for level in ("2", "3") if level in summary["calibration_gain"])
    return bundle, summary


def render_calibration(summaries: dict[str, Any]) -> str:
    lines = []
    for name, s in summaries.items():
        lines.append(f"\n[{name}] tau {s['tau']} (never_correct {s['never_correct']}), best epoch {s['best_epoch']}, "
                     f"records {s['records']}")
        lines.append("  calibration F1  level  " + "  ".join(f"{rule:>13}" for rule in RULES))
        for level, row in s["calibration_f1"].items():
            lines.append(f"                 {level:>5}  " + "  ".join(f"{row[rule]:13.4f}" for rule in RULES))
        lines.append("  selection rate  level  " + "  ".join(f"{rule:>13}" for rule in s["calibration_selection"]["0"]))
        for level, row in s["calibration_selection"].items():
            lines.append(f"                 {level:>5}  " + "  ".join(f"{value:13.3f}" for value in row.values()))
        lines.append(f"  clean drop: selector {s['calibration_clean_drop']['selector']:+.4f}, "
                     f"gated {s['calibration_clean_drop']['gated']:+.4f}; "
                     f"selector gains at levels 2 and 3: {s['selector_gains_at_levels_2_3']}")
    return "\n".join(lines)


# ------------------------------------------------------------- evaluate
def anchor_check(table: dict[str, dict[str, float]], reference: dict[str, dict[str, float]],
                 tolerance: float) -> dict[str, Any]:
    differences = {
        f"{level}/{ours}": abs(table[level][ours] - reference[level][theirs])
        for level in table for ours, theirs in ANCHOR_PATHS.items()
    }
    largest = max(differences.values())
    return {"max_abs_difference": largest, "tolerance": tolerance, "passed": largest <= tolerance,
            "differences": differences}


def load_bundles(bundle_dir: Path, names: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    bundles, digests = {}, {}
    for name in names:
        path = bundle_dir / f"selector_{name}.pth"
        digests[name] = sha256(path)
        bundles[name] = torch.load(path, map_location="cpu", weights_only=False)
        if bundles[name]["dataset"] != name:
            raise SystemExit(f"{path} was trained for {bundles[name]['dataset']}")
    return bundles, digests


def run_evaluate(config: dict[str, Any], f1_config: dict[str, Any], evaluation: dict[str, Any],
                 localizer: TrainedLocalizer, detectors: dict[str, dict[str, Any]], bundles: dict[str, dict[str, Any]],
                 reference: dict[str, Any] | None, device: torch.device, output_dir: Path, *,
                 max_crops: int | None = None, split: str = "val") -> dict[str, Any]:
    """Frozen selectors on the ``split`` crops; ``reference`` (the F1-step decision) anchors validation only."""

    h = harness(evaluation)
    rule = f1_config["decision"]
    decision: dict[str, Any] = {"gate": "p27_selector" if split == "val" else f"p27_{split}", "datasets": {},
                                "split": split, "test_split_touched": split == "test"}
    all_records = []
    for spec in evaluation["datasets"]:
        name = spec["name"]
        if spec["split"] != split:
            raise SystemExit(f"{name}: dataset split {spec['split']} is not {split}")
        dataset = open_dataset(spec, spec["split"])
        crops = all_crops(dataset, int(spec["tile_size"]), h["size"])
        if max_crops is not None:
            crops = crops[:max_crops]
        records = records_for(dataset, spec, [(crop, split) for crop in crops], localizer, detectors[name], h,
                              device, "evaluate" if split == "val" else split)
        bundle = bundles[name]
        tau = math.inf if bundle["tau"] is None else float(bundle["tau"])
        probability = selector_probability(bundle, feature_matrix(records, bundle["feature_names"])).tolist()
        table = f1_table(records, probability, tau)
        anchor = None
        if reference is not None:
            anchor = anchor_check(table, reference["datasets"][name]["f1"], float(config["anchor_tolerance"]))
            if not anchor["passed"]:
                decision["datasets"][name] = {"verdict": "anchor_failed", "anchor": anchor}
                continue
        verdict = f1_decision(
            table, levels=h["levels"], primary=config["evaluation"]["primary_path"],
            binding_levels=[int(level) for level in rule["binding_levels"]], min_gain=float(rule["min_gain"]),
            max_clean_drop=float(rule["max_clean_drop"]),
            min_clean_raw_f1=float(f1_config["instrument"]["min_clean_raw_f1"]),
        )
        decision["datasets"][name] = {
            **verdict,
            "anchor": anchor,
            "f1": table,
            "selection_rate": selection_rates(records, probability, tau),
            "tau": bundle["tau"],
            "crops": len(crops),
            "clean_drop_by_rule": {r: table["0"]["raw"] - table["0"][r] for r in RULES},
            "gain_by_rule": {level: {r: table[level][r] - table[level]["raw"] for r in RULES}
                             for level in table if level != "0"},
        }
        for record, p in zip(records, probability):
            record["selector_probability"] = p
        all_records.extend(records)
    decision["verdicts"] = {name: entry["verdict"] for name, entry in decision["datasets"].items()}
    if all(entry["verdict"] != "anchor_failed" for entry in decision["datasets"].values()):
        filename = "validation_records.csv" if split == "val" else f"{split}_records.csv"
        write_csv(output_dir / filename, all_records)
    return decision


def render_evaluation(decision: dict[str, Any]) -> str:
    lines = [f"split: {decision.get('split', 'val')}", f"verdicts: {decision['verdicts']}"]
    for name, entry in decision["datasets"].items():
        anchor = entry["anchor"]
        note = "" if anchor is None else f"  anchor max diff {anchor['max_abs_difference']:.5f}"
        lines.append(f"\n[{name}] {entry['verdict']}{note}")
        if entry["verdict"] == "anchor_failed":
            continue
        lines.append(f"  tau {entry['tau']}, {entry['crops']} crops")
        lines.append("  F1    level  " + "  ".join(f"{rule:>13}" for rule in RULES))
        for level, row in entry["f1"].items():
            lines.append(f"        {level:>5}  " + "  ".join(f"{row[rule]:13.4f}" for rule in RULES))
        lines.append("  rate  level  " + "  ".join(f"{rule:>13}" for rule in entry["selection_rate"]["0"]))
        for level, row in entry["selection_rate"].items():
            lines.append(f"        {level:>5}  " + "  ".join(f"{value:13.3f}" for value in row.values()))
        lines.append(f"  clean drop: selector {entry['clean_drop']:+.4f}, gated {entry['clean_drop_by_rule']['gated']:+.4f}; "
                     f"checks {entry['checks']}")
    return "\n".join(lines)


# ----------------------------------------------------------------- main
def _bundle_manifest_sha(bundle_digests: dict[str, str], localizer_sha: str) -> str:
    text = json.dumps({"localizer": localizer_sha, "selectors": bundle_digests}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def with_split(evaluation: dict[str, Any], split: str) -> dict[str, Any]:
    copied = json.loads(json.dumps(evaluation))
    for spec in copied["datasets"]:
        spec["split"] = split
    return copied


def load_test_config(path: Path, selector_config_sha: str) -> dict[str, Any]:
    test_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if test_config.get("status") != "preregistered":
        raise SystemExit("P27 test config is not marked preregistered")
    if test_config["selector_config_sha256"] != selector_config_sha:
        raise SystemExit("the test config names a different selector config")
    return test_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "train", "evaluate", "test"))
    parser.add_argument("--config", default="configs/p27_selector.yaml")
    parser.add_argument("--test-config", default="configs/p27_test.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features-dir", help="train: output of the features command")
    parser.add_argument("--bundle-dir", help="evaluate / test: output of the train command")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config, f1_config, evaluation = load_config(Path(args.config))
    config_sha = sha256(Path(args.config))
    test_config = None
    if args.command == "test":
        test_config = load_test_config(Path(args.test_config), config_sha)
        evaluation = with_split(evaluation, str(test_config["split"]))
    names = [spec["name"] for spec in evaluation["datasets"]]
    output_dir = Path(args.output_dir)

    if args.command == "train":
        features_dir = Path(args.features_dir)
        features_summary = json.loads((features_dir / "features_summary.json").read_text(encoding="utf-8"))
        if features_summary["config_sha256"] != config_sha:
            raise SystemExit("the features were extracted under a different selector config")
        output_dir.mkdir(parents=True, exist_ok=False)
        summaries = {}
        for name in names:
            bundle, summaries[name] = train_dataset(name, read_records(features_dir / f"features_{name}.csv"), config)
            bundle.update(config_sha256=config_sha, localizer_sha256=features_summary["localizer_sha256"])
            torch.save(bundle, output_dir / f"selector_{name}.pth")
        payload = {"config_sha256": config_sha, "localizer_sha256": features_summary["localizer_sha256"],
                   "datasets": summaries}
        (output_dir / "train_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(render_calibration(summaries))
        return

    device = torch.device(args.device)
    checkpoint, localizer_config_path = localizer_files(config)
    localizer_sha = sha256(checkpoint)
    detectors = load_detectors(f1_config, device)
    reference_file = reference_path(config["f1_reference_decision"])
    reference = json.loads(reference_file.read_text(encoding="utf-8"))
    check_reference(reference, localizer_sha, detectors)
    f1_markers, selector_markers, test_markers = step_marker_dirs(f1_config)
    for spec in evaluation["datasets"]:
        split = config["training_crops"]["split"] if args.command == "features" else spec["split"]
        if not open_dataset(spec, split).rows:
            raise SystemExit(f"{spec['name']}: no {split} rows")
    localizer = TrainedLocalizer(load_localizer(str(checkpoint), device), device)

    if args.command == "features":
        localizer_config = yaml.safe_load(localizer_config_path.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=False)
        summary = run_features(config, evaluation, localizer, detectors, localizer_config, device, output_dir)
        payload = {"config_sha256": config_sha, "localizer_sha256": localizer_sha, "datasets": summary,
                   "f1_reference": {"path": str(reference_file), "sha256": sha256(reference_file)}}
        (output_dir / "features_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    # evaluate (validation, once) and test (test, once, after evaluate).
    localizer_marker = f1_markers / f"{localizer_sha}.json"
    if not localizer_marker.is_file() or json.loads(localizer_marker.read_text(encoding="utf-8")).get("step") != "p27_f1":
        raise SystemExit("this localizer has no F1-step marker; the selector step follows the F1 step")
    bundles, bundle_digests = load_bundles(Path(args.bundle_dir), names)
    for name, bundle in bundles.items():
        if bundle.get("config_sha256") != config_sha or bundle.get("localizer_sha256") != localizer_sha:
            raise SystemExit(f"selector for {name} was trained under a different config or localizer")
    manifest_sha = _bundle_manifest_sha(bundle_digests, localizer_sha)
    record = {"step": "p27_selector" if test_config is None else "p27_test", "config_sha256": config_sha,
              "localizer_sha256": localizer_sha, "selector_sha256": bundle_digests,
              "bundle_dir": args.bundle_dir, "output_dir": args.output_dir,
              "f1_reference": {"path": str(reference_file), "sha256": sha256(reference_file)}}
    if test_config is None:
        output_dir.mkdir(parents=True, exist_ok=False)
        marker = claim(manifest_sha, {"p27": {"max_frozen_versions": int(config["evaluation"]["max_selector_versions"])}},
                       selector_markers, record)
        decision = run_evaluate(config, f1_config, evaluation, localizer, detectors, bundles, reference, device,
                                output_dir)
        filename = "selector_decision.json"
    else:
        if not (selector_markers / f"{manifest_sha}.json").is_file():
            raise SystemExit("these selectors have not been evaluated on validation; the test follows that step")
        output_dir.mkdir(parents=True, exist_ok=False)
        record["test_config_sha256"] = sha256(Path(args.test_config))
        marker = claim(manifest_sha, {"p27": {"max_frozen_versions": int(test_config["max_test_runs"])}},
                       test_markers, record)
        decision = run_evaluate(config, f1_config, evaluation, localizer, detectors, bundles, None, device,
                                output_dir, split=str(test_config["split"]))
        decision["reporting_only"] = True
        filename = "test_decision.json"
    decision.update(frozen=record, marker=str(marker))
    (output_dir / filename).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(render_evaluation(decision))


if __name__ == "__main__":
    main()
