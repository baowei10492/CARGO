"""P33 reviewer experiments on the selector side (docs/P33_REVIEWER_EXPERIMENTS.md).

Border-free P30/P31 protocol, unchanged: the same test crops, draws, scoring pixels, hold-out tiles and selector
recipe (configs/p27_selector.yaml) as the paper's rows. Commands:

    pipeline   features -> train -> test for one localizer and some detector rows: the P27 selector recipe on the
               localizer's own estimates (the 19 features, the same crops). Localizers:
                 v2                                    the paper's registration module
                 v2r_conf_only, v2r_no_change_label    the registration module retrained without change-aware
                                                       weights / without change supervision (configs/p33/)
                 roma_ts_ransac, phasecorr_fm, ecc_ts  matching / classical baselines in the 3-DoF family; their
                                                       pairwise estimates are cached on disk and shared by every
                                                       detector row (rs_cd.estimate_cache)
               --s2looking runs S2Looking (never seen by the registration module) with the S2Looking detector.
    interp     the paper's CARGO (v2 + the frozen selectors) on the test crops: levels 0-3 exactly as the frozen test
               run (they must reproduce its table), plus an interpolation control, T2 resampled by a per-crop
               sub-pixel transform (<= 0.5 px, +-0.5 % scale; rs_cd.p33_protocol), scored on the same pixels.
    identity   the registration module on identical pairs (T1, T1) and (T2, T2) and on T1 against a known level-1
               transform of itself (signed errors); RoMa on identical pairs for a subset.

Finished pieces are kept (resumable). Reporting only; nothing here changes the paper's frozen runs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache
from rs_cd.affine_f1 import DETECTOR_PATHS, add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import affine_parameters, mean_displacement, perturbation_matrix
from rs_cd.datasets.localizer_pairs import normalize
from rs_cd.estimate_cache import DiskCachedEstimator
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p27_selector import COUNT_KEYS, TASK_FEATURES, feature_matrix, registration_features, selector_probability, task_features
from rs_cd.p30_geometry import BorderFreeGeometry
from rs_cd.p33_protocol import INTERP_LEVEL, INTERP_RANGES, RETRAINED_LOCALIZERS, S2LOOKING, S2LOOKING_DETECTOR_POINTER
from tools import p31_localizer_ablation, run_p30, run_p31_plugins
from tools.p31_selector_ablation import DATASETS, sources
from tools.run_p27_f1 import open_detectors
from tools.run_p27_selector import harness, open_dataset, read_records, train_dataset, write_csv
from tools.run_p27_step1 import sha256

ROOT = Path("reports/p33")
P31_CONFIG = Path("configs/p31_border_free.yaml")
P32_CONFIG = Path("configs/p32_plugin_extension.yaml")
CACHED = ("roma_ts_ransac", "roma_ts_huber", "sift_ts_ransac", "phasecorr_fm", "ecc_ts")
ALL_ROWS = ("dinov3", "bit", "cdlamba", "cdxlstm", "snunet", "bifa")
LEVELS = ("0", "1", "2", "3")


# ---------------------------------------------------------------------------------------------- opening
def configs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """([P31 config, P32 config], P30 config, evaluation config, selector recipe), chained hashes checked."""
    p31, p30, evaluation, selector = run_p31_plugins.load(P31_CONFIG)
    p32 = run_p31_plugins.load(P32_CONFIG)[0]
    return [p31, p32], p30, evaluation, selector


def open_row(row: str, name: str, row_configs: Sequence[dict[str, Any]], p30: dict[str, Any],
             device: torch.device) -> dict[str, Any]:
    if row == "dinov3":
        return run_p30.open_detector(p30, "original", name, device)
    if row == "dinov3_s2looking":
        return open_detectors({"detectors": {name: {"checkpoint_pointer": S2LOOKING_DETECTOR_POINTER}}}, device)[name]
    for config in row_configs:
        if row in config["rows"]:
            return run_p31_plugins.row_detector(config, row, name, device)
    raise SystemExit(f"unknown detector row {row}")


def open_localizer(variant: str, p30: dict[str, Any], evaluation: dict[str, Any], device: torch.device):
    if variant == "v2":
        return run_p30.open_localizer(p30, "p30", device)[0]
    return p31_localizer_ablation.build(variant, p30, evaluation, device)


def localizer_identity(variant: str, p30: dict[str, Any]) -> str:
    if variant == "v2":
        return sha256(run_p30.localizer_checkpoint(p30, "p30"))
    if variant in RETRAINED_LOCALIZERS:
        return sha256(run_p30.pointed(RETRAINED_LOCALIZERS[variant]) / "checkpoints" / "best.pth")
    return variant


def with_cache(localizer, variant: str, name: str, cache_dir: Path):
    """Matching baselines: memo their estimates per pair key on disk (shared by all detector rows)."""
    if variant in CACHED:
        inner = localizer.estimator
        if isinstance(inner, DiskCachedEstimator):
            inner.close()
            inner = inner.estimator
        localizer.estimator = DiskCachedEstimator(inner, cache_dir / f"{variant}_{name}.jsonl")
    return localizer


def localizer_config(p30: dict[str, Any]) -> dict[str, Any]:
    """The paper's localizer run config: its hold-out tiles define the selector's calibration crops for every run."""
    return yaml.safe_load((run_p30.localizer_checkpoint(p30, "p30").parent.parent / "config.yaml").read_text(encoding="utf-8"))


def release(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    partial = path.with_suffix(".partial.csv")
    write_csv(partial, rows)
    partial.replace(path)


# --------------------------------------------------------------------------------------------- pipeline
def render(variant: str, row: str, name: str, entry: dict[str, Any]) -> str:
    f1, rate = entry["f1"], entry["selection_rate"]
    def line(rule: str) -> str:
        return " / ".join(f"{f1[level][rule] * 100:5.1f}" for level in LEVELS)
    return "\n".join([
        f"[{variant} {row} {name}] {entry['crops']} crops; F1 % at L0 / L1 / L2 / L3; tau {entry['tau']}",
        f"  never    {line('raw')}",
        f"  always   {line('estimated')}",
        f"  selector {line('selector')}   adopted L0 {rate['0']['selector']:.3f}, L3 {rate['3']['selector']:.3f}"])


def run_pipeline(args: argparse.Namespace) -> None:
    row_configs, p30, evaluation, selector = configs()
    device = torch.device(args.device)
    h = harness(evaluation)
    margin = int(p30["crop_margin"])
    held = localizer_config(p30)
    localizer = open_localizer(args.localizer, p30, evaluation, device)
    identity = localizer_identity(args.localizer, p30)
    specs = [dict(S2LOOKING)] if args.s2looking else [s for s in evaluation["datasets"] if s["name"] in args.datasets]
    rows = ["dinov3_s2looking"] if args.s2looking else list(args.rows)
    root = Path(args.output_root) / "selector" / args.localizer
    for spec in specs:
        name = spec["name"]
        with_cache(localizer, args.localizer, name, root / "cache")
        training = open_dataset(spec, selector["training_crops"]["split"])
        items = run_p30.training_items(training, spec, selector, held, h["size"], margin)
        if args.max_crops:      # smoke runs: spread over the hold-out crops (listed first) and the fit crops
            items = items[:: max(1, len(items) // args.max_crops)][: args.max_crops]
        for row in rows:
            out = root / row
            out.mkdir(parents=True, exist_ok=True)
            features = out / f"features_{name}.csv"
            if not features.is_file():
                detector = open_row(row, name, row_configs, p30, device)
                records = run_p30.records_for(training, spec, items, localizer, detector, h, device,
                                              f"{args.localizer} {row} features")
                atomic_csv(features, records)
                del detector
                release(device)
            bundle_path = out / f"selector_{name}.pth"
            if not bundle_path.is_file():
                bundle, summary = train_dataset(name, read_records(features), selector)
                bundle.update(protocol="p33", localizer=args.localizer, localizer_sha256=identity)
                torch.save(bundle, bundle_path)
                (out / f"train_{name}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        test = open_dataset(spec, "test")
        crops = run_p30.margin_crops(test, spec, h["size"], margin)
        crops = crops[: args.max_crops] if args.max_crops else crops
        for row in rows:
            out = root / row
            result_path = out / f"test_{name}.json"
            if result_path.is_file():
                print(f"[{args.localizer} {row} {name}] kept ({result_path})", flush=True)
                continue
            bundle = torch.load(out / f"selector_{name}.pth", map_location="cpu", weights_only=False)
            detector = open_row(row, name, row_configs, p30, device)
            records = run_p30.records_for(test, spec, [(crop, "test") for crop in crops], localizer, detector, h,
                                          device, f"{args.localizer} {row} test")
            system = run_p30.system_table(records, bundle)
            atomic_csv(out / f"records_{name}.csv", records)
            entry = {"localizer": args.localizer, "localizer_sha256": identity, "row": row, "dataset": name,
                     "crops": len(crops), "detector_sha256": detector["sha256"], **system}
            result_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            print(render(args.localizer, row, name, entry), flush=True)
            del detector
            release(device)
        estimator = getattr(localizer, "estimator", None)
        if isinstance(estimator, DiskCachedEstimator):
            print(f"[{args.localizer} {name}] cache: {len(estimator)} pairs, {estimator.misses} computed, "
                  f"{estimator.hits} reused", flush=True)


# ----------------------------------------------------------------------------------------------- interp
@torch.no_grad()
def interp_records(localizer, detector: dict[str, Any], spec: dict[str, Any], item: dict[str, Any], t1: np.ndarray,
                   t2: np.ndarray, mask: np.ndarray, h: dict[str, Any], geometry: BorderFreeGeometry,
                   device: torch.device) -> dict[str, Any]:
    """One record for the interpolation-control view, built as rs_cd.p27_selector.crop_records builds a level."""
    size = h["size"]
    truth_mask = mask > int(spec["mask_threshold"])
    standard = [perturbation_matrix(*affine_parameters(item["crop_id"], level, h["ranges"], h["seed"]),
                                    width=size, height=size) for level in h["levels"]]
    valid = scoring_mask(standard, size)                                   # the paper's scoring pixels
    tiny = perturbation_matrix(*affine_parameters(item["crop_id"], INTERP_LEVEL, INTERP_RANGES, h["seed"]),
                               width=size, height=size)
    target_image = geometry.perturb(tiny)
    estimate = localizer.estimate_many([t1], [target_image])[0]
    back = localizer.estimate_many([target_image], [t1])[0]
    views = geometry.path_images(None, tiny, estimate.matrix)
    t2_batch = torch.stack([normalize(view) for view in views.values()]).to(device)
    t1_batch = normalize(t1).unsqueeze(0).repeat(len(views), 1, 1, 1).to(device)
    logits = detector["model"](t1_batch, t2_batch)["change_logits"]
    target = torch.from_numpy(truth_mask.astype(np.float32))[None, None].to(device)
    keep = torch.from_numpy(valid.astype(np.float32))[None, None].to(device)
    counts = confusion_counts(logits, target, keep, detector["threshold"])
    raw, estimated = DETECTOR_PATHS.index("raw"), DETECTOR_PATHS.index("estimated")
    inside = torch.from_numpy(scoring_mask([estimate.matrix], size).astype(np.float32))[None, None].to(device)
    task = task_features(logits[[raw]].float(), logits[[estimated]].float(), inside, detector["threshold"]).cpu().tolist()[0]
    record: dict[str, Any] = {"dataset": spec["name"], "tile_id": item["sample_id"], "crop_id": item["crop_id"],
                              "view": "0_interp", "tiny_displacement_px": mean_displacement(tiny, h["points"])}
    record.update(registration_features(estimate.matrix, back.matrix, estimate.misaligned_probability, h["points"], size))
    record.update(zip(TASK_FEATURES, task))
    for j, path in enumerate(DETECTOR_PATHS):
        record.update({f"{path}_{key}": int(value) for key, value in zip(COUNT_KEYS, counts[j])})
    return record


def pooled(records: Sequence[dict[str, Any]], choose) -> float:
    metrics = BinaryChangeMetrics(threshold=0.5)
    for record in records:
        path = choose(record)
        add_counts(metrics, [record[f"{path}_{key}"] for key in COUNT_KEYS])
    return float(metrics.compute()["f1"])


def view_summary(records: Sequence[dict[str, Any]], probability: np.ndarray, tau: float) -> dict[str, Any]:
    adopted = probability >= tau
    picks = {id(r): ("estimated" if a else "raw") for r, a in zip(records, adopted)}
    features = ("estimate_displacement_px", "forward_backward_px", "misaligned_probability")
    return {"crops": len(records), "adoption_rate": float(adopted.mean()),
            "f1": {"raw": pooled(records, lambda r: "raw"), "estimated": pooled(records, lambda r: "estimated"),
                   "selector": pooled(records, lambda r: picks[id(r)])},
            "median_probability": float(np.median(probability)),
            "median_features": {name: float(np.median([float(r[name]) for r in records])) for name in features}}


def frozen_anchor(row: str, name: str) -> dict[str, dict[str, float]]:
    """{level: {rule: F1}} of the frozen test run of a row."""
    if row == "dinov3":
        data = json.loads((run_p30.pointed("reports/p30/test/LATEST") / "p30_test.json").read_text(encoding="utf-8"))
        return data["datasets"][name]["proposed"]["f1"]
    _, records = sources(row)
    data = json.loads((records[0].parent / "p31_test.json").read_text(encoding="utf-8"))
    return data["datasets"][name]["f1"]


def run_interp(args: argparse.Namespace) -> None:
    row_configs, p30, evaluation, _ = configs()
    device = torch.device(args.device)
    h = harness(evaluation)
    margin = int(p30["crop_margin"])
    localizer = open_localizer("v2", p30, evaluation, device)
    root = Path(args.output_root) / "interp"
    for row in args.rows:
        train_root, _ = sources(row)
        for spec in evaluation["datasets"]:
            name = spec["name"]
            if name not in args.datasets:
                continue
            out = root / row
            result_path = out / f"interp_{name}.json"
            if result_path.is_file():
                print(f"[interp {row} {name}] kept", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            bundle = torch.load(train_root / "bundles" / f"selector_{name}.pth", map_location="cpu", weights_only=False)
            tau = math.inf if bundle["tau"] is None else float(bundle["tau"])
            detector = open_row(row, name, row_configs, p30, device)
            test = open_dataset(spec, "test")
            crops = run_p30.margin_crops(test, spec, h["size"], margin)
            crops = crops[: args.max_crops] if args.max_crops else crops
            items = [(crop, "test") for crop in crops]
            standard = run_p30.records_for(test, spec, items, localizer, detector, h, device, f"interp {row} levels")
            cache = TileCache(test, int(spec["tile_size"]))
            control = []
            for index, item in enumerate(crops, start=1):
                t1, t2, mask = cache.crop(item, h["size"])
                geometry = BorderFreeGeometry(cache.arrays[1], item["top"], item["left"], h["size"])
                control.append(interp_records(localizer, detector, spec, item, t1, t2, mask, h, geometry, device))
                if index % 500 == 0 or index == len(crops):
                    print(f"[interp {row} {name} control {index}/{len(crops)}]", flush=True)
            system = run_p30.system_table(standard, bundle)
            level0 = [r for r in standard if int(r["level"]) == 0]
            p_level0 = np.array([float(r["selector_probability"]) for r in level0])
            p_control = selector_probability(bundle, feature_matrix(control, bundle["feature_names"])).numpy().astype(float)
            for record, p in zip(control, p_control):
                record["selector_probability"] = float(p)
            entry: dict[str, Any] = {"row": row, "dataset": name, "tau": bundle["tau"], "crops": len(crops),
                                     "interp_ranges": {"max_translation_px": INTERP_RANGES.max_translation_px / 3,
                                                       "scale": [1 + (INTERP_RANGES.scale_low - 1) / 3,
                                                                 1 + (INTERP_RANGES.scale_high - 1) / 3]},
                                     "levels": system["f1"], "selection_rate": system["selection_rate"],
                                     "level0": view_summary(level0, p_level0, tau),
                                     "interp": view_summary(control, p_control, tau)}
            if not args.max_crops:
                anchor = frozen_anchor(row, name)
                differences = {f"{level}/{rule}": abs(system["f1"][level][rule] - float(anchor[level][rule]))
                               for level in LEVELS for rule in ("raw", "estimated", "selector")}
                entry["anchor"] = {"max_abs_difference": max(differences.values()), "differences": differences,
                                   "passed": max(differences.values()) <= 1e-3}
            atomic_csv(out / f"interp_records_{name}.csv", control)
            result_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            a, b = entry["level0"], entry["interp"]
            print(f"[interp {row} {name}] adopted L0 {a['adoption_rate']:.3f} -> control {b['adoption_rate']:.3f}; "
                  f"selector - raw L0 {(a['f1']['selector'] - a['f1']['raw']) * 100:+.2f} -> control "
                  f"{(b['f1']['selector'] - b['f1']['raw']) * 100:+.2f}; anchor "
                  f"{entry.get('anchor', {}).get('max_abs_difference', float('nan')):.2e}", flush=True)
            del detector
            release(device)


# --------------------------------------------------------------------------------------------- identity
def signed_errors(estimate: np.ndarray, truth: np.ndarray, size: int) -> dict[str, float]:
    centre = np.array([(size - 1) / 2.0, (size - 1) / 2.0])
    moved = estimate[:, :2] @ centre + estimate[:, 2] - (truth[:, :2] @ centre + truth[:, 2])
    return {"centre_dx": float(moved[0]), "centre_dy": float(moved[1]),
            "log_scale_ratio": float(math.log(max(estimate[0, 0], 1e-6) / max(truth[0, 0], 1e-6)))}


def summarise(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(array)), "mean": float(array.mean()), "p90": float(np.percentile(array, 90)),
            "max": float(array.max())}


@torch.no_grad()
def run_identity(args: argparse.Namespace) -> None:
    _, p30, evaluation, _ = configs()
    device = torch.device(args.device)
    h = harness(evaluation)
    size, margin = h["size"], int(p30["crop_margin"])
    localizer = open_localizer("v2", p30, evaluation, device)
    roma = open_localizer("roma_ts_ransac", p30, evaluation, device) if args.roma_crops else None
    out = Path(args.output_root) / "identity"
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"datasets": {}}
    for spec in evaluation["datasets"]:
        name = spec["name"]
        test = open_dataset(spec, "test")
        crops = run_p30.margin_crops(test, spec, size, margin)
        crops = crops[: args.max_crops] if args.max_crops else crops
        cache = TileCache(test, int(spec["tile_size"]))
        same1, same2, known_error, signed, roma_same = [], [], [], [], []
        for index, item in enumerate(crops, start=1):
            t1, t2, _ = cache.crop(item, size)
            known = perturbation_matrix(*affine_parameters(item["crop_id"], 1, h["ranges"], h["seed"]), width=size, height=size)
            shifted = BorderFreeGeometry(cache.arrays[0], item["top"], item["left"], size).perturb(known)
            estimates = localizer.estimate_many([t1, t2, t1], [t1, t2, shifted])
            same1.append(mean_displacement(estimates[0].matrix, h["points"]))
            same2.append(mean_displacement(estimates[1].matrix, h["points"]))
            difference = estimates[2].matrix - known
            known_error.append(float(np.mean(np.linalg.norm(h["points"] @ difference[:, :2].T + difference[:, 2], axis=1))))
            signed.append(signed_errors(estimates[2].matrix, known, size))
            if roma is not None and index <= args.roma_crops:
                roma_same.append(mean_displacement(roma.estimate_many([t1], [t1])[0].matrix, h["points"]))
            if index % 500 == 0 or index == len(crops):
                print(f"[identity {name} {index}/{len(crops)}]", flush=True)
        entry = {"crops": len(crops), "identical_t1_px": summarise(same1), "identical_t2_px": summarise(same2),
                 "known_level1_error_px": summarise(known_error),
                 "known_level1_signed": {key: summarise([s[key] for s in signed]) for key in signed[0]}}
        if roma_same:
            entry["roma_identical_t1_px"] = {**summarise(roma_same), "crops": len(roma_same)}
        result["datasets"][name] = entry
        print(json.dumps({name: entry}, indent=1), flush=True)
    (out / "identity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pipeline", "interp", "identity"))
    parser.add_argument("--localizer", default="v2")
    parser.add_argument("--rows", nargs="+", default=list(ALL_ROWS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--s2looking", action="store_true")
    parser.add_argument("--roma-crops", type=int, default=300, help="identity: RoMa on identical pairs, first N crops")
    parser.add_argument("--max-crops", type=int, default=None, help="smoke runs only (with --output-root elsewhere)")
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_crops and Path(args.output_root).resolve() == ROOT.resolve():
        raise SystemExit("--max-crops writes partial results; give --output-root another directory")
    {"pipeline": run_pipeline, "interp": run_interp, "identity": run_identity}[args.command](args)


if __name__ == "__main__":
    main()
