"""P27 development diagnosis on the training hold-out.

Training hold-out only (the hashed 5% of training tiles the trainer already
evaluates); no validation or test data, and nothing here is a gate. For one
checkpoint and every hold-out crop:

  real       T1 reference, T2 perturbed: the trainer's hold-out rows, kept per crop;
  synthetic  T2 reference, T2 perturbed: the same perturbations without any
             bi-temporal difference, i.e. the localizer's own precision;
  identical  (T1, T1) and (T2, T2): what an identical pair yields;
  swapped    (T2, T1) composed with the clean estimate (T1, T2): forward-backward
             consistency on the real pair.

Masks feed the change-fraction column only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml

from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    AffineRanges,
    compose,
    dataset_summary,
    grid_points,
    mean_displacement,
)
from rs_cd.localizer_training import TrainedLocalizer, load_localizer, step1_rows
from tools.run_p27_step1 import sha256
from tools.train_similarity_localizer import build_holdout

CLEAN_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, math.inf))
CONSISTENCY_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf))
GROSS_ERROR_PX = 5.0


def matrix_of(row: dict[str, Any]) -> np.ndarray:
    return np.array([[row["a11"], row["a12"], row["a13"]], [row["a21"], row["a22"], row["a23"]]], dtype=np.float64)


def best_error(row: dict[str, Any]) -> float:
    """The error the success rule looks at: the smaller of raw and differential."""

    errors = [error for error in (row["ate_raw"], row["ate_diff"]) if error is not None]
    return min(errors) if errors else math.inf


def diagnose_crop(
    localizer: TrainedLocalizer,
    *,
    dataset: str,
    item: dict[str, Any],
    ranges: AffineRanges,
    levels: Sequence[int],
    seed: int,
    points: np.ndarray,
    tolerance: float,
    trigger_probability: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common = dict(dataset=dataset, crop_id=item["crop_id"], change_fraction=item["change_fraction"],
                  ranges=ranges, levels=levels, seed=seed, points=points, tolerance=tolerance,
                  trigger_probability=trigger_probability)
    t1, t2 = item["t1"], item["t2"]
    real = step1_rows(localizer, t1=t1, t2=t2, **common)
    synthetic = step1_rows(localizer, t1=t2, t2=t2, **common)
    same1, same2, swapped = localizer.estimate_many([t1, t2, t2], [t1, t2, t1])
    crop = {
        "dataset": dataset,
        "crop_id": item["crop_id"],
        "change_fraction": item["change_fraction"],
        "clean_px": real[0]["estimate_displacement_px"],
        "clean_trigger": real[0]["trigger"],
        "identical_t1_px": mean_displacement(same1.matrix, points),
        "identical_t2_px": mean_displacement(same2.matrix, points),
        "identical_t1_probability": same1.misaligned_probability,
        "identical_t2_probability": same2.misaligned_probability,
        # T1 -> T2 -> T1 should be the identity whatever the pair's true offset is.
        "forward_backward_px": mean_displacement(compose(swapped.matrix, matrix_of(real[0])), points),
    }
    for pair, rows in (("real", real), ("synthetic", synthetic)):
        for row in rows:
            row["pair"] = pair
            if row["level"] != CLEAN_LEVEL:
                crop[f"{pair}_l{row['level']}_error"] = best_error(row)
                crop[f"{pair}_l{row['level']}_success"] = row["success"]
    return real + synthetic, crop


# ------------------------------------------------------------- summaries
def rate(flags: Iterable[bool]) -> float | None:
    flags = [bool(flag) for flag in flags]
    return sum(flags) / len(flags) if flags else None


def quantiles(values: Iterable[float]) -> dict[str, float] | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {"q50": float(np.quantile(finite, 0.5)), "q80": float(np.quantile(finite, 0.8)),
            "q90": float(np.quantile(finite, 0.9)), "max": max(finite)}


def binned(crops: Sequence[dict[str, Any]], key: str, bins, top: int) -> list[dict[str, Any]]:
    out = []
    for low, high in bins:
        members = [crop for crop in crops if low <= crop[key] < high]
        out.append({
            "range": [low, None if math.isinf(high) else high],
            "crops": len(members),
            "real_top_success": rate(crop[f"real_l{top}_success"] for crop in members),
            "synthetic_top_success": rate(crop[f"synthetic_l{top}_success"] for crop in members),
        })
    return out


def failure_pattern(crops: Sequence[dict[str, Any]], levels: Sequence[int]) -> dict[str, float | None]:
    """Whether real-pair failures belong to the crop (every level) or to the perturbation."""

    failures = [sum(not crop[f"real_l{level}_success"] for level in levels) for crop in crops]
    return {
        "fails_none": rate(count == 0 for count in failures),
        "fails_some": rate(0 < count < len(levels) for count in failures),
        "fails_all": rate(count == len(levels) for count in failures),
    }


def error_summary(errors: Sequence[float]) -> dict[str, Any]:
    return {**(quantiles(errors) or {}), "above_gross": rate(error > GROSS_ERROR_PX for error in errors)}


def summarize(rows: Sequence[dict[str, Any]], crops: Sequence[dict[str, Any]],
              evaluation: dict[str, Any]) -> dict[str, Any]:
    levels = [int(level) for level in evaluation["levels"]]
    top = max(levels)
    trigger = float(evaluation["trigger_probability"])
    by_pair = {
        pair: dataset_summary([row for row in rows if row["pair"] == pair], evaluation)["levels"]
        for pair in ("real", "synthetic")
    }
    return {
        "crops": len(crops),
        "success": {pair: {str(level): by_pair[pair][str(level)]["success_rate"] for level in levels}
                    for pair in by_pair},
        "clean_trigger_rate": by_pair["real"][str(CLEAN_LEVEL)]["trigger_rate"],
        "identical_trigger_rate": rate(
            [crop["identical_t1_probability"] > trigger for crop in crops]
            + [crop["identical_t2_probability"] > trigger for crop in crops]
        ),
        "identical_px": quantiles([crop["identical_t1_px"] for crop in crops]
                                  + [crop["identical_t2_px"] for crop in crops]),
        "clean_px": quantiles(crop["clean_px"] for crop in crops),
        "forward_backward_px": quantiles(crop["forward_backward_px"] for crop in crops),
        "top_error": {pair: error_summary([crop[f"{pair}_l{top}_error"] for crop in crops])
                      for pair in ("real", "synthetic")},
        "failure_pattern": failure_pattern(crops, levels),
        "top_success_by_clean_px": binned(crops, "clean_px", CLEAN_BINS, top),
        "top_success_by_forward_backward_px": binned(crops, "forward_backward_px", CONSISTENCY_BINS, top),
    }


def diagnose(config: dict[str, Any], localizer: TrainedLocalizer, holdout: dict[str, list[dict[str, Any]]]):
    evaluation = config["holdout"]["evaluation"]
    size = int(config["data"]["crop_size"])
    levels = [int(level) for level in evaluation["levels"]]
    # Same ranges, seed and grid as the trainer's hold-out evaluation.
    ranges = AffineRanges(0.0, float(evaluation["scale_range"][0]), float(evaluation["scale_range"][1]),
                          float(evaluation["max_translation_px"]), len(levels))
    points = grid_points(size, size, int(evaluation["grid_points"]))
    all_rows, all_crops, summaries = [], [], {}
    for name, items in holdout.items():
        rows, crops = [], []
        for item in items:
            crop_rows, crop = diagnose_crop(
                localizer, dataset=name, item=item, ranges=ranges, levels=levels, seed=int(config["seed"]),
                points=points, tolerance=float(evaluation["success_ate_px"]),
                trigger_probability=float(evaluation["trigger_probability"]),
            )
            rows.extend(crop_rows)
            crops.append(crop)
        summaries[name] = summarize(rows, crops, evaluation)
        all_rows.extend(rows)
        all_crops.extend(crops)
    return all_rows, all_crops, summaries


def trainer_difference(summaries: dict[str, Any], trainer: dict[str, Any]) -> float:
    """Largest gap between the real-pair success here and the trainer's hold-out file."""

    return max(
        abs(summary["success"]["real"][level] - trainer["summaries"][name]["levels"][level]["success_rate"])
        for name, summary in summaries.items()
        for level in summary["success"]["real"]
    )


# ---------------------------------------------------------------- output
def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _quantiles(stats: dict[str, float] | None) -> str:
    return "-" if not stats else "/".join(_fmt(stats[key]) for key in ("q50", "q80", "q90"))


def _bins(entries: Sequence[dict[str, Any]]) -> str:
    parts = []
    for entry in entries:
        low, high = entry["range"]
        label = f"{low:g}+" if high is None else f"{low:g}-{high:g}"
        parts.append(f"{label}: n={entry['crops']} real {_fmt(entry['real_top_success'])} "
                     f"syn {_fmt(entry['synthetic_top_success'])}")
    return " | ".join(parts)


def render(summary: dict[str, Any]) -> str:
    checkpoint = summary["checkpoint"]
    lines = [f"checkpoint {checkpoint['source']} step {checkpoint['step']} sha {checkpoint['sha256'][:12]}"]
    if summary.get("trainer_max_difference") is not None:
        lines.append(f"real success vs trainer hold-out file: max |difference| {summary['trainer_max_difference']:.3f}")
    for name, entry in summary["datasets"].items():
        success = entry["success"]
        top = entry["top_error"]
        pattern = entry["failure_pattern"]
        lines += [
            f"{name} ({entry['crops']} crops)",
            "  success by level      real " + "/".join(_fmt(v) for v in success["real"].values())
            + "   synthetic " + "/".join(_fmt(v) for v in success["synthetic"].values()),
            f"  trigger rate          clean real pair {_fmt(entry['clean_trigger_rate'])}"
            f"   identical pair {_fmt(entry['identical_trigger_rate'])}",
            f"  identical pair px     q50/q80/q90 {_quantiles(entry['identical_px'])}",
            f"  clean estimate px     q50/q80/q90 {_quantiles(entry['clean_px'])}",
            f"  forward-backward px   q50/q80/q90 {_quantiles(entry['forward_backward_px'])}",
            f"  top-level error px    real {_quantiles(top['real'])} (>{GROSS_ERROR_PX:g} px {_fmt(top['real']['above_gross'])})"
            f"   synthetic {_quantiles(top['synthetic'])} (>{GROSS_ERROR_PX:g} px {_fmt(top['synthetic']['above_gross'])})",
            f"  real failures         none {_fmt(pattern['fails_none'])}  some levels {_fmt(pattern['fails_some'])}"
            f"  all levels {_fmt(pattern['fails_all'])}",
            "  top success by clean px            " + _bins(entry["top_success_by_clean_px"]),
            "  top success by forward-backward px " + _bins(entry["top_success_by_forward_backward_px"]),
        ]
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="training run directory (config.yaml, checkpoints/)")
    parser.add_argument("--checkpoint", default="last.pth", help="file name under checkpoints/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    # The trainer may rewrite its checkpoints while this runs; work on a copy.
    source = run_dir / "checkpoints" / args.checkpoint
    snapshot = output_dir / "checkpoint_snapshot.pth"
    shutil.copy2(source, snapshot)
    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    checkpoint = {"source": str(source), "sha256": sha256(snapshot), "step": int(payload["step"]),
                  "holdout_score": payload.get("holdout_score")}
    device = torch.device(args.device)
    localizer = TrainedLocalizer(load_localizer(str(snapshot), device), device, amp=bool(config["optim"]["amp"]))
    snapshot.unlink()

    start = time.perf_counter()
    rows, crops, summaries = diagnose(config, localizer, build_holdout(config))
    summary = {"purpose": "P27 development diagnosis on the training hold-out; not a gate",
               "checkpoint": checkpoint, "datasets": summaries, "seconds": time.perf_counter() - start}
    trainer_file = run_dir / f"holdout_step{checkpoint['step']:06d}.json"
    if trainer_file.is_file():
        summary["trainer_max_difference"] = trainer_difference(
            summaries, json.loads(trainer_file.read_text(encoding="utf-8")))
    write_csv(output_dir / "diagnosis_rows.csv", rows)
    write_csv(output_dir / "diagnosis_crops.csv", crops)
    (output_dir / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(render(summary))


if __name__ == "__main__":
    main()
