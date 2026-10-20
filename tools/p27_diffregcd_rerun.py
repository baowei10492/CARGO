"""DiffRegCD re-run with the wiring its change head requires (docs/P27_DIFFREGCD_RERUN.md).

Runs in the DiffRegCD environment. A = the perturbed T1, B = the unperturbed T2, label in
the original frame; the matrix is the one the other methods see (they get T2 perturbed by
it). Parts:

    mechanism    level-3 draw, B = T2, A = perturbed T1 / T1 with only the border / perturbed T2 /
                 perturbed T1 of another tile / zeros
    border_free  P30 margin crops, A cut from the whole T1 tile warped by the matrix (no padding);
                 plus the level-3 per-crop warp with zero border on the same crops
    bordered     P27 grid crops and perturbation (per-crop warp, zero border)

Every view is scored on the P27 scoring pixels (and on all pixels), for the released test
path and the training-time wiring. Parts are resumable per dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_crops, select_margin_crops
from rs_cd.affine_f1 import add_counts, confusion_counts, scoring_mask
from rs_cd.affine_registration import CLEAN_LEVEL, AffineRanges, affine_parameters, perturb, perturbation_matrix
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_diffregcd import stable_inference_seed
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.p30_geometry import BorderFreeGeometry
from tools.run_p27_diffregcd_f1 import all_crops, audit_weights, change_logit, load_config, sha256, unit, weight_paths

WIRINGS = ("official_unmodified", "intended_warped")
PARTS = ("mechanism", "border_free", "bordered")
MECHANISM_VIEWS = ("normal", "border_only", "same_date", "unrelated", "black")
PRIMARY = "official_unmodified"


def harness(evaluation: dict[str, Any]) -> tuple[int, list[int], AffineRanges, int]:
    perturbation = evaluation["perturbation"]
    levels = [int(level) for level in perturbation["levels"]]
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(0.0, float(low), float(high), float(perturbation["max_translation_px"]), len(levels))
    return int(perturbation["crop_size"]), levels, ranges, int(evaluation["seed"])


def matrices_for(crop_id: str, levels: list[int], ranges: AffineRanges, seed: int, size: int) -> dict[int, np.ndarray]:
    """The P27 draws: the matrix every other method sees for this crop and level."""
    return {level: perturbation_matrix(*affine_parameters(crop_id, level, ranges, seed), width=size, height=size)
            for level in [CLEAN_LEVEL, *levels]}


def with_border_only(t1: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """T1 unmoved, zero exactly where T1 perturbed by the matrix has no content."""
    covered = perturb(np.full_like(t1, 255), matrix).max(axis=2) > 0
    return np.where(covered[..., None], t1, 0).astype(t1.dtype)


class Views:
    """(views, scoring pixels) per crop for one part; a view is (key, image A, image B)."""

    def __init__(self, part: str, dataset: PairedChangeDataset, spec: dict[str, Any], evaluation: dict[str, Any], *,
                 margin: int, cap: int | None) -> None:
        self.part = part
        self.size, self.levels, self.ranges, self.seed = harness(evaluation)
        tile = int(spec["tile_size"])
        self.cache = TileCache(dataset, tile)
        if part == "bordered":
            self.crops = all_crops(dataset, tile, self.size)
        elif part == "border_free":
            self.crops = select_margin_crops(dataset.rows, tile, self.size, margin)
        elif part == "mechanism":
            everything = len(dataset.rows) * (tile // self.size) ** 2
            self.crops = select_crops(dataset.rows, tile, self.size, cap if cap is not None else max(everything, 1))
            self.other_cache = TileCache(dataset, tile)
        else:
            raise ValueError(f"unknown part {part}")

    def other(self, index: int) -> dict[str, Any]:
        """A crop from another tile, half the list away."""
        count = len(self.crops)
        for step in range(count // 2, count // 2 + count):
            candidate = self.crops[(index + step) % count]
            if candidate["row"] != self.crops[index]["row"]:
                return candidate
        raise SystemExit("the mechanism part needs crops from at least two tiles")

    def make(self, index: int, t1: np.ndarray, t2: np.ndarray) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], np.ndarray]:
        item = self.crops[index]
        matrices = matrices_for(item["crop_id"], self.levels, self.ranges, self.seed, self.size)
        top = matrices[self.levels[-1]]
        if self.part == "mechanism":
            other_t1 = self.other_cache.crop(self.other(index), self.size)[0]
            views = [("normal", perturb(t1, top), t2), ("border_only", with_border_only(t1, top), t2),
                     ("same_date", perturb(t2, top), t2), ("unrelated", perturb(other_t1, top), t2),
                     ("black", np.zeros_like(t1), t2)]
            return views, scoring_mask([top], self.size)
        views = [(str(CLEAN_LEVEL), t1, t2)]
        if self.part == "bordered":
            views += [(str(level), perturb(t1, matrices[level]), t2) for level in self.levels]
        else:
            geometry = BorderFreeGeometry(self.cache.arrays[0], item["top"], item["left"], self.size)
            views += [(str(level), geometry.perturb(matrices[level]), t2) for level in self.levels]
            views.append((f"bordered_{self.levels[-1]}", perturb(t1, top), t2))
        return views, scoring_mask([matrices[level] for level in self.levels], self.size)


def run_part(part: str, spec: dict[str, Any], evaluation: dict[str, Any], infer: Callable[[torch.Tensor, torch.Tensor], dict],
             *, margin: int, cap: int | None, seed: int, threshold: float, device: torch.device,
             max_crops: int | None = None) -> dict[str, Any]:
    dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                  normalization="unit", mask_threshold=int(spec["mask_threshold"]))
    views = Views(part, dataset, spec, evaluation, margin=margin, cap=cap)
    crops = views.crops[:max_crops] if max_crops is not None else views.crops
    size = views.size
    everywhere = torch.ones(1, 1, size, size, device=device)
    totals: dict[str, BinaryChangeMetrics] = {}
    for index, item in enumerate(crops):
        t1, t2, mask = views.cache.crop(item, size)
        target = torch.from_numpy((mask > int(spec["mask_threshold"])).astype(np.float32))[None, None].to(device)
        crop_views, valid = views.make(index, t1, t2)
        keep = {"scoring": torch.from_numpy(valid.astype(np.float32))[None, None].to(device), "all": everywhere}
        crop_seed = stable_inference_seed(seed, item["crop_id"], 0)     # as the P27 run: shared by the crop's views
        for key, image_a, image_b in crop_views:
            torch.manual_seed(crop_seed)
            np.random.seed(crop_seed % (2**32))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(crop_seed)
            outputs = infer(unit(image_a)[None].to(device) * 2.0 - 1.0, unit(image_b)[None].to(device) * 2.0 - 1.0)
            for wiring in WIRINGS:
                logit = change_logit(outputs[wiring])
                for scoring, pixels in keep.items():
                    metrics = totals.setdefault(f"{key}/{wiring}/{scoring}", BinaryChangeMetrics(threshold=threshold))
                    add_counts(metrics, confusion_counts(logit, target, pixels, threshold)[0])
        if (index + 1) % 200 == 0 or index + 1 == len(crops):
            print(f"[diffregcd rerun {part} {spec['name']} {index + 1}/{len(crops)}]", flush=True)
    computed = {key: metrics.compute() for key, metrics in totals.items()}
    return {"part": part, "crops": len(crops),
            "f1": {key: float(values["f1"]) for key, values in computed.items()},
            "precision": {key: float(values["precision"]) for key, values in computed.items()},
            "recall": {key: float(values["recall"]) for key, values in computed.items()}}


def border_reading(results: dict[str, dict[str, Any]], top_level: int) -> dict[str, Any]:
    drops = {name: r["f1"][f"bordered_{top_level}/{PRIMARY}/scoring"] - r["f1"][f"{top_level}/{PRIMARY}/scoring"]
             for name, r in results.items()}
    mean = sum(drops.values()) / len(drops)
    verdict = "border_dependent" if mean >= 0.05 else "border_independent" if mean <= 0.02 else "partial"
    return {"drops": drops, "mean_drop": mean, "verdict": verdict}


def dates_reading(result: dict[str, Any]) -> dict[str, Any]:
    normal = result["f1"][f"normal/{PRIMARY}/scoring"]
    same = result["f1"][f"same_date/{PRIMARY}/scoring"]
    ratio = same / normal if normal > 0 else math.nan
    verdict = ("ignores_dates" if ratio >= 0.8 else "uses_dates" if ratio <= 0.3 else "partial") if normal > 0 else "undefined"
    return {"normal_f1": normal, "same_date_f1": same, "ratio": ratio, "verdict": verdict}


def render(name: str, result: dict[str, Any]) -> str:
    keys = sorted({key.split("/")[0] for key in result["f1"]},
                  key=lambda k: (MECHANISM_VIEWS.index(k) if k in MECHANISM_VIEWS else 10 + len(k), k))
    lines = [f"[{result['part']} {name}] {result['crops']} crops; F1 % (recall %): official scoring pixels | "
             "official all pixels | training-time wiring scoring pixels"]
    for key in keys:
        cells = [f"{result['f1'][f'{key}/{wiring}/{scoring}'] * 100:5.1f} ({result['recall'][f'{key}/{wiring}/{scoring}'] * 100:4.1f})"
                 for wiring, scoring in ((PRIMARY, "scoring"), (PRIMARY, "all"), ("intended_warped", "scoring"))]
        lines.append(f"  {key:<12} " + " | ".join(cells))
    return "\n".join(lines)


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rerun = yaml.safe_load(path.read_text(encoding="utf-8"))
    if rerun.get("status") != "preregistered":
        raise SystemExit("the re-run config is not marked preregistered")
    if sha256(Path(rerun["diffregcd_config"])) != rerun["diffregcd_config_sha256"]:
        raise SystemExit(f"{rerun['diffregcd_config']} was changed")
    config, evaluation = load_config(Path(rerun["diffregcd_config"]))
    return rerun, config, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_diffregcd_rerun.yaml")
    parser.add_argument("--output-dir", required=True, help="resumable: finished part/dataset files are kept")
    args = parser.parse_args()
    rerun, config, evaluation = load(Path(args.config))
    root = Path(config["external"]["root"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != config["external"]["commit"]:
        raise SystemExit(f"DiffRegCD commit {commit} is not {config['external']['commit']}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from tools.evaluate_diffregcd_external import ExternalDiffRegCD

    device = torch.device("cuda")
    specs = {spec["name"]: {**spec, "split": config["split"]} for spec in evaluation["datasets"]}
    names = list(config["datasets"])
    top_level = harness(evaluation)[1][-1]
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    for part in rerun["parts"]:
        results = {}
        for name in names:
            path = output_dir / part / f"{name}.json"
            if path.is_file():
                results[name] = json.loads(path.read_text(encoding="utf-8"))
                continue
            audit = audit_weights(config, name)
            adapter = ExternalDiffRegCD(root, config_path=root / config["datasets"][name]["config_json"],
                                        checkpoint_paths=weight_paths(config, name))
            result = run_part(part, specs[name], evaluation, lambda a, b: adapter.infer(a, b),
                              margin=int(rerun["crop_margin"]), cap=rerun["mechanism_crops"].get(name),
                              seed=int(config["seed"]), threshold=float(config["probability_threshold"]), device=device)
            result.update(dataset=name, weights=audit, external_commit=commit, config_sha256=sha256(Path(args.config)))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results[name] = result
            print(render(name, result), flush=True)
            del adapter
            torch.cuda.empty_cache()
        if part == "mechanism":
            summary["mechanism"] = {name: dates_reading(result) for name, result in results.items()}
            summary["mechanism"]["verdict"] = summary["mechanism"]["LEVIR-CD"]["verdict"]
        elif part == "border_free":
            summary["border_free"] = border_reading(results, top_level)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"===== {part} done: {json.dumps(summary.get(part), indent=1) if part in summary else 'rows only'}",
              flush=True)


if __name__ == "__main__":
    main()
