"""P26 step 1: DiffRegCD's per-dataset registration head against the P25
affine perturbation (docs/P26_DIFFREGCD_REGISTRATION_PROTOCOL.md).

Runs in the DiffRegCD environment. Crops, perturbations, success rule and
summaries are those of P25 step 1; only the localizer differs. P26 perturbed
T1 and fed it as the author "A", following the README; the P26-A amendment
(docs/P26A_DIFFREGCD_INPUT_ORDER_AMENDMENT.md) feeds the unperturbed T1 as A
and the perturbed T2 as B, as the network requires (main record 191.3). A
self-check on self-pairs first calibrates the flow convention. No detector
runs; test splits are refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rs_cd.affine_crops import TileCache, select_crops, selfcheck_crops
from rs_cd.affine_registration import (
    CLEAN_LEVEL,
    AffineRanges,
    affine_parameters,
    compose,
    dataset_summary,
    fit_similarity,
    grid_points,
    is_success,
    mean_displacement,
    perturb,
    perturbation_matrix,
    transfer_error,
)
from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_flow_registration import (
    ALL_CONVENTIONS,
    INPUT_ORDERS,
    calibration_outcome,
    choose_convention,
    coarse_points,
    correspondences,
    normalized_to_pixels,
    ordered_inputs,
    parse_convention,
)
from rs_cd.robustness import _stable_seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_with_line_comments(path: Path) -> dict[str, Any]:
    text = "".join(
        line.split("//", maxsplit=1)[0] + "\n"
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    return json.loads(text)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    p26 = yaml.safe_load(path.read_text(encoding="utf-8"))
    if p26.get("status") != "preregistered":
        raise SystemExit("P26 config is not marked preregistered")
    p25_path = Path(p26["p25_config"])
    digest = sha256(p25_path)
    if digest != p26["p25_config_sha256"]:
        raise SystemExit(f"{p25_path} has sha256 {digest}, P26 expects {p26['p25_config_sha256']}")
    p25 = yaml.safe_load(p25_path.read_text(encoding="utf-8"))
    wanted = list(p26["datasets"])
    specs = {spec["name"]: spec for spec in p25["datasets"]}
    missing = [name for name in wanted if name not in specs]
    if missing:
        raise SystemExit(f"P26 datasets not in P25: {missing}")
    for name in wanted:
        if specs[name]["split"] != "val":
            raise SystemExit(f"{name}: P26 reads validation splits only")
        if name not in p26["external"]["registration_heads"]:
            raise SystemExit(f"{name}: no registration head configured")
    if p26.get("input_order", "A_perturbed") not in INPUT_ORDERS:
        raise SystemExit(f"unknown input_order {p26.get('input_order')!r}")
    if p26.get("perturbation_target", "t1") not in ("t1", "t2"):
        raise SystemExit(f"unknown perturbation_target {p26.get('perturbation_target')!r}")
    return p26, p25


def roles(t1: np.ndarray, t2: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    """(reference, image to perturb) for the configured perturbation target."""

    return (t2, t1) if target == "t1" else (t1, t2)


class DiffRegCDRegistration:
    """Frozen DDPM features + one dataset's registration head (no CD head)."""

    def __init__(self, external_root: Path, *, config_json: str, backbone: Path, head: Path,
                 timestep: int, level: int) -> None:
        if not torch.cuda.is_available():
            raise SystemExit("DiffRegCD requires CUDA")
        root = str(external_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        previous = Path.cwd()
        try:
            os.chdir(root)
            import importlib

            author_model = importlib.import_module("model")
            losses = importlib.import_module("losses.roma_flow_cls_256")
            opt = read_json_with_line_comments(Path(root) / config_json)
            opt.update(phase="test", gpu_ids=[0], distributed=False, enable_wandb=False,
                       len_train_dataloader=0, len_val_dataloader=0, len_test_dataloader=0)
            opt["path"]["resume_state"] = str(backbone)[: -len("_gen.pth")]
            opt["path_reg"]["resume_state"] = str(head)[: -len("_gen.pth")]
            opt["path_cd"]["resume_state"] = None  # registration only
            self.diffusion = author_model.create_model(opt)
            self.diffusion.set_new_noise_schedule(opt["model"]["beta_schedule"]["test"], schedule_phase="test")
            self.joint = author_model.create_CD_model_256(opt)
            self.softargmax = losses._softargmax_flow
            # Training registers the features of the first configured timestep.
            first = int(opt["model_cd"]["t"][0])
            if first != int(timestep) or opt["model_cd"]["feat_type"] != "dec":
                raise SystemExit(f"{config_json}: first timestep {first}, feat_type {opt['model_cd']['feat_type']}")
        finally:
            os.chdir(previous)
        self.timestep = int(timestep)
        self.level = int(level)
        self.diffusion.netG.eval()
        self.joint.netReg.eval()

    @staticmethod
    def _tensor(image: np.ndarray) -> torch.Tensor:
        unit = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        return (unit * 2.0 - 1.0).unsqueeze(0).cuda()

    @torch.inference_mode()
    def flows(self, image_a: np.ndarray, image_b: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        """Flows for the author inputs A and B; callers decide which image is which."""

        seed_all(seed)
        self.diffusion.feed_data({"A": self._tensor(image_a), "B": self._tensor(image_b)})
        _, decoder_a, _, decoder_b = self.diffusion.get_feats(t=self.timestep)
        net = self.joint.netReg
        model_flow, logits = net(decoder_a[self.level], decoder_b[self.level])
        classes = logits.shape[1]
        side = int(round(math.sqrt(classes)))
        if side * side != classes:
            raise SystemExit(f"registration logits have {classes} classes, not a square")
        loss_flow = self.softargmax(logits, side, net.temperature).permute(0, 2, 3, 1)
        return {
            "model": model_flow[0].float().cpu().numpy(),
            "loss": loss_flow[0].float().cpu().numpy(),
        }


def estimate(flows: dict[str, np.ndarray], convention: str, size: int, fit: dict[str, Any], seed: int):
    decoding, sign, grid = parse_convention(convention)
    flow = flows[decoding]
    points = coarse_points(flow.shape[0], flow.shape[1], size, size)
    source, target = correspondences(normalized_to_pixels(flow, size, size), points, sign=sign, grid=grid)
    return fit_similarity(source, target, seed=seed, **fit)


def timed(function, *args, **kwargs):
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if cuda:
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(p26: dict[str, Any], p25: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    external = p26["external"]
    root = Path(external["root"])
    perturbation = p25["perturbation"]
    step1 = p25["step1"]
    size = int(perturbation["crop_size"])
    seed = int(p25["seed"])
    low, high = perturbation["scale_range"]
    ranges = AffineRanges(float(perturbation["max_rotation_deg"]), float(low), float(high),
                          float(perturbation["max_translation_px"]), len(perturbation["levels"]))
    levels = [int(level) for level in perturbation["levels"]]
    points = grid_points(size, size, int(step1["grid_points"]))
    fit = dict(p26["fit"])
    trigger_px = float(p26["trigger_mean_displacement_px"])
    tolerance = float(step1["success_ate_px"])
    specs = {spec["name"]: spec for spec in p25["datasets"]}
    input_order = p26.get("input_order", "A_perturbed")
    target = p26.get("perturbation_target", "t1")
    continue_imprecise = bool(p26["selfcheck"].get("continue_when_imprecise", False))

    backbone = root / external["backbone"]["path"]
    weights = {"backbone": {"path": str(backbone), "sha256": sha256(backbone)}}
    if weights["backbone"]["sha256"] != external["backbone"]["sha256"]:
        raise SystemExit(f"backbone sha256 {weights['backbone']['sha256']} differs from the config")
    decision: dict[str, Any] = {
        "gate": p26.get("gate", "p26_step1"),
        "localizer": "DiffRegCD per-dataset registration head + RANSAC similarity",
        "perturbation_target": target,
        "input_order": input_order,
        "labels_read": "validation masks, change_fraction column only (descriptive)",
        "test_split_touched": False,
        "fit": fit,
        "datasets": {},
    }
    all_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for name in p26["datasets"]:
        spec = specs[name]
        head_spec = external["registration_heads"][name]
        head = root / head_spec["path"]
        head_sha = sha256(head)
        weights[name] = {"path": str(head), "sha256": head_sha}
        if head_sha != head_spec["sha256"]:
            raise SystemExit(f"{name}: registration head sha256 {head_sha} differs from the config")
        adapter = DiffRegCDRegistration(
            root, config_json=head_spec["config_json"], backbone=backbone, head=head,
            timestep=int(external["feature_timestep"]), level=int(external["feature_level"]),
        )
        dataset = PairedChangeDataset(spec["dataset_root"], spec["manifest"], split=spec["split"],
                                      normalization="unit", mask_threshold=int(spec["mask_threshold"]))
        crops = select_crops(dataset.rows, int(spec["tile_size"]), size, int(step1["max_crops_per_dataset"]))
        cache = TileCache(dataset, int(spec["tile_size"]))

        def pair_flows(reference: np.ndarray, perturbed: np.ndarray, pair_seed: int) -> dict[str, np.ndarray]:
            return adapter.flows(*ordered_inputs(reference, perturbed, input_order), pair_seed)

        # Convention calibration on self-pairs (T1 against its own level-3 perturbation).
        worst = {key: 0.0 for key in ALL_CONVENTIONS}
        for item in selfcheck_crops(crops, cache, size, int(step1["selfcheck_pool"]), int(step1["selfcheck_crops"])):
            t1, _, _ = cache.crop(item, size)
            matrix = perturbation_matrix(*affine_parameters(item["crop_id"], levels[-1], ranges, seed),
                                         width=size, height=size)
            crop_seed = _stable_seed(seed, item["crop_id"], "p26", 0, "diffusion")
            flows = pair_flows(t1, perturb(t1, matrix), crop_seed)
            for key in ALL_CONVENTIONS:
                result = estimate(flows, key, size, fit, crop_seed)
                error = (transfer_error(result.matrix, matrix, points, width=size, height=size)
                         if result.converged else None)
                calibration_rows.append({"dataset": name, "crop_id": item["crop_id"], "convention": key,
                                         "converged": result.converged, "ate_raw": error})
                worst[key] = max(worst[key], math.inf if error is None else error)
        calibration = choose_convention(worst, tolerance=float(p26["selfcheck"]["tolerance_px"]),
                                        ambiguity_ratio=float(p26["selfcheck"]["ambiguity_ratio"]))
        print(f"[calibration] {name}: {json.dumps(calibration)}")
        outcome = calibration_outcome(calibration, continue_imprecise)
        if outcome == "stop":
            decision["datasets"][name] = {"verdict": "selfcheck_failed", "calibration": calibration}
            del adapter
            torch.cuda.empty_cache()
            continue
        convention = calibration["selected"]

        rows = []
        for index, item in enumerate(crops, start=1):
            t1, t2, mask = cache.crop(item, size)
            reference, moving = roles(t1, t2, target)
            change = float((mask > int(spec["mask_threshold"])).mean())
            crop_seed = _stable_seed(seed, item["crop_id"], "p26", 0, "diffusion")
            clean_estimate = None
            for level in [CLEAN_LEVEL, *levels]:
                angle, scale, tx, ty = affine_parameters(item["crop_id"], level, ranges, seed)
                matrix = perturbation_matrix(angle, scale, tx, ty, width=size, height=size)
                perturbed = moving if level == CLEAN_LEVEL else perturb(moving, matrix)
                flows, seconds = timed(pair_flows, reference, perturbed, crop_seed)
                result = estimate(flows, convention, size, fit, crop_seed)
                if level == CLEAN_LEVEL:
                    clean_estimate = result
                ate_raw = transfer_error(result.matrix, matrix, points, width=size, height=size)
                ate_diff = None
                if level != CLEAN_LEVEL and clean_estimate is not None and clean_estimate.converged:
                    ate_diff = transfer_error(result.matrix, compose(matrix, clean_estimate.matrix),
                                              points, width=size, height=size)
                displacement = mean_displacement(result.matrix, points)
                row = {
                    "dataset": name, "crop_id": item["crop_id"], "level": level,
                    "angle": angle, "scale": scale, "tx": tx, "ty": ty,
                    "perturbation_px": mean_displacement(matrix, points),
                    "converged": result.converged, "inliers": result.inliers,
                    "inlier_fraction": result.inlier_fraction,
                    "a11": result.matrix[0, 0], "a12": result.matrix[0, 1], "a13": result.matrix[0, 2],
                    "a21": result.matrix[1, 0], "a22": result.matrix[1, 1], "a23": result.matrix[1, 2],
                    "estimate_displacement_px": displacement,
                    "trigger": result.converged and displacement > trigger_px,
                    "ate_raw": ate_raw, "ate_diff": ate_diff,
                    "seconds": seconds, "change_fraction": change,
                }
                row["success"] = is_success(row, tolerance) if level != CLEAN_LEVEL else None
                rows.append(row)
            if index % 50 == 0 or index == len(crops):
                print(f"[p26 {name} {index}/{len(crops)}]")
        summary = dataset_summary(rows, step1)
        if outcome == "descriptive":
            # Measured only as a description: the self-check missed its tolerance.
            summary["descriptive_verdict"] = summary["verdict"]
            summary["verdict"] = "selfcheck_imprecise"
        summary.update(calibration=calibration, crops=len(crops), tiles=len(dataset.rows))
        decision["datasets"][name] = summary
        all_rows.extend(rows)
        del adapter
        torch.cuda.empty_cache()

    decision["weights"] = weights
    decision["verdicts"] = {name: entry["verdict"] for name, entry in decision["datasets"].items()}
    write_csv(output_dir / "step1_estimates.csv", all_rows)
    write_csv(output_dir / "calibration.csv", calibration_rows)
    (output_dir / "step1_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p26_diffregcd_registration.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    p26, p25 = load_configs(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = run(p26, p25, output_dir)
    print(json.dumps(result["verdicts"], indent=2))


if __name__ == "__main__":
    main()
