from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.external_diffregcd import (
    ConfusionCounts,
    clean_reproduction_audit,
    external_test_summary,
    nonoverlapping_tiles,
    same_support_decision,
    stable_inference_seed,
    support_digest,
)
from rs_cd.geometry import interior_valid_mask
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec


EXPECTED_COMMIT = "6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38"
INFERENCE_PATHS = ("official_unmodified", "intended_warped")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_json_with_line_comments(path: Path) -> dict[str, Any]:
    text = "".join(
        line.split("//", maxsplit=1)[0] + "\n"
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _checkpoint_prefix(path: Path) -> str:
    suffix = "_gen.pth"
    text = str(path)
    if not text.endswith(suffix):
        raise ValueError(f"Checkpoint must end in {suffix}: {path}")
    return text[: -len(suffix)]


def _seed_inference(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scenario_specs(protocol: dict[str, Any]) -> list[CorruptionSpec]:
    seed = int(protocol["seed"])
    specs = []
    for row in protocol["scenarios"]:
        name = str(row["name"])
        spec = CorruptionSpec(
            name=name,
            severity=int(row["severity"]),
            mode=str(row["mode"]),
            seed=seed,
        )
        if name == "shift" and float(row["pixels"]) != spec.value:
            raise ValueError(
                f"Scenario {spec.key} declares pixels={row['pixels']} "
                f"but the frozen corruption protocol uses {spec.value}"
            )
        specs.append(spec)
    expected = [
        "clean:s0:common",
        "shift:s1:t2",
        "shift:s2:t2",
        "shift:s3:t2",
    ]
    actual = [spec.key for spec in specs]
    if actual != expected:
        raise ValueError(
            f"P13 scenario order changed: expected {expected}, got {actual}"
        )
    return specs


def _audit_weights(
    external_root: Path,
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    results: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, config in protocol["weights"].items():
        path = (
            external_root / str(config["relative_path"])
        ).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing DiffRegCD weight: {path}")
        actual = _sha256(path)
        expected = str(config["sha256"]).lower()
        paths[str(name)] = path
        results[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
    return results, paths


class ExternalDiffRegCD:
    """Read-only adapter around the frozen author implementation."""

    def __init__(
        self,
        external_root: Path,
        *,
        config_path: Path,
        checkpoint_paths: dict[str, Path],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DiffRegCD author implementation requires CUDA"
            )
        self.external_root = external_root.resolve()
        root_text = str(self.external_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        previous_cwd = Path.cwd()
        try:
            os.chdir(self.external_root)
            author_model = importlib.import_module("model")
            author_warp = importlib.import_module(
                "registration.utils256"
            )
            opt = _read_json_with_line_comments(config_path)
            opt["phase"] = "test"
            opt["gpu_ids"] = [0]
            opt["distributed"] = False
            opt["enable_wandb"] = False
            opt["len_train_dataloader"] = 0
            opt["len_val_dataloader"] = 0
            opt["len_test_dataloader"] = 0
            opt["path"]["resume_state"] = _checkpoint_prefix(
                checkpoint_paths["backbone"]
            )
            opt["path_reg"]["resume_state"] = _checkpoint_prefix(
                checkpoint_paths["registration"]
            )
            opt["path_cd"]["resume_state"] = _checkpoint_prefix(
                checkpoint_paths["change_detection"]
            )
            self.opt = opt
            self.diffusion = author_model.create_model(opt)
            self.diffusion.set_new_noise_schedule(
                opt["model"]["beta_schedule"]["test"],
                schedule_phase="test",
            )
            self.joint = author_model.create_CD_model_256(opt)
            self.warp_features_all = author_warp.warp_features_all
            self.warp_single_feature = author_warp.warp_single_feature
        finally:
            os.chdir(previous_cwd)

        self.diffusion.netG.eval()
        self.joint.netCD.eval()
        self.joint.netReg.eval()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def infer(
        self,
        image_a_t1: torch.Tensor,
        image_b_t2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        data = {"A": image_a_t1, "B": image_b_t2}
        self.diffusion.feed_data(data)
        features_a: list[list[torch.Tensor]] = []
        features_b: list[list[torch.Tensor]] = []
        for timestep in self.opt["model_cd"]["t"]:
            encoder_a, decoder_a, encoder_b, decoder_b = (
                self.diffusion.get_feats(t=int(timestep))
            )
            if self.opt["model_cd"]["feat_type"] == "dec":
                features_a.append(decoder_a)
                features_b.append(decoder_b)
                del encoder_a, encoder_b
            else:
                features_a.append(encoder_a)
                features_b.append(encoder_b)
                del decoder_a, decoder_b

        # This is the frozen author test wiring.
        official = self.joint.netCD(features_a, features_b)

        # This is the sole preregistered minimal wiring repair: reproduce the
        # feature warp used by optimize_parameters(), without changing any
        # model, checkpoint, decoder, threshold, or loss.
        registration_a = features_a[0][11]
        registration_b = features_b[0][11]
        predicted_flow, registration_logits = self.joint.netReg(
            registration_a,
            registration_b,
        )
        warped_b = self.warp_features_all(
            features_b,
            predicted_flow.detach().clone(),
        )
        intended = self.joint.netCD(features_a, warped_b)
        return {
            "official_unmodified": official,
            "intended_warped": intended,
            "predicted_flow": predicted_flow,
            "registration_logits": registration_logits,
        }

    @torch.inference_mode()
    def direction_audit(self) -> dict[str, Any]:
        height = width = 16
        reference = torch.arange(
            height * width,
            dtype=torch.float32,
            device="cuda",
        ).view(1, 1, height, width) / float(height * width)
        target = torch.empty_like(reference)
        target[:, :, :, 1:] = reference[:, :, :, :-1]
        target[:, :, :, 0] = reference[:, :, :, 0]
        magnitude = 2.0 / float(width - 1)
        flow = torch.zeros(
            1,
            height,
            width,
            2,
            dtype=torch.float32,
            device="cuda",
        )
        flow[..., 0] = magnitude
        positive = self.warp_single_feature(target, flow)
        zero = self.warp_single_feature(target, torch.zeros_like(flow))
        negative = self.warp_single_feature(target, -flow)
        interior = (..., slice(1, -1), slice(1, -1))

        def error(value: torch.Tensor) -> float:
            return float(
                (value[interior] - reference[interior]).abs().mean()
            )

        errors = {
            "positive_b_to_a_flow": error(positive),
            "zero_flow": error(zero),
            "negative_flow": error(negative),
        }
        passed = (
            errors["positive_b_to_a_flow"] < errors["zero_flow"]
            and errors["positive_b_to_a_flow"]
            < errors["negative_flow"]
            and errors["positive_b_to_a_flow"] < 1e-4
        )
        return {
            "passed": passed,
            "project_convention": (
                "positive dx means T2 content moved right relative to T1"
            ),
            "external_convention": (
                "positive normalized dx samples B at x+dx to align B to A"
            ),
            "errors": errors,
        }


def _update_counts(
    counts: ConfusionCounts,
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold: float,
) -> None:
    probability = torch.softmax(logits.float(), dim=1)[:, 1:2]
    prediction = probability >= threshold
    truth = targets >= 0.5
    keep = valid >= 0.5
    prediction = prediction[keep]
    truth = truth[keep]
    counts.update(
        true_positive=int((prediction & truth).sum()),
        false_positive=int((prediction & ~truth).sum()),
        false_negative=int((~prediction & truth).sum()),
        true_negative=int((~prediction & ~truth).sum()),
    )


def _tile_tensor(
    tensor: torch.Tensor,
    coordinates: list[tuple[int, int, int, int]],
) -> torch.Tensor:
    return torch.stack(
        [
            tensor[..., top:bottom, left:right]
            for top, left, bottom, right in coordinates
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _smoke(
    adapter: ExternalDiffRegCD,
    *,
    seed: int,
) -> dict[str, Any]:
    _seed_inference(seed)
    image_a = torch.rand(1, 3, 256, 256, device="cuda") * 2.0 - 1.0
    image_b = image_a.clone()
    started = time.perf_counter()
    outputs = adapter.infer(image_a, image_b)
    elapsed = time.perf_counter() - started
    checks = {
        name: {
            "shape": list(outputs[name].shape),
            "finite": bool(torch.isfinite(outputs[name]).all()),
        }
        for name in (*INFERENCE_PATHS, "predicted_flow")
    }
    passed = (
        all(value["finite"] for value in checks.values())
        and checks["official_unmodified"]["shape"] == [1, 2, 256, 256]
        and checks["intended_warped"]["shape"] == [1, 2, 256, 256]
    )
    return {
        "status": "completed" if passed else "failed",
        "uses_validation_data": False,
        "seed": seed,
        "seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()
        / (1024**3),
        "outputs": checks,
    }


@torch.inference_mode()
def _evaluate(
    adapter: ExternalDiffRegCD,
    *,
    protocol: dict[str, Any],
    dataset_root: str,
    manifest: str,
    split: str,
    max_samples: int | None,
    inference_paths: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if split != str(protocol["split"]):
        raise ValueError(
            f"Protocol split is {protocol['split']}, requested {split}"
        )
    if split == "test" and not bool(
        protocol["test_policy"]["run_test"]
    ):
        raise ValueError("P13 test remains sealed by the protocol")

    base_dataset = PairedChangeDataset(
        dataset_root=dataset_root,
        manifest_path=manifest,
        split=split,
        image_size=None,
        normalization="unit",
        augment=False,
        mask_threshold=127,
        cache_mode="ram",
    )
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        sample_count = min(max_samples, len(base_dataset))
    else:
        sample_count = len(base_dataset)
    tile_size = int(protocol["input"]["tile_size"])
    tile_batch_size = int(protocol["input"]["tile_batch_size"])
    threshold = float(
        protocol["evaluation"]["probability_threshold"]
    )
    border = int(protocol["evaluation"]["interior_border_pixels"])
    seed = int(protocol["seed"])
    rows: list[dict[str, Any]] = []
    support: dict[str, dict[str, Any]] = {}

    for scenario_index, spec in enumerate(_scenario_specs(protocol)):
        print(
            f"[P13 {scenario_index + 1}/4] {spec.key} "
            f"samples={sample_count}"
        )
        dataset = CorruptedPairDataset(
            base_dataset,
            spec,
            normalization="unit",
        )
        accumulators = {
            name: ConfusionCounts() for name in inference_paths
        }
        support_rows: list[tuple[str, bytes]] = []
        scenario_started = time.perf_counter()
        for sample_index in range(sample_count):
            sample = dataset[sample_index]
            sample_id = str(sample["id"])
            image_a = sample["image_t1"]
            image_b = sample["image_t2"]
            target = sample["mask"]
            height, width = target.shape[-2:]
            if (
                height != int(protocol["input"]["original_image_size"])
                or width != int(
                    protocol["input"]["original_image_size"]
                )
            ):
                raise ValueError(
                    f"{sample_id} is {height}x{width}; P13 requires "
                    f"{protocol['input']['original_image_size']}x"
                    f"{protocol['input']['original_image_size']}"
                )
            valid = sample["valid_mask"].float()
            valid = valid * interior_valid_mask(
                1,
                height,
                width,
                border,
            ).squeeze(0)
            packed = np.packbits(
                (valid >= 0.5).numpy().astype(np.uint8),
                bitorder="little",
            ).tobytes()
            support_rows.append((sample_id, packed))

            coordinates = nonoverlapping_tiles(
                height,
                width,
                tile_size,
            )
            tiles_a = _tile_tensor(image_a, coordinates)
            tiles_b = _tile_tensor(image_b, coordinates)
            tiles_target = _tile_tensor(target, coordinates)
            tiles_valid = _tile_tensor(valid, coordinates)
            for start in range(0, len(coordinates), tile_batch_size):
                stop = min(start + tile_batch_size, len(coordinates))
                inference_seed = stable_inference_seed(
                    seed,
                    sample_id,
                    start,
                )
                _seed_inference(inference_seed)
                batch_a = (
                    tiles_a[start:stop].to(
                        "cuda",
                        non_blocking=True,
                    )
                    * 2.0
                    - 1.0
                )
                batch_b = (
                    tiles_b[start:stop].to(
                        "cuda",
                        non_blocking=True,
                    )
                    * 2.0
                    - 1.0
                )
                batch_target = tiles_target[start:stop].to(
                    "cuda",
                    non_blocking=True,
                )
                batch_valid = tiles_valid[start:stop].to(
                    "cuda",
                    non_blocking=True,
                )
                outputs = adapter.infer(batch_a, batch_b)
                for method in inference_paths:
                    _update_counts(
                        accumulators[method],
                        outputs[method],
                        batch_target,
                        batch_valid,
                        threshold=threshold,
                    )
                del outputs, batch_a, batch_b, batch_target, batch_valid

            if (
                (sample_index + 1) % 4 == 0
                or sample_index + 1 == sample_count
            ):
                elapsed = time.perf_counter() - scenario_started
                print(
                    f"  {sample_index + 1}/{sample_count} "
                    f"elapsed={elapsed / 60.0:.1f} min"
                )

        valid_pixels = accumulators[inference_paths[0]].valid_pixels
        if any(
            value.valid_pixels != valid_pixels
            for value in accumulators.values()
        ):
            raise RuntimeError(
                f"Inference paths used different support in {spec.key}"
            )
        support[spec.key] = {
            "sha256": support_digest(support_rows),
            "valid_pixels": valid_pixels,
            "samples": sample_count,
        }
        seconds = time.perf_counter() - scenario_started
        for method, accumulator in accumulators.items():
            rows.append(
                {
                    "scenario": spec.key,
                    "corruption": spec.name,
                    "severity": spec.severity,
                    "severity_value": spec.value,
                    "mode": spec.mode,
                    "method": method,
                    **accumulator.compute(),
                    "samples": sample_count,
                    "scenario_seconds": seconds,
                }
            )
        print(
            "  "
            + " ".join(
                f"{method} change_F1="
                f"{accumulators[method].compute()['change_f1']:.6f}"
                for method in inference_paths
            )
        )
    return rows, support


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preregistered P13 DiffRegCD validation adapter. It reports both "
            "the frozen official test wiring and the sole intended-warped "
            "repair without modifying the external repository."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--external-config",
        default="config/levir_test.json",
    )
    parser.add_argument(
        "--protocol",
        default="configs/p13_diffregcd_validation.yaml",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    external_root = Path(args.external_root).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(
        protocol_path.read_text(encoding="utf-8")
    )
    protocol_status = str(protocol["status"])
    if protocol_status not in {
        "preregistered",
        "frozen_external_test",
    }:
        raise ValueError(f"Unsupported P13 protocol status: {protocol_status}")
    if str(protocol["external_commit"]) != EXPECTED_COMMIT:
        raise ValueError("P13 external commit changed")
    frozen_input = protocol["input"]
    if (
        int(frozen_input["tile_overlap"]) != 0
        or not bool(frozen_input["shift_before_tiling"])
        or not bool(
            frozen_input["common_random_numbers_across_scenarios"]
        )
        or str(frozen_input["normalization"]) != "minus_one_to_one"
    ):
        raise ValueError("P13 frozen input protocol changed")
    inference_paths = tuple(protocol["evaluation"]["inference_paths"])
    if protocol_status == "preregistered":
        if str(protocol["split"]) != "val":
            raise ValueError("Preregistered P13 protocol must use val")
        if inference_paths != INFERENCE_PATHS:
            raise ValueError("P13 validation inference paths changed")
    else:
        if str(protocol["split"]) != "test":
            raise ValueError("Frozen external test protocol must use test")
        if inference_paths != ("official_unmodified",):
            raise ValueError(
                "Frozen external test must use only official_unmodified"
            )
        if not bool(
            protocol["test_policy"].get(
                "reporting_only_no_selection",
                False,
            )
        ):
            raise ValueError("External test must be reporting-only")
    actual_commit = _git_commit(external_root)
    if actual_commit != EXPECTED_COMMIT:
        raise ValueError(
            f"DiffRegCD commit mismatch: {actual_commit}"
        )
    weight_audit, checkpoint_paths = _audit_weights(
        external_root,
        protocol,
    )
    if not all(row["passed"] for row in weight_audit.values()):
        raise ValueError("One or more DiffRegCD weight hashes differ")
    external_config = (
        external_root / args.external_config
    ).resolve()

    torch.cuda.reset_peak_memory_stats()
    adapter = ExternalDiffRegCD(
        external_root,
        config_path=external_config,
        checkpoint_paths=checkpoint_paths,
    )
    direction = adapter.direction_audit()
    if not direction["passed"]:
        raise RuntimeError("External warp direction audit failed")
    common = {
        "external_commit": actual_commit,
        "external_config": str(external_config),
        "external_config_sha256": _sha256(external_config),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "weights": weight_audit,
        "direction_audit": direction,
    }

    if args.smoke_only:
        smoke = _smoke(adapter, seed=int(protocol["seed"]))
        result = {**common, **smoke}
        (output_dir / "smoke.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if smoke["status"] != "completed":
            raise SystemExit(2)
        return

    if not args.dataset_root or not args.manifest:
        parser.error(
            "--dataset-root and --manifest are required unless --smoke-only"
        )
    if args.max_samples is not None:
        raise ValueError(
            "P13 full evaluation forbids max_samples; use smoke-only for "
            "engineering checks without consuming held-out examples"
        )

    freeze_manifest: dict[str, Any] | None = None
    if args.split == "test":
        if not args.freeze_manifest:
            raise ValueError(
                "Frozen external test requires --freeze-manifest"
            )
        freeze_path = Path(args.freeze_manifest).expanduser().resolve()
        freeze_manifest = json.loads(
            freeze_path.read_text(encoding="utf-8")
        )
        if freeze_manifest.get("status") != "frozen_external_test":
            raise ValueError("External test freeze manifest is invalid")
        if (
            freeze_manifest.get("selected_external_path")
            != "official_unmodified"
        ):
            raise ValueError("External test path was not frozen")
        if not bool(freeze_manifest.get("test_authorized", False)):
            raise ValueError("External test was not authorized")
        frozen_validation = Path(
            freeze_manifest["validation_results"]
        ).expanduser().resolve()
        if (
            _sha256(frozen_validation)
            != str(freeze_manifest["validation_results_sha256"])
        ):
            raise ValueError(
                "Frozen P13 validation result changed before test"
            )
        expected_protocol_sha = str(
            freeze_manifest["test_protocol_sha256"]
        )
        actual_protocol_sha = _sha256(protocol_path)
        if actual_protocol_sha != expected_protocol_sha:
            raise ValueError("Frozen external test protocol hash changed")

    rows, support = _evaluate(
        adapter,
        protocol=protocol,
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        split=args.split,
        max_samples=None,
        inference_paths=inference_paths,
    )
    support_gate = same_support_decision(support)
    engineering_passed = (
        support_gate["passed"]
        and direction["passed"]
        and all(row["passed"] for row in weight_audit.values())
    )
    execution = {
        "shift_before_tiling": True,
        "tile_size": int(protocol["input"]["tile_size"]),
        "tile_overlap": int(protocol["input"]["tile_overlap"]),
        "tile_batch_size": int(
            protocol["input"]["tile_batch_size"]
        ),
        "common_random_numbers_across_scenarios": True,
        "threshold": float(
            protocol["evaluation"]["probability_threshold"]
        ),
        "interior_border_pixels": int(
            protocol["evaluation"]["interior_border_pixels"]
        ),
        "peak_allocated_gib": torch.cuda.max_memory_allocated()
        / (1024**3),
    }
    if args.split == "test":
        if not engineering_passed:
            raise RuntimeError(
                "External test engineering gates failed after consumption"
            )
        summary = external_test_summary(
            rows,
            method="official_unmodified",
        )
        result = {
            **common,
            **summary,
            "split": "test",
            "test_consumed": True,
            "freeze_manifest": freeze_manifest,
            "execution": execution,
            "support": support,
            "support_gate": support_gate,
            "scenario_results": rows,
        }
        _write_csv(output_dir / "scenario_metrics.csv", rows)
        (output_dir / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "final_test_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "protocol_resolved.yaml").write_text(
            yaml.safe_dump(
                protocol,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved one-shot P13 external test to {output_dir}")
        return

    clean_audit = clean_reproduction_audit(
        rows,
        protocol["author_reference"],
    )
    decision = {
        "status": "completed",
        "engineering_gates_passed": engineering_passed,
        "clean_reproduction_status": clean_audit["status"],
        "test_authorized": False,
        "next_action": (
            "manual_clean_reproduction_review_before_freezing_external_test"
            if engineering_passed
            else "diagnose_adapter_without_running_external_test"
        ),
    }
    result = {
        **common,
        "status": "completed",
        "reporting_only_validation": True,
        "split": args.split,
        "test_consumed": False,
        "execution": execution,
        "support": support,
        "support_gate": support_gate,
        "clean_reproduction_audit": clean_audit,
        "scenario_results": rows,
        "decision": decision,
    }
    _write_csv(output_dir / "scenario_metrics.csv", rows)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Saved P13 validation to {output_dir}")
    if not engineering_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
