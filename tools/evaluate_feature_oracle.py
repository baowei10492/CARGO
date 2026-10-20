from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.geometry import interior_valid_mask, warp_reference_from_target
from rs_cd.metrics import BinaryChangeMetrics
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_hash(protocol: dict[str, Any]) -> str:
    text = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw, forced feature alignment, checkpoint blend and "
            "image-level Oracle from one P5c checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p7_feature_oracle_t2.yaml",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Feature Oracle protocol must be a mapping")

    device = select_device(args.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = copy.deepcopy(checkpoint["config"])
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest
    seed = int(protocol["seed"])
    seed_everything(seed)
    split = str(args.split or protocol.get("split", "val"))
    data = config["data"]
    base_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=split,
        image_size=data.get("eval_crop_size"),
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if not hasattr(model, "forward_with_forced_offset"):
        raise TypeError(
            "Feature-path Oracle requires a global-registration checkpoint"
        )

    specs = [
        CorruptionSpec(
            name="clean",
            severity=0,
            mode="common",
            seed=seed,
        ),
        *[
            CorruptionSpec(
                name="shift",
                severity=int(severity),
                mode="t2",
                seed=seed,
            )
            for severity in protocol["shift_severities"]
        ],
    ]
    variants = (
        "raw",
        "feature_aligned",
        "checkpoint_blend",
        "image_oracle",
    )
    rows: list[dict[str, Any]] = []
    feature_strides: list[tuple[float, float]] | None = None

    for scenario_index, spec in enumerate(specs):
        print(f"[{scenario_index + 1}/{len(specs)}] {spec.key}")
        dataset = CorruptedPairDataset(
            base_dataset,
            spec,
            normalization=str(data.get("normalization", "imagenet")),
        )
        loader = _loader(
            dataset=dataset,
            batch_size=int(protocol["batch_size"]),
            shuffle=False,
            num_workers=int(protocol["num_workers"]),
            seed=seed + scenario_index,
            device=device,
            prefetch_factor=int(protocol["prefetch_factor"]),
        )
        accumulators = {
            name: BinaryChangeMetrics(
                threshold=float(protocol["threshold"])
            )
            for name in variants
        }
        sample_count = 0
        valid_pixel_count = 0
        for batch in loader:
            t1 = batch["image_t1"].to(device, non_blocking=True)
            t2 = batch["image_t2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            offset = torch.stack(
                (
                    torch.as_tensor(batch["shift_dx"]),
                    torch.as_tensor(batch["shift_dy"]),
                ),
                dim=1,
            ).to(device=device, dtype=t1.dtype)
            forced = model.forward_with_forced_offset(t1, t2, offset)
            corrected_t2 = warp_reference_from_target(t2, offset)
            zero = torch.zeros_like(offset)
            image_oracle = model.forward_with_forced_offset(
                t1,
                corrected_t2,
                zero,
            )
            logits = {
                "raw": forced["raw_change_logits"],
                "feature_aligned": forced["aligned_change_logits"],
                "checkpoint_blend": forced["blended_change_logits"],
                "image_oracle": image_oracle["raw_change_logits"],
            }
            if feature_strides is None:
                feature_strides = list(forced["feature_strides_xy"])
            valid = batch.get("valid_mask")
            if isinstance(valid, torch.Tensor):
                valid = valid.to(device, non_blocking=True)
            else:
                valid = torch.ones_like(mask)
            interior = interior_valid_mask(
                batch_size=int(t1.shape[0]),
                height=int(t1.shape[-2]),
                width=int(t1.shape[-1]),
                border=int(protocol["evaluation_border_pixels"]),
                device=device,
            )
            valid = valid * interior
            sample_count += int(t1.shape[0])
            valid_pixel_count += int(valid.sum())
            for name, values in logits.items():
                if not isinstance(values, torch.Tensor):
                    raise TypeError(f"{name} logits are not a tensor")
                accumulators[name].update(
                    values,
                    mask,
                    from_logits=True,
                    valid_mask=valid,
                )

        scenario_metrics: dict[str, dict[str, float | int]] = {}
        for name, accumulator in accumulators.items():
            values = accumulator.compute()
            scenario_metrics[name] = values
            rows.append(
                {
                    "scenario": spec.key,
                    "severity": spec.severity,
                    "shift_pixels": spec.value,
                    "method": name,
                    **values,
                    "samples": sample_count,
                    "valid_pixels": valid_pixel_count,
                }
            )
        raw_f1 = float(scenario_metrics["raw"]["f1"])
        image_oracle_f1 = float(
            scenario_metrics["image_oracle"]["f1"]
        )
        denominator = image_oracle_f1 - raw_f1
        for name in ("feature_aligned", "checkpoint_blend"):
            method_f1 = float(scenario_metrics[name]["f1"])
            recovery = (
                (method_f1 - raw_f1) / denominator
                if denominator > 1e-12
                else None
            )
            for row in reversed(rows):
                if row["scenario"] == spec.key and row["method"] == name:
                    row["oracle_gap_recovery"] = recovery
                    break
        print(
            " ".join(
                f"{name}={float(values['f1']):.6f}"
                for name, values in scenario_metrics.items()
            )
        )

    by_key = {
        (str(row["scenario"]), str(row["method"])): row for row in rows
    }
    clean_raw = float(by_key[("clean:s0:common", "raw")]["f1"])
    clean_aligned = float(
        by_key[("clean:s0:common", "feature_aligned")]["f1"]
    )
    gates: dict[str, Any] = {
        "clean_feature_aligned_drop": {
            "value": clean_raw - clean_aligned,
            "maximum": float(protocol["clean_f1_drop_max"]),
            "passed": (
                clean_raw - clean_aligned
                <= float(protocol["clean_f1_drop_max"])
            ),
        }
    }
    for pixels, key in ((4, "shift_4px"), (8, "shift_8px")):
        scenario = next(
            spec.key
            for spec in specs
            if spec.name == "shift" and int(spec.value) == pixels
        )
        recovery = by_key[(scenario, "feature_aligned")].get(
            "oracle_gap_recovery"
        )
        minimum = float(protocol["feature_oracle_recovery_min"][key])
        gates[key] = {
            "value": recovery,
            "minimum": minimum,
            "passed": recovery is not None and float(recovery) >= minimum,
        }
    passed = all(bool(item["passed"]) for item in gates.values())
    summary = {
        "status": "passed" if passed else "failed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": str(protocol_path),
        "protocol_sha256": _protocol_hash(protocol),
        "split": split,
        "coordinate_convention": (
            "reference(x,y) corresponds to target(x+dx,y+dy)"
        ),
        "feature_strides_xy": feature_strides,
        "gates": gates,
        "results": rows,
    }
    save_json(output_dir / "results.json", summary)
    _write_csv(output_dir / "results.csv", rows)
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Saved Feature-path Oracle audit to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
