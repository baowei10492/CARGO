from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.engine import evaluate
from rs_cd.losses import BinaryChangeLoss
from rs_cd.robustness import (
    CorruptedPairDataset,
    CorruptionSpec,
    OracleInverseShiftDataset,
    PairedShiftSupportDataset,
)
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_protocol(protocol: dict[str, Any]) -> list[CorruptionSpec]:
    seed = int(protocol.get("seed", 3407))
    specs = [CorruptionSpec(name="clean", severity=0, mode="common", seed=seed)]
    for entry in protocol["corruptions"]:
        name = str(entry["name"])
        for severity in entry.get("severities", [1, 2, 3]):
            for mode in entry.get("modes", ["t1", "t2", "common"]):
                specs.append(
                    CorruptionSpec(
                        name=name,
                        severity=int(severity),
                        mode=str(mode),
                        seed=seed,
                    )
                )
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Robustness protocol contains duplicate scenarios")
    return specs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "corruption",
        "severity",
        "severity_value",
        "mode",
        "precision",
        "recall",
        "f1",
        "iou",
        "oa",
        "f1_drop",
        "relative_f1_drop",
        "valid_pixels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key) for key in fieldnames} for row in rows
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen checkpoint under deterministic shifts."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--oracle-inverse-shift",
        action="store_true",
        help="Undo known one-sided shift scenarios before inference.",
    )
    parser.add_argument(
        "--paired-shift-support",
        action="store_true",
        help=(
            "Evaluate raw and oracle shift paths on the same symmetric "
            "interior support. Pass this flag to both paired runs."
        ),
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict) or "corruptions" not in protocol:
        raise ValueError("Protocol must be a mapping with corruptions")
    specs = _expand_protocol(protocol)
    protocol_text = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    protocol_hash = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()

    device = select_device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint["config"]
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    seed = int(protocol.get("seed", config["experiment"]["seed"]))
    seed_everything(seed)

    data = config["data"]
    base_dataset = PairedChangeDataset(
        dataset_root=data["dataset_root"],
        manifest_path=data["manifest"],
        split=args.split,
        image_size=data.get("eval_crop_size"),
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
    )
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    loss_config = config["loss"]
    criterion = BinaryChangeLoss(
        bce_weight=float(loss_config.get("bce_weight", 1.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    evaluation = config["evaluation"]
    batch_size = int(
        protocol.get(
            "batch_size",
            evaluation.get("batch_size", config["training"]["batch_size"]),
        )
    )
    num_workers = int(protocol.get("num_workers", data.get("num_workers", 4)))
    prefetch_factor = int(
        protocol.get("prefetch_factor", data.get("prefetch_factor", 2))
    )
    threshold = float(evaluation.get("threshold", 0.5))

    rows: list[dict[str, Any]] = []
    clean_f1: float | None = None
    for scenario_index, spec in enumerate(specs):
        print(
            f"[{scenario_index + 1}/{len(specs)}] "
            f"scenario={spec.key} value={spec.value}"
        )
        dataset = CorruptedPairDataset(
            base_dataset=base_dataset,
            spec=spec,
            normalization=str(data.get("normalization", "imagenet")),
        )
        if args.paired_shift_support and spec.name == "shift":
            dataset = PairedShiftSupportDataset(dataset)
        if args.oracle_inverse_shift and spec.name == "shift":
            if spec.mode not in {"t1", "t2"}:
                raise ValueError(
                    "Oracle inverse shift only supports t1 or t2 modes"
                )
            dataset = OracleInverseShiftDataset(
                shifted_dataset=dataset,
                target=spec.mode,
            )
        loader = _loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed + scenario_index,
            device=device,
            prefetch_factor=prefetch_factor,
        )
        metrics = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )
        valid_pixels = int(
            metrics["tp"]
            + metrics["fp"]
            + metrics["fn"]
            + metrics["tn"]
        )
        if spec.name == "clean":
            clean_f1 = float(metrics["f1"])
        if clean_f1 is None:
            raise RuntimeError("Clean scenario must be evaluated first")
        f1 = float(metrics["f1"])
        row = {
            "scenario": spec.key,
            "corruption": spec.name,
            "severity": spec.severity,
            "severity_value": spec.value,
            "mode": spec.mode,
            **metrics,
            "f1_drop": clean_f1 - f1,
            "relative_f1_drop": (clean_f1 - f1) / max(clean_f1, 1e-12),
            "valid_pixels": valid_pixels,
        }
        rows.append(row)
        print(
            f"f1={f1:.4f} iou={float(metrics['iou']):.4f} "
            f"drop={row['f1_drop']:.4f}"
        )

    corrupted_rows = [row for row in rows if row["corruption"] != "clean"]
    summary = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "split": args.split,
        "threshold": threshold,
        "oracle_inverse_shift": args.oracle_inverse_shift,
        "paired_shift_support": args.paired_shift_support,
        "scenario_count": len(rows),
        "clean_f1": clean_f1,
        "mean_corrupted_f1": sum(
            float(row["f1"]) for row in corrupted_rows
        )
        / max(1, len(corrupted_rows)),
        "worst_corrupted_f1": min(
            (float(row["f1"]) for row in corrupted_rows),
            default=clean_f1,
        ),
        "mean_f1_drop": sum(
            float(row["f1_drop"]) for row in corrupted_rows
        )
        / max(1, len(corrupted_rows)),
        "results": rows,
    }
    save_json(output_dir / "results.json", summary)
    _write_csv(output_dir / "results.csv", rows)
    with (output_dir / "protocol_resolved.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False, allow_unicode=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    print(f"Saved robustness results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
