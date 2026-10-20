from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from rs_cd.datasets import PairedChangeDataset
from rs_cd.models import TaskAwareInputRegistrationEstimator
from rs_cd.robustness import CorruptedPairDataset, CorruptionSpec
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.train import _loader


def _prediction(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    stride: torch.Tensor,
) -> torch.Tensor:
    return offsets.to(stride)[indices] * stride


def _correct(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - target, dim=1) <= 1.0


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose P7 zero-anchor collapse versus search-boundary support."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--expanded-radius", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config: dict[str, Any] = checkpoint["config"]
    if args.dataset_root is not None:
        config["data"]["dataset_root"] = args.dataset_root
    seed = int(config["experiment"]["seed"])
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
    corruption_name = (
        "clean" if int(args.severity) == 0 else "shift"
    )
    corruption_mode = (
        "common" if int(args.severity) == 0 else "t2"
    )
    dataset = CorruptedPairDataset(
        base_dataset,
        CorruptionSpec(
            name=corruption_name,
            severity=int(args.severity),
            mode=corruption_mode,
            seed=3407,
        ),
        normalization=str(data.get("normalization", "imagenet")),
    )
    loader = _loader(
        dataset=dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(data.get("num_workers", 4)),
        seed=seed + int(args.severity),
        device=device,
        prefetch_factor=int(data.get("prefetch_factor", 2)),
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    registration = getattr(model, "registration", None)
    if not isinstance(
        registration,
        TaskAwareInputRegistrationEstimator,
    ):
        raise TypeError("Checkpoint is not a P7 task-aware estimator")

    expanded = TaskAwareInputRegistrationEstimator(
        in_channels=3,
        descriptor_channels=int(
            config["model"].get("descriptor_channels", 32)
        ),
        search_radius=int(args.expanded_radius),
        top_fraction=float(config["model"].get("top_fraction", 0.25)),
        initial_local_logit_scale=float(
            config["model"].get("initial_local_logit_scale", 10.0)
        ),
        initial_global_logit_scale=float(
            config["model"].get("initial_global_logit_scale", 5.0)
        ),
        surface_decoder_channels=0,
        change_exclusion_radius=int(
            config["model"].get("change_exclusion_radius", 2)
        ),
    ).to(device)
    expanded.descriptor.load_state_dict(registration.descriptor.state_dict())
    expanded.reliability.load_state_dict(
        registration.reliability.state_dict()
    )
    expanded.log_local_logit_scale.copy_(
        registration.log_local_logit_scale
    )
    expanded.log_global_logit_scale.copy_(
        registration.log_global_logit_scale
    )
    expanded.eval()

    rows: list[dict[str, Any]] = []
    for batch in loader:
        t1 = batch["image_t1"].to(device, non_blocking=True)
        t2 = batch["image_t2"].to(device, non_blocking=True)
        reference = registration.descriptor(t1)
        target_descriptor = registration.descriptor(t2)
        current = registration.forward_descriptors(reference, target_descriptor)
        wider = expanded.forward_descriptors(reference, target_descriptor)
        stride = current["offset_feature"].new_tensor(
            [
                t1.shape[-1] / reference.shape[-1],
                t1.shape[-2] / reference.shape[-2],
            ]
        )
        target = torch.stack(
            (
                torch.as_tensor(batch["shift_dx"]),
                torch.as_tensor(batch["shift_dy"]),
            ),
            dim=1,
        ).to(stride)

        final_indices = current["offset_logits"].argmax(dim=1)
        raw_indices = current["score_surface"].argmax(dim=1)
        nonzero_logits = current["offset_logits"].clone()
        side = 2 * registration.search_radius + 1
        center = registration.search_radius * side + registration.search_radius
        nonzero_logits[:, center] = -torch.inf
        nonzero_indices = nonzero_logits.argmax(dim=1)
        wider_indices = wider["score_surface"].argmax(dim=1)

        predictions = {
            "final": _prediction(
                final_indices,
                registration.offsets,
                stride,
            ),
            "raw_current": _prediction(
                raw_indices,
                registration.offsets,
                stride,
            ),
            "best_nonzero": _prediction(
                nonzero_indices,
                registration.offsets,
                stride,
            ),
            "raw_expanded": _prediction(
                wider_indices,
                expanded.offsets,
                stride,
            ),
        }
        correctness = {
            name: _correct(prediction, target)
            for name, prediction in predictions.items()
        }
        for index, sample_id in enumerate(batch["id"]):
            row: dict[str, Any] = {
                "id": str(sample_id),
                "target_dx": float(target[index, 0]),
                "target_dy": float(target[index, 1]),
                "zero_probability": float(
                    current["zero_probability"][index]
                ),
                "confidence": float(current["confidence"][index]),
            }
            for name, prediction in predictions.items():
                row[f"{name}_dx"] = float(prediction[index, 0])
                row[f"{name}_dy"] = float(prediction[index, 1])
                row[f"{name}_correct"] = bool(
                    correctness[name][index]
                )
            row["conditional_boundary"] = bool(
                max(
                    abs(row["final_dx"]),
                    abs(row["final_dy"]),
                )
                >= 2.0 * registration.search_radius - 0.5
            )
            row["q_shift_decision"] = bool(
                row["zero_probability"] < 0.5
            )
            row["q_or_boundary_decision"] = bool(
                row["q_shift_decision"]
                or row["conditional_boundary"]
            )
            target_is_shifted = bool(
                abs(row["target_dx"]) > 0.5
                or abs(row["target_dy"]) > 0.5
            )
            row["q_applied_correct"] = bool(
                (
                    row["q_shift_decision"]
                    and row["final_correct"]
                )
                if target_is_shifted
                else not row["q_shift_decision"]
            )
            row["q_or_boundary_applied_correct"] = bool(
                (
                    row["q_or_boundary_decision"]
                    and row["final_correct"]
                )
                if target_is_shifted
                else not row["q_or_boundary_decision"]
            )
            rows.append(row)

    methods = ("final", "raw_current", "best_nonzero", "raw_expanded")
    summary = {
        method: {
            "pck1": sum(bool(row[f"{method}_correct"]) for row in rows)
            / max(1, len(rows)),
            "correct": sum(
                bool(row[f"{method}_correct"]) for row in rows
            ),
            "samples": len(rows),
        }
        for method in methods
    }
    failed_final = [row for row in rows if not row["final_correct"]]
    failed_rescue = {
        method: {
            "rescued": sum(
                bool(row[f"{method}_correct"]) for row in failed_final
            ),
            "failed_final_samples": len(failed_final),
            "rescue_rate": sum(
                bool(row[f"{method}_correct"]) for row in failed_final
            )
            / max(1, len(failed_final)),
        }
        for method in methods[1:]
    }
    by_direction: dict[str, dict[str, float | int]] = {}
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["target_dx"]), int(row["target_dy"]))
        ].append(row)
    for direction, subset in sorted(grouped.items()):
        key = f"{direction[0]:+d},{direction[1]:+d}"
        by_direction[key] = {
            "samples": len(subset),
            **{
                f"{method}_pck1": sum(
                    bool(row[f"{method}_correct"]) for row in subset
                )
                / len(subset)
                for method in methods
            },
        }

    output = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "severity": int(args.severity),
        "current_radius_feature": registration.search_radius,
        "expanded_radius_feature": int(args.expanded_radius),
        "summary": summary,
        "decision_support": {
            "q_shift_rate": sum(
                bool(row["q_shift_decision"]) for row in rows
            )
            / max(1, len(rows)),
            "conditional_boundary_rate": sum(
                bool(row["conditional_boundary"]) for row in rows
            )
            / max(1, len(rows)),
            "q_or_boundary_rate": sum(
                bool(row["q_or_boundary_decision"]) for row in rows
            )
            / max(1, len(rows)),
            "q_applied_pck1": sum(
                bool(row["q_applied_correct"]) for row in rows
            )
            / max(1, len(rows)),
            "q_or_boundary_applied_pck1": sum(
                bool(row["q_or_boundary_applied_correct"])
                for row in rows
            )
            / max(1, len(rows)),
        },
        "failed_final_rescue": failed_rescue,
        "by_direction": by_direction,
        "results": rows,
    }
    save_json(args.output, output)
    print("===== OVERALL PCK@1 =====")
    for method in methods:
        value = summary[method]
        print(
            f"{method:16s} "
            f"{value['correct']:2d}/{value['samples']:2d} "
            f"({value['pck1']:.3f})"
        )
    print("\n===== RESCUE OF FINAL FAILURES =====")
    for method in methods[1:]:
        value = failed_rescue[method]
        print(
            f"{method:16s} "
            f"{value['rescued']:2d}/{value['failed_final_samples']:2d} "
            f"({value['rescue_rate']:.3f})"
        )
    print("\n===== DECISION SUPPORT =====")
    print(
        json.dumps(
            output["decision_support"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("\n===== BY DIRECTION =====")
    print(json.dumps(by_direction, indent=2, ensure_ascii=False))
    print(f"Saved search-support diagnosis to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
