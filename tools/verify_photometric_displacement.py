from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.geometry import warp_reference_from_target
from rs_cd.registration_controls import to_grayscale
from rs_cd.runtime import save_json, seed_everything


def zero_mean_ncc(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    border: int,
) -> float:
    """Zero-mean normalized cross-correlation over the interior."""

    if border > 0:
        first = first[border:-border, border:-border]
        second = second[border:-border, border:-border]
    a = first - first.mean()
    b = second - second.mean()
    denominator = a.norm() * b.norm()
    if float(denominator) < 1e-12:
        return 0.0
    return float((a * b).sum() / denominator)


def verify(
    similarities: list[dict[str, float]],
    *,
    improvement_fraction_min: float,
    control_margin_min: float,
) -> dict[str, Any]:
    """Measure whether an estimated warp improves its own objective.

    This is a useful consistency diagnostic, but not a causal test for a
    physical displacement: the same photometric objective selects the
    offset and evaluates it. A gain over random directions can therefore
    still reflect optimizer selection bias or temporal appearance change.
    """

    if not similarities:
        raise ValueError("No similarity records")
    improved = [
        row for row in similarities if row["estimated"] > row["identity"]
    ]
    fraction = len(improved) / len(similarities)
    mean_gain = sum(
        row["estimated"] - row["identity"] for row in similarities
    ) / len(similarities)
    mean_control_gain = sum(
        row["control"] - row["identity"] for row in similarities
    ) / len(similarities)
    margin = mean_gain - mean_control_gain
    bands_passed = bool(
        fraction >= float(improvement_fraction_min)
        and margin >= float(control_margin_min)
    )
    rows_with_magnitude = [
        row for row in similarities if "magnitude" in row
    ]
    zero_offset_samples = sum(
        abs(float(row["magnitude"])) <= 1e-12
        for row in rows_with_magnitude
    )
    return {
        "status": "completed",
        "samples": len(similarities),
        "fraction_improved_over_identity": fraction,
        "mean_ncc_identity": sum(
            row["identity"] for row in similarities
        )
        / len(similarities),
        "mean_ncc_estimated": sum(
            row["estimated"] for row in similarities
        )
        / len(similarities),
        "mean_ncc_random_control": sum(
            row["control"] for row in similarities
        )
        / len(similarities),
        "mean_gain_over_identity": mean_gain,
        "mean_control_gain": mean_control_gain,
        "gain_margin_over_control": margin,
        "bands": {
            "improvement_fraction_min": float(improvement_fraction_min),
            "control_margin_min": float(control_margin_min),
        },
        "self_objective_bands_passed": bands_passed,
        "causal_real_displacement_established": False,
        "zero_offset_samples": zero_offset_samples,
        "magnitude_metadata_samples": len(rows_with_magnitude),
        "verdict": (
            "photometric_objective_gain_observed_causal_geometry_unresolved"
            if bands_passed
            else "photometric_objective_gain_not_established"
        ),
        "limitation": (
            "Offset selection and verification use related photometric "
            "evidence; this result cannot prove physical displacement."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether a clean-pair estimate improves photometric "
            "similarity relative to identity and random-direction "
            "controls. This is a consistency diagnostic, not proof of "
            "physical displacement."
        )
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", default="phase")
    parser.add_argument("--border", type=int, default=24)
    parser.add_argument(
        "--improvement-fraction-min",
        type=float,
        default=0.70,
    )
    parser.add_argument("--control-margin-min", type=float, default=0.002)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(
        Path(args.protocol).expanduser().read_text(encoding="utf-8")
    )
    seed_everything(int(protocol["seed"]))
    with Path(args.records).expanduser().open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    estimates = {
        str(row["id"]): (float(row["est_dx"]), float(row["est_dy"]))
        for row in rows
        if str(row["method"]) == args.method
        and str(row["scenario"]) == "clean:s0:common"
    }
    if not estimates:
        raise ValueError(f"No clean rows for method {args.method}")

    dataset = PairedChangeDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split=str(protocol["validation_split"]),
        image_size=None,
        normalization="imagenet",
        augment=False,
        cache_mode="none",
    )
    generator = torch.Generator().manual_seed(int(protocol["seed"]))
    similarities: list[dict[str, float]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = str(sample["id"])
        if sample_id not in estimates:
            continue
        dx, dy = estimates[sample_id]
        t1 = sample["image_t1"]
        t2 = sample["image_t2"]
        magnitude = (dx * dx + dy * dy) ** 0.5
        angle = float(
            torch.rand(1, generator=generator) * 2.0 * torch.pi
        )
        control = (
            magnitude * float(torch.cos(torch.tensor(angle))),
            magnitude * float(torch.sin(torch.tensor(angle))),
        )

        def warped(offset: tuple[float, float]) -> torch.Tensor:
            return warp_reference_from_target(
                t2.unsqueeze(0),
                torch.tensor([[offset[0], offset[1]]], dtype=t2.dtype),
            )[0]

        gray_t1 = to_grayscale(t1, "imagenet")
        similarities.append(
            {
                "id": sample_id,
                "magnitude": magnitude,
                "identity": zero_mean_ncc(
                    gray_t1,
                    to_grayscale(t2, "imagenet"),
                    border=int(args.border),
                ),
                "estimated": zero_mean_ncc(
                    gray_t1,
                    to_grayscale(warped((dx, dy)), "imagenet"),
                    border=int(args.border),
                ),
                "control": zero_mean_ncc(
                    gray_t1,
                    to_grayscale(warped(control), "imagenet"),
                    border=int(args.border),
                ),
            }
        )

    decision = verify(
        similarities,
        improvement_fraction_min=float(args.improvement_fraction_min),
        control_margin_min=float(args.control_margin_min),
    )
    decision["method"] = str(args.method)
    decision["per_sample"] = similarities
    save_json(args.output, decision)
    printable = {
        key: value
        for key, value in decision.items()
        if key != "per_sample"
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
