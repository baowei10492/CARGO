from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tools.evaluate_safe_alignment_transfer import merge_detector_state
from tools.train_evaluate_safe_alignment_classifier import (
    _sha256_of_file,
)


def filter_state_by_prefixes(
    state: dict[str, torch.Tensor],
    drop_prefixes: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop keys whose shapes depend on a changed architecture setting.

    Used when warm-starting a model whose search radius differs from the
    source checkpoint: radius-dependent heads (zero_anchor_head,
    shiftedness_head) must be re-initialized, everything else transfers.
    """

    if not drop_prefixes:
        return dict(state), []
    kept: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, value in state.items():
        if key.startswith(drop_prefixes):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the C2b stage-A initialization checkpoint: "
            "S2Looking P3b detector weights (backbone/decoder) merged "
            "with the LEVIR P7C registration stack as a warm start. "
            "The registration stack is retrained on S2Looking; the "
            "detector stays frozen during registration-only training."
        )
    )
    parser.add_argument("--p7-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument(
        "--drop-prefixes",
        default="",
        help=(
            "Comma-separated key prefixes to drop from the merged "
            "state (for radius-dependent heads when the target "
            "architecture differs from the source checkpoint)."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    p7_path = Path(args.p7_checkpoint).expanduser().resolve()
    detector_path = Path(
        args.detector_checkpoint
    ).expanduser().resolve()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    p7_checkpoint = torch.load(
        p7_path,
        map_location="cpu",
        weights_only=False,
    )
    detector_checkpoint = torch.load(
        detector_path,
        map_location="cpu",
        weights_only=False,
    )
    merged, replaced = merge_detector_state(
        p7_checkpoint["model"],
        detector_checkpoint["model"],
    )
    drop_prefixes = tuple(
        prefix
        for prefix in args.drop_prefixes.split(",")
        if prefix.strip()
    )
    merged, dropped = filter_state_by_prefixes(merged, drop_prefixes)
    if dropped:
        print(f"dropped_keys={len(dropped)}")
        for key in dropped:
            print(f"  dropped: {key}")
    torch.save(
        {
            "model": merged,
            "sources": {
                "p7_checkpoint": str(p7_path),
                "p7_checkpoint_sha256": _sha256_of_file(p7_path),
                "detector_checkpoint": str(detector_path),
                "detector_checkpoint_sha256": _sha256_of_file(
                    detector_path
                ),
                "detector_parameters_replaced": replaced,
            },
        },
        output,
    )
    print(f"detector_parameters_replaced={replaced}")
    print(f"Saved C2b init checkpoint to {output.resolve()}")


if __name__ == "__main__":
    main()
