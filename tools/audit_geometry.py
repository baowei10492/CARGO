from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
import yaml

from rs_cd.geometry import (
    interior_valid_mask,
    scale_input_offset,
    warp_reference_from_target,
)
from rs_cd.registration_controls import (
    ecc_translation_offset,
    phase_correlation_offset,
)
from rs_cd.robustness import _translate_reflect
from rs_cd.runtime import build_model, save_json, seed_everything, select_device


def _protocol_hash(protocol: dict[str, Any]) -> str:
    text = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit P7 translation signs, units, warps and strides."
    )
    parser.add_argument(
        "--protocol",
        default="configs/p7_geometry_audit.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-ecc",
        action="store_true",
        help="Fail instead of recording unavailable when OpenCV is absent.",
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Geometry protocol must be a mapping")
    seed = int(protocol["seed"])
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    height = int(protocol["tensor_height"])
    width = int(protocol["tensor_width"])
    tensor = torch.rand(1, 3, height, width, generator=generator)

    zero = torch.zeros(1, 2)
    identity = warp_reference_from_target(tensor, zero)
    identity_error = float((identity - tensor).abs().max())
    # A nonzero fractional offset bypasses the production function's
    # intentional exact-zero shortcut. Compare it with an independently
    # constructed affine grid so the audit exercises sampling geometry.
    fractional = torch.tensor([[0.375, -0.625]], dtype=tensor.dtype)
    production_fractional = warp_reference_from_target(tensor, fractional)
    theta = torch.tensor(
        [
            [
                [1.0, 0.0, 2.0 * 0.375 / width],
                [0.0, 1.0, 2.0 * -0.625 / height],
            ]
        ],
        dtype=tensor.dtype,
    )
    reference_grid = F.affine_grid(
        theta,
        tensor.shape,
        align_corners=False,
    )
    reference_fractional = F.grid_sample(
        tensor,
        reference_grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    fractional_reference_error = float(
        (production_fractional - reference_fractional).abs().max()
    )
    rows: list[dict[str, Any]] = []
    for dx_value, dy_value in protocol["translations"]:
        dx, dy = int(dx_value), int(dy_value)
        shifted, _ = _translate_reflect(tensor[0], dx=dx, dy=dy)
        offset = torch.tensor([[dx, dy]], dtype=torch.float32)
        recovered = warp_reference_from_target(shifted.unsqueeze(0), offset)
        border = max(abs(dx), abs(dy))
        valid = interior_valid_mask(1, height, width, border).bool()
        valid = valid.expand_as(recovered)
        recovery_error = float(
            (recovered[valid] - tensor[valid]).abs().max()
        )
        phase = phase_correlation_offset(
            tensor[0],
            shifted,
            normalization="unit",
            max_abs_shift=int(protocol["max_abs_shift"]),
        )
        ecc: dict[str, Any]
        try:
            ecc_result = ecc_translation_offset(
                tensor[0],
                shifted,
                normalization="unit",
                initial_offset=(phase.dx, phase.dy),
            )
            ecc = {
                "status": "completed",
                "dx": ecc_result.dx,
                "dy": ecc_result.dy,
                "score": ecc_result.score,
                "converged": ecc_result.converged,
            }
        except ImportError:
            if args.require_ecc:
                raise
            ecc = {"status": "unavailable"}
        rows.append(
            {
                "target_dx": dx,
                "target_dy": dy,
                "warp_recovery_max_abs": recovery_error,
                "phase_dx": phase.dx,
                "phase_dy": phase.dy,
                "phase_exact": phase.dx == dx and phase.dy == dy,
                "phase_score": phase.score,
                "ecc": ecc,
            }
        )

    stride_audit: dict[str, Any] = {"status": "not_requested"}
    checkpoint_path: Path | None = None
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        device = select_device(args.device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        config = checkpoint["config"]
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        image_size = int(protocol["feature_probe_image_size"])
        probe = torch.zeros(1, 3, image_size, image_size, device=device)
        features = model._encode(probe)
        levels = []
        for index, feature in enumerate(features):
            feature_size = tuple(int(value) for value in feature.shape[-2:])
            scaled = scale_input_offset(
                torch.tensor([[8.0, -4.0]], device=device),
                image_size=(image_size, image_size),
                feature_size=feature_size,
            )
            levels.append(
                {
                    "level": index,
                    "feature_height": feature_size[0],
                    "feature_width": feature_size[1],
                    "stride_x": image_size / feature_size[1],
                    "stride_y": image_size / feature_size[0],
                    "input_offset_8_-4_as_feature": [
                        float(scaled[0, 0]),
                        float(scaled[0, 1]),
                    ],
                }
            )
        stride_audit = {"status": "completed", "levels": levels}

    identity_tolerance = float(
        protocol["identity_max_abs_tolerance"]
    )
    recovery_tolerance = float(
        protocol["recovery_max_abs_tolerance"]
    )
    checks = {
        "zero_offset_passthrough": identity_error <= identity_tolerance,
        "fractional_warp_matches_independent_reference": (
            fractional_reference_error <= recovery_tolerance
        ),
        "known_translation_recovery": all(
            float(row["warp_recovery_max_abs"]) <= recovery_tolerance
            for row in rows
        ),
        "phase_exact": all(bool(row["phase_exact"]) for row in rows),
    }
    output = {
        "status": "passed" if all(checks.values()) else "failed",
        "protocol": str(protocol_path),
        "protocol_sha256": _protocol_hash(protocol),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "coordinate_convention": (
            "reference(x,y) corresponds to target(x+dx,y+dy)"
        ),
        "identity_max_abs_error": identity_error,
        "fractional_reference_max_abs_error": (
            fractional_reference_error
        ),
        "checks": checks,
        "translations": rows,
        "feature_stride_audit": stride_audit,
    }
    save_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
