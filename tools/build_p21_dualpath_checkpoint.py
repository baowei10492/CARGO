"""Compose the stage-2 checkpoint of one P21 cell.

The router and classifier tools load a single checkpoint, rebuild the model
from its embedded config and load the weights strictly. For a P21 cell that
model is a DualPathAdapter, assembled here from two frozen parts:

    localizer : the localizer checkpoint fixed for this dataset
    detector  : the stage-1 checkpoint of the external detector

Two decisions are made in code rather than in YAML, because a hand-written
copy is exactly how silent drift enters:

* The localizer config is copied verbatim from its own checkpoint. The P7
  constructor defaults differ from the frozen P7C settings (for example
  ``use_change_mask_for_matching`` and ``decoupled_shift_decision``), so a
  YAML block that merely looks right can change the estimated offsets. The
  only edits are the inference overrides the P12 router tool itself applies,
  plus ``pretrained: false`` because the weights come from the checkpoint.
* Data, loss and evaluation settings come from the detector's stage-1 run,
  so routing uses the dataset the detector was trained on.

Both halves load strictly; any missing or unexpected key aborts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rs_cd.runtime import build_model

# Exactly what tools/train_evaluate_task_benefit_router.py main() sets on the
# P7 checkpoint config before routing. tests/test_p21_dualpath_checkpoint.py
# reads that source file and fails if the two drift apart.
LOCALIZER_INFERENCE_OVERRIDES: dict[str, Any] = {
    "registration_only": False,
    "freeze_registration": True,
    "offset_decoding": "hard",
    "alignment_gate": "threshold_nonzero",
}


def localizer_config_from_payload(payload: Any) -> dict[str, Any]:
    """The localizer's own training config with router-time overrides."""

    try:
        model_config = payload["config"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "localizer checkpoint carries no embedded config, so its frozen "
            "settings cannot be reproduced; pass a tools.train checkpoint"
        ) from error
    config = copy.deepcopy(dict(model_config))
    config.update(LOCALIZER_INFERENCE_OVERRIDES)
    config["pretrained"] = False
    return config


def compose_config(
    localizer_payload: Any,
    detector_payload: Any,
    *,
    cell: dict[str, str] | None = None,
) -> dict[str, Any]:
    run_config = detector_payload.get("config") if isinstance(
        detector_payload, dict
    ) else None
    if not isinstance(run_config, dict) or "model" not in run_config:
        raise ValueError("detector checkpoint carries no embedded config")
    detector_config = copy.deepcopy(dict(run_config["model"]))
    if detector_config.get("name") == "dual_path_adapter":
        raise ValueError("detector checkpoint is already a dual-path adapter")

    config = copy.deepcopy(run_config)
    config["model"] = {
        "name": "dual_path_adapter",
        "freeze_localizer": True,
        "localizer": localizer_config_from_payload(localizer_payload),
        "detector": detector_config,
    }
    if cell is not None:
        config["cell"] = dict(cell)
    return config


def compose_state(
    localizer_state: dict[str, torch.Tensor],
    detector_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Prefix each half with the attribute name DualPathAdapter uses."""

    state = {f"localizer.{key}": value for key, value in localizer_state.items()}
    state.update(
        {f"detector.{key}": value for key, value in detector_state.items()}
    )
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localizer-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    localizer_path = Path(args.localizer_checkpoint).expanduser().resolve()
    detector_path = Path(args.detector_checkpoint).expanduser().resolve()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    localizer_payload = torch.load(
        localizer_path, map_location="cpu", weights_only=False
    )
    detector_payload = torch.load(
        detector_path, map_location="cpu", weights_only=False
    )
    config = compose_config(
        localizer_payload,
        detector_payload,
        cell={"detector": args.detector, "dataset": args.dataset},
    )
    model = build_model(config)
    model.load_state_dict(
        compose_state(localizer_payload["model"], detector_payload["model"]),
        strict=True,
    )
    if any(parameter.requires_grad for parameter in model.localizer.parameters()):
        raise SystemExit("localizer is not frozen inside the adapter")

    provenance = {
        "cell": config["cell"],
        "localizer_checkpoint": str(localizer_path),
        "localizer_sha256": _sha256(localizer_path),
        "localizer_epoch": localizer_payload.get("epoch"),
        "detector_checkpoint": str(detector_path),
        "detector_sha256": _sha256(detector_path),
        "detector_epoch": detector_payload.get("epoch"),
        "detector_val_metrics": detector_payload.get("val_metrics"),
        "localizer_overrides": LOCALIZER_INFERENCE_OVERRIDES,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config, "provenance": provenance},
        output,
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
