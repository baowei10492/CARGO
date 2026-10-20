"""Frozen detectors for the P27 steps (docs/P27_PLUGIN_AMENDMENT.md).

A detector is named by a tools.train checkpoint (``{checkpoint: path}``, the
DINOv3 rows), by a pointer file written after training
(``{checkpoint_pointer: path}``, the WHU rows of the P21 detectors), or by a
frozen P21 cell (``{cell: bit_levir}``). A cell is looked up in the P21 validation freeze
manifest; its stage-1 checkpoint is used when present, otherwise the detector
half of the cell's dual-path checkpoint. Either way the identity is the
sha256 of the stage-1 checkpoint, so both sources compare equal.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rs_cd.runtime import build_model

DUAL_PATH_PREFIX = "detector."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _from_training_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["config"], payload["model"], sha256(path)


def _from_dual_path(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    if config["model"].get("name") != "dual_path_adapter":
        raise SystemExit(f"{path} is not a dual-path checkpoint")
    config["model"] = config["model"]["detector"]
    state = {key[len(DUAL_PATH_PREFIX):]: value for key, value in payload["model"].items()
             if key.startswith(DUAL_PATH_PREFIX)}
    return config, state, str(payload["provenance"]["detector_sha256"])


def cell_report(cell: str, manifest_path: Path, project_root: Path = Path(".")) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or cell not in manifest["cells"]:
        raise SystemExit(f"{cell} is not a frozen cell of {manifest_path}")
    return project_root / manifest["cells"][cell]["report_dir"]


def resolve_detector(spec: dict[str, Any], *, manifest_path: Path | None = None,
                     project_root: Path = Path(".")) -> dict[str, Any]:
    """Where the weights come from, without building the model."""

    if "checkpoint" in spec:
        path = Path(spec["checkpoint"])
        if not path.is_file():
            raise SystemExit(f"missing detector {path}")
        return {"kind": "checkpoint", "path": path}
    if "checkpoint_pointer" in spec:
        # A detector trained for the P27 rows: the pointer file names its best.pth.
        pointer = project_root / spec["checkpoint_pointer"]
        if not pointer.is_file():
            raise SystemExit(f"{pointer} does not exist; train the detector first")
        path = project_root / pointer.read_text(encoding="utf-8").strip()
        if not path.is_file():
            raise SystemExit(f"{pointer} names {path}, which does not exist")
        return {"kind": "checkpoint", "path": path, "pointer": str(pointer)}
    if manifest_path is None:
        raise SystemExit(f"detector cell {spec['cell']} needs the P21 freeze manifest")
    report = cell_report(str(spec["cell"]), manifest_path, project_root)
    pointer = report / "DETECTOR_CHECKPOINT"
    if pointer.is_file():
        stage1 = project_root / pointer.read_text(encoding="utf-8").strip()
        if stage1.is_file():
            return {"kind": "checkpoint", "path": stage1, "cell": spec["cell"], "report": str(report)}
    dual_path = report / "dual_path.pth"
    if dual_path.is_file():
        return {"kind": "dual_path", "path": dual_path, "cell": spec["cell"], "report": str(report)}
    raise SystemExit(f"{spec['cell']}: neither the stage-1 checkpoint named in {pointer} "
                     f"nor {dual_path} exists on this machine")


def open_detector(spec: dict[str, Any], device: torch.device, *, manifest_path: Path | None = None,
                  project_root: Path = Path(".")) -> tuple[torch.nn.Module, dict[str, Any], str, dict[str, Any]]:
    """(model in eval mode, its run config, stage-1 sha256, source description)."""

    source = resolve_detector(spec, manifest_path=manifest_path, project_root=project_root)
    loader = _from_training_checkpoint if source["kind"] == "checkpoint" else _from_dual_path
    config, state, digest = loader(source["path"])
    if spec.get("sha256") and digest != spec["sha256"]:
        raise SystemExit(f"detector {source['path']} has sha256 {digest}, expected {spec['sha256']}")
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    description = {key: str(value) for key, value in source.items()}
    return model.to(device).eval(), config, digest, description
