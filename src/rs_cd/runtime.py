from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .models import (
    DinoV3ConvNeXtChangeDetector,
    DinoV3GlobalRegistrationChangeDetector,
    DinoV3SelectiveCorrespondenceChangeDetector,
    DinoV3TaskAwareInputRegistrationChangeDetector,
    FCSiamDiff,
)
from rs_cd.models.dual_path_adapter import DualPathAdapter
from rs_cd.models.external_bifa import BiFADetector
from rs_cd.models.external_rschange import RSChangeDetector


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return config


def save_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")
    return device


def create_grad_scaler(
    device: torch.device, enabled: bool
) -> Any:
    """Create a GradScaler across supported PyTorch 2.x APIs."""
    modern_scaler = getattr(torch.amp, "GradScaler", None)
    if modern_scaler is not None:
        try:
            return modern_scaler(device.type, enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    name = str(model_config["name"]).lower()
    if name == "fc_siam_diff":
        return FCSiamDiff(
            in_channels=int(model_config.get("in_channels", 3)),
            base_channels=int(model_config.get("base_channels", 32)),
        )
    if name == "dinov3_convnext_cd":
        return DinoV3ConvNeXtChangeDetector(
            backbone_name=str(
                model_config.get(
                    "backbone_name",
                    "convnext_tiny.dinov3_lvd1689m",
                )
            ),
            pretrained=bool(model_config.get("pretrained", True)),
            checkpoint_path=model_config.get("checkpoint_path"),
            freeze_backbone=bool(
                model_config.get("freeze_backbone", True)
            ),
            trainable_stages=model_config.get("trainable_stages"),
            decoder_channels=int(
                model_config.get("decoder_channels", 128)
            ),
        )
    if name == "dinov3_selective_corr_cd":
        return DinoV3SelectiveCorrespondenceChangeDetector(
            backbone_name=str(
                model_config.get(
                    "backbone_name",
                    "convnext_tiny.dinov3_lvd1689m",
                )
            ),
            pretrained=bool(model_config.get("pretrained", True)),
            checkpoint_path=model_config.get("checkpoint_path"),
            freeze_backbone=bool(
                model_config.get("freeze_backbone", True)
            ),
            trainable_stages=model_config.get("trainable_stages"),
            decoder_channels=int(
                model_config.get("decoder_channels", 128)
            ),
            correspondence_channels=int(
                model_config.get("correspondence_channels", 32)
            ),
            correspondence_radius=int(
                model_config.get("correspondence_radius", 3)
            ),
            freeze_base_model=bool(
                model_config.get("freeze_base_model", True)
            ),
            change_margin_radius=int(
                model_config.get("change_margin_radius", 1)
            ),
        )
    if name == "dinov3_global_registration_cd":
        return DinoV3GlobalRegistrationChangeDetector(
            backbone_name=str(
                model_config.get(
                    "backbone_name",
                    "convnext_tiny.dinov3_lvd1689m",
                )
            ),
            pretrained=bool(model_config.get("pretrained", True)),
            checkpoint_path=model_config.get("checkpoint_path"),
            freeze_backbone=bool(
                model_config.get("freeze_backbone", True)
            ),
            trainable_stages=model_config.get("trainable_stages"),
            decoder_channels=int(
                model_config.get("decoder_channels", 128)
            ),
            search_radius=int(model_config.get("search_radius", 3)),
            initial_logit_scale=float(
                model_config.get("initial_logit_scale", 200.0)
            ),
            surface_decoder_channels=int(
                model_config.get("surface_decoder_channels", 0)
            ),
            normalize_score_surface=bool(
                model_config.get("normalize_score_surface", False)
            ),
            offset_decoding=str(
                model_config.get("offset_decoding", "soft")
            ),
            alignment_gate=str(
                model_config.get("alignment_gate", "none")
            ),
            zero_probability_threshold=float(
                model_config.get("zero_probability_threshold", 0.25)
            ),
            alignment_parameterization=str(
                model_config.get("alignment_parameterization", "direct")
            ),
            alignment_logit_bias=float(
                model_config.get("alignment_logit_bias", -3.0)
            ),
            registration_only=bool(
                model_config.get("registration_only", True)
            ),
            freeze_registration=bool(
                model_config.get("freeze_registration", False)
            ),
            use_confidence_gate=bool(
                model_config.get("use_confidence_gate", False)
            ),
            target_distribution_sigma=float(
                model_config.get("target_distribution_sigma", 0.5)
            ),
            offset_distribution_weight=float(
                model_config.get("offset_distribution_weight", 1.0)
            ),
            offset_regression_weight=float(
                model_config.get("offset_regression_weight", 1.0)
            ),
            fusion_regularization_weight=float(
                model_config.get("fusion_regularization_weight", 0.0)
            ),
        )
    if name == "dinov3_task_aware_input_registration_cd":
        return DinoV3TaskAwareInputRegistrationChangeDetector(
            backbone_name=str(
                model_config.get(
                    "backbone_name",
                    "convnext_tiny.dinov3_lvd1689m",
                )
            ),
            pretrained=bool(model_config.get("pretrained", True)),
            checkpoint_path=model_config.get("checkpoint_path"),
            decoder_channels=int(
                model_config.get("decoder_channels", 128)
            ),
            descriptor_channels=int(
                model_config.get("descriptor_channels", 32)
            ),
            search_radius=int(model_config.get("search_radius", 4)),
            top_fraction=float(
                model_config.get("top_fraction", 0.25)
            ),
            initial_local_logit_scale=float(
                model_config.get("initial_local_logit_scale", 10.0)
            ),
            initial_global_logit_scale=float(
                model_config.get("initial_global_logit_scale", 5.0)
            ),
            surface_decoder_channels=int(
                model_config.get("surface_decoder_channels", 16)
            ),
            change_exclusion_radius=int(
                model_config.get("change_exclusion_radius", 2)
            ),
            offset_decoding=str(
                model_config.get("offset_decoding", "hard")
            ),
            alignment_gate=str(
                model_config.get(
                    "alignment_gate",
                    "threshold_nonzero",
                )
            ),
            zero_probability_threshold=float(
                model_config.get("zero_probability_threshold", 0.5)
            ),
            registration_only=bool(
                model_config.get("registration_only", True)
            ),
            freeze_registration=bool(
                model_config.get("freeze_registration", False)
            ),
            target_distribution_sigma=float(
                model_config.get("target_distribution_sigma", 0.5)
            ),
            offset_distribution_weight=float(
                model_config.get("offset_distribution_weight", 1.0)
            ),
            offset_regression_weight=float(
                model_config.get("offset_regression_weight", 1.0)
            ),
            shiftedness_weight=float(
                model_config.get("shiftedness_weight", 0.25)
            ),
            clean_anchor_weight=float(
                model_config.get("clean_anchor_weight", 2.0)
            ),
            decoupled_shift_decision=bool(
                model_config.get("decoupled_shift_decision", False)
            ),
            freeze_localizer=bool(
                model_config.get("freeze_localizer", False)
            ),
            use_change_mask_for_matching=bool(
                model_config.get(
                    "use_change_mask_for_matching",
                    True,
                )
            ),
            skip_change_detector_during_registration=bool(
                model_config.get(
                    "skip_change_detector_during_registration",
                    False,
                )
            ),
        )
    if name == "rschange_detector":
        # P21: a published detector taken from an rschange checkout and run
        # under this project's protocol. Only the nn.Module is borrowed.
        return RSChangeDetector(
            repo_root=model_config["repo_root"],
            backbone=dict(model_config["backbone"]),
            decoderhead=dict(model_config["decoderhead"]),
            name=str(model_config.get("detector_name", "rschange")),
            inference_tile_size=model_config.get("inference_tile_size"),
        )
    if name == "bifa_detector":
        # P32: BiFA from its own checkout, run under this project's protocol
        # (src/rs_cd/models/external_bifa.py).
        return BiFADetector(
            repo_root=model_config["repo_root"],
            pretrained_backbone=model_config["pretrained_backbone"],
            name=str(model_config.get("detector_name", "BiFA")),
            backbone=str(model_config.get("backbone", "mit_b0")),
            inference_tile_size=model_config.get("inference_tile_size", 256),
        )
    if name == "dual_path_adapter":
        # P21: frozen localizer in front of an arbitrary detector. The two
        # sub-models are built by this same factory, so any detector added
        # below is immediately usable in the dual-path pipeline.
        localizer_config = dict(model_config["localizer"])
        detector_config = dict(model_config["detector"])
        localizer = build_model({"model": localizer_config})
        checkpoint = model_config.get("localizer_checkpoint")
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            state = state.get("model", state)
            missing, unexpected = localizer.load_state_dict(state, strict=False)
            # Loading a P7 checkpoint into the localizer legitimately leaves
            # the detector half unmatched, but a wholesale mismatch means the
            # wrong checkpoint: fail rather than screen with random weights
            # (main record 122.2 is exactly this failure).
            matched = len(state) - len(unexpected)
            if matched == 0:
                raise ValueError(
                    f"No parameter of {checkpoint} matched the localizer; "
                    "refusing to run with an unconfigured estimator"
                )
        return DualPathAdapter(
            localizer,
            build_model({"model": detector_config}),
            freeze_localizer=bool(
                model_config.get("freeze_localizer", True)
            ),
        )
    raise ValueError(f"Unsupported model '{name}'")


def create_run_directory(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{stamp}_seed{int(experiment['seed'])}"
    run_dir = Path(experiment.get("output_root", "outputs")) / str(
        experiment["name"]
    ) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def collect_system_info(device: torch.device) -> dict[str, Any]:
    git_commit = "not-a-git-repository"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    cuda_name = None
    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": cuda_name,
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit,
    }
