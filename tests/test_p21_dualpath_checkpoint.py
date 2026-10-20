"""The stage-2 composer decides which localizer settings every cell runs with.

A wrong setting here does not crash: it changes the estimated offsets for
every detector in the matrix at once. The tests pin that the frozen settings
are copied, that the overrides match the P12 router tool character for
character, and that the weights land in the half they belong to.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from rs_cd.models.dual_path_adapter import DualPathAdapter
from tools.build_p21_dualpath_checkpoint import (
    LOCALIZER_INFERENCE_OVERRIDES,
    compose_config,
    compose_state,
    localizer_config_from_payload,
)

ROUTER_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "train_evaluate_task_benefit_router.py"
)

P7C_MODEL = {
    "name": "dinov3_task_aware_input_registration_cd",
    "pretrained": True,
    "search_radius": 4,
    "decoupled_shift_decision": True,
    "use_change_mask_for_matching": False,
    "skip_change_detector_during_registration": True,
    "zero_probability_threshold": 0.5,
    "registration_only": True,
    "freeze_registration": False,
    "offset_decoding": "hard",
    "alignment_gate": "threshold_nonzero",
}


def _payload(model_config, state=None, **extra):
    return {
        "config": {
            "experiment": {"name": "run", "seed": 3407},
            "data": {"dataset_root": "/data/LEVIR-CD", "manifest": "m.csv"},
            "loss": {"bce_weight": 1.0, "dice_weight": 1.0},
            "evaluation": {"threshold": 0.5},
            "model": model_config,
        },
        "model": state or {},
        **extra,
    }


def test_frozen_localizer_settings_are_copied_not_rewritten() -> None:
    source = dict(P7C_MODEL)
    config = localizer_config_from_payload(_payload(source))

    # Settings whose constructor defaults differ from the frozen P7C run.
    assert config["use_change_mask_for_matching"] is False
    assert config["skip_change_detector_during_registration"] is True
    assert config["decoupled_shift_decision"] is True
    assert config["search_radius"] == 4
    for key, value in LOCALIZER_INFERENCE_OVERRIDES.items():
        assert config[key] == value
    assert config["pretrained"] is False
    assert source == P7C_MODEL, "the checkpoint payload must not be mutated"


def test_overrides_match_the_router_tool_source() -> None:
    """If the P12 router tool changes its overrides, the composer must too."""

    text = ROUTER_TOOL.read_text(encoding="utf-8")
    for key, value in LOCALIZER_INFERENCE_OVERRIDES.items():
        literal = repr(value) if isinstance(value, bool) else f'"{value}"'
        assert f'config["model"]["{key}"] = {literal}' in text, key


def test_detector_run_supplies_data_loss_and_cell() -> None:
    detector = _payload(
        {"name": "rschange_detector", "detector_name": "BIT"},
        val_metrics={"f1": 0.9},
    )
    detector["config"]["data"]["dataset_root"] = "/data/GZ-CD-tiles"
    config = compose_config(
        _payload(P7C_MODEL),
        detector,
        cell={"detector": "BIT", "dataset": "GZ-CD"},
    )
    assert config["model"]["name"] == "dual_path_adapter"
    assert config["model"]["freeze_localizer"] is True
    assert config["model"]["detector"]["detector_name"] == "BIT"
    assert config["model"]["localizer"]["search_radius"] == 4
    assert config["data"]["dataset_root"] == "/data/GZ-CD-tiles"
    assert config["loss"]["dice_weight"] == 1.0
    assert config["cell"] == {"detector": "BIT", "dataset": "GZ-CD"}


def test_nested_adapter_is_refused() -> None:
    with pytest.raises(ValueError, match="already a dual-path adapter"):
        compose_config(
            _payload(P7C_MODEL), _payload({"name": "dual_path_adapter"})
        )


def test_bare_state_dict_cannot_stand_in_for_the_localizer() -> None:
    with pytest.raises(ValueError, match="no embedded config"):
        localizer_config_from_payload({"backbone.weight": torch.zeros(1)})


class _Localizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 2)

    def estimate_registration(self, image_t1, image_t2):  # pragma: no cover
        return {}


class _Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2, 2)

    def forward(self, image_t1, image_t2):  # pragma: no cover
        return image_t1


def test_composed_state_loads_strictly_into_the_right_halves() -> None:
    localizer, detector = _Localizer(), _Detector()
    nn.init.constant_(localizer.head.weight, 1.0)
    nn.init.constant_(detector.head.weight, 2.0)
    adapter = DualPathAdapter(_Localizer(), _Detector())
    adapter.load_state_dict(
        compose_state(localizer.state_dict(), detector.state_dict()),
        strict=True,
    )
    assert torch.all(adapter.localizer.head.weight == 1.0)
    assert torch.all(adapter.detector.head.weight == 2.0)
