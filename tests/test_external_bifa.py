"""Pin the silent-corruption risks in the BiFA wrapper.

BiFA's constructor loads its ImageNet weights with
``load_state_dict(..., False)``: a key the author renamed between versions
leaves that tensor at its random initialisation, prints nothing and trains to
a plausible but lower F1 -- the failure of main record 85. The checks below
run against a fake checkout whose package ``__init__`` files raise on import,
as the real model zoo's do when a detector's dependencies are missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from rs_cd.models.external_bifa import BiFADetector, _UnusedSegFormerHead

EXPLODING_INIT = "raise ImportError('this model zoo needs dependencies we do not have')\n"

# The shape of the author's file: mmcv/mmseg imports, a head that forward
# never calls, weights read from a path on the author's own machine, and the
# two keys popped from the checkpoint afterwards.
BIFA_SOURCE = '''
import torch
from torch import nn
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from models.seghead.segformer_head import SegFormerHead

PRETRAINED = r"E:\\pertrain_weight\\segformer\\mit_b0.pth"


class mit_b0(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)


class BiFA(nn.Module):
    def __init__(self, backbone="mit_b5"):
        super().__init__()
        if backbone == "mit_b0":
            self.segformer = mit_b0()
            self.ckpt = torch.load(PRETRAINED)
            self.segformer.load_state_dict(self.ckpt, False)
            self.head = SegFormerHead(in_channels=[32], num_classes=2)
        self.ckpt.pop("head.weight")
        self.ckpt.pop("head.bias")
        self.logger = get_root_logger()
        self.out = nn.Conv2d(4, 2, 1)

    def forward(self, x1, x2):
        return self.out(self.segformer.stem(x1) - self.segformer.stem(x2))
'''


@pytest.fixture(autouse=True)
def _isolated_imports():
    saved_path = list(sys.path)
    yield
    for name in [
        n
        for n in sys.modules
        if n in ("models", "mmcv", "mmseg") or n.startswith(("models.", "mmcv.", "mmseg."))
    ]:
        del sys.modules[name]
    sys.path[:] = saved_path


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    for package in ("models", "models/backbone", "models/bifa_help", "models/seghead"):
        (tmp_path / package).mkdir(parents=True, exist_ok=True)
        (tmp_path / package / "__init__.py").write_text(EXPLODING_INIT)
    (tmp_path / "models/bifa.py").write_text(BIFA_SOURCE)
    return tmp_path


def weights(tmp_path: Path, state: dict[str, torch.Tensor], name: str = "mit_b0.pth") -> Path:
    path = tmp_path / name
    torch.save(state, path)
    return path


def imagenet(generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    generator = generator or torch.Generator().manual_seed(3407)
    return {
        "stem.weight": torch.randn(4, 3, 3, 3, generator=generator),
        "stem.bias": torch.randn(4, generator=generator),
        "head.weight": torch.randn(1000, 256, generator=generator),
        "head.bias": torch.randn(1000, generator=generator),
    }


def build(checkout: Path, path: Path, **parameters) -> BiFADetector:
    return BiFADetector(checkout, pretrained_backbone=path, **parameters)


def test_the_imagenet_weights_reach_the_backbone(checkout: Path, tmp_path: Path) -> None:
    """The hard-coded path is answered with ours, and every tensor is checked."""

    state = imagenet()
    model = build(checkout, weights(tmp_path, state))

    assert model.pretrained["checked"] == 2   # both backbone tensors, not the head
    torch.testing.assert_close(model.model.segformer.stem.weight, state["stem.weight"])
    torch.testing.assert_close(model.model.segformer.stem.bias, state["stem.bias"])


def test_a_renamed_key_raises_instead_of_training_from_noise(
    checkout: Path, tmp_path: Path
) -> None:
    """load_state_dict(..., False) would accept this file and say nothing."""

    state = imagenet()
    state["backbone.stem.weight"] = state.pop("stem.weight")   # as a rename would look
    with pytest.raises(ValueError, match="did not take the ImageNet weights"):
        build(checkout, weights(tmp_path, state))


def test_weights_without_the_head_keys_fail_with_a_clear_message(
    checkout: Path, tmp_path: Path
) -> None:
    """BiFA.__init__ pops them; a bare KeyError hours in is not a diagnosis."""

    state = {key: value for key, value in imagenet().items() if not key.startswith("head.")}
    with pytest.raises(ValueError, match="head.weight"):
        build(checkout, weights(tmp_path, state))


def test_torch_load_is_restored_after_construction(checkout: Path, tmp_path: Path) -> None:
    """The interception must not outlive the constructor: the run loads checkpoints."""

    original = torch.load
    build(checkout, weights(tmp_path, imagenet()))
    assert torch.load is original

    other = tmp_path / "other.pth"
    torch.save({"value": torch.ones(2)}, other)
    torch.testing.assert_close(
        torch.load(other, weights_only=True)["value"], torch.ones(2)
    )


def test_package_init_files_are_never_executed(checkout: Path, tmp_path: Path) -> None:
    """models/__init__.py imports a zoo whose dependencies we do not install."""

    model = build(checkout, weights(tmp_path, imagenet()))
    assert model.model.__class__.__name__ == "BiFA"


def test_two_class_output_becomes_one_logit(checkout: Path, tmp_path: Path) -> None:
    model = build(checkout, weights(tmp_path, imagenet()), inference_tile_size=None).eval()
    pair = (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16))

    logits = model(*pair)["change_logits"]
    published = model.model(*pair)

    assert logits.shape == (2, 1, 16, 16)
    torch.testing.assert_close(
        torch.sigmoid(logits), torch.softmax(published, dim=1)[:, 1].unsqueeze(1)
    )


def test_evaluation_tiles_and_training_does_not(checkout: Path, tmp_path: Path) -> None:
    """Amendment C: eval runs on tiles of the training size, training on whole inputs."""

    model = build(checkout, weights(tmp_path, imagenet()), inference_tile_size=8).eval()
    t1, t2 = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)

    with torch.no_grad():
        tiled = model(t1, t2)["change_logits"]
        corner = model.model(t1[..., :8, :8], t2[..., :8, :8])
    assert tiled.shape == (1, 1, 16, 16)
    torch.testing.assert_close(
        tiled[..., :8, :8],
        (corner[:, 1] - corner[:, 0]).unsqueeze(1),
    )

    model.train()
    whole = model(t1, t2)["change_logits"]
    assert whole.shape == (1, 1, 16, 16)


def test_the_stub_head_refuses_to_be_used(checkout: Path, tmp_path: Path) -> None:
    """If a future BiFA starts using self.head, this must break loudly."""

    model = build(checkout, weights(tmp_path, imagenet()))
    assert isinstance(model.model.head, _UnusedSegFormerHead)
    assert not list(model.model.head.parameters())
    with pytest.raises(NotImplementedError):
        model.model.head(torch.rand(1, 32, 4, 4))


def test_a_backbone_branch_the_constructor_does_not_build_is_refused(
    checkout: Path, tmp_path: Path
) -> None:
    """mit_b5 is BiFA's default and leaves self.ckpt unset: fail before training."""

    with pytest.raises(ValueError, match="mit_b0"):
        build(checkout, weights(tmp_path, imagenet()), backbone="mit_b5")
