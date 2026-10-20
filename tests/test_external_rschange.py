"""Pin the silent-corruption and late-failure risks in the rschange wrapper.

Every failure below either produces plausible F1 numbers rather than an
error, or only surfaces on the server hours into a run. They are checked
against a fake checkout written to disk, whose package ``__init__`` files
raise on import exactly as rschange's do when Mamba kernels are missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from rs_cd.models.dual_path_adapter import DualPathAdapter
from rs_cd.models.external_rschange import (
    INFERENCE_SEED,
    RSChangeDetector,
    binary_logit_from_two_class,
)

EXPLODING_INIT = "raise ImportError('mamba_ssm is not installed')\n"

BACKBONE_SOURCE = '''
from pathlib import Path

import torch
from torch import nn

# rschange's cdloma.py flips global cudnn flags at import time.
torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark


class FakeBackbone(nn.Module):
    def __init__(self, channels=4, weights=None):
        super().__init__()
        if weights is not None:
            # SeaFormer loads its weights by a checkout-relative path.
            self.loaded = Path(weights).read_text()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)

    def forward(self, xA, xB):
        return [self.conv(xA), self.conv(xB)]


class NoisyBackbone(FakeBackbone):
    """Fresh noise on every call, in eval too: CD-Lamba's Gumbel top-k."""

    def forward(self, xA, xB):
        noise = torch.rand_like(xA[:, :1])
        return [self.conv(xA) + noise, self.conv(xB)]
'''

HEAD_SOURCE = '''
import torch
from torch import nn
import torch.nn.functional as F


class FakeHead(nn.Module):
    def __init__(self, channels=4, out_channels=2, mode="plain"):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(channels, out_channels, 1)

    def forward(self, inputs):  # one argument, as rschange's myModel calls it
        features_a, features_b = inputs
        logits = self.conv(features_a - features_b)
        if self.mode == "tuple":
            return (logits, F.avg_pool2d(logits, 2))
        if self.mode == "lowres":
            return F.avg_pool2d(logits, 4)
        if self.mode == "mask_classification":
            return [torch.sigmoid(logits[:, :1]), {"pred_logits": logits}]
        return logits
'''

BACKBONE = {"type": "FakeBackbone", "module": "rscd.models.backbones.fake_backbone"}
NOISY_BACKBONE = {"type": "NoisyBackbone", "module": "rscd.models.backbones.fake_backbone"}


def _head(**parameters):
    return {
        "type": "FakeHead",
        "module": "rscd.models.decoderheads.fake_head",
        **parameters,
    }


def _pair(batch: int = 2, size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(batch, 3, size, size), torch.rand(batch, 3, size, size)


class _NoLocalizer(torch.nn.Module):
    """Satisfies DualPathAdapter; forward_with_forced_offset never calls it."""

    def estimate_registration(self, image_t1, image_t2):
        return {}


@pytest.fixture(autouse=True)
def _isolated_imports():
    saved_path = list(sys.path)
    saved_cwd = os.getcwd()
    yield
    for name in [n for n in sys.modules if n == "rscd" or n.startswith("rscd.")]:
        del sys.modules[name]
    sys.path[:] = saved_path
    os.chdir(saved_cwd)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    for package in (
        "rscd",
        "rscd/models",
        "rscd/models/backbones",
        "rscd/models/decoderheads",
    ):
        (tmp_path / package).mkdir(parents=True, exist_ok=True)
        (tmp_path / package / "__init__.py").write_text(EXPLODING_INIT)
    (tmp_path / "rscd/models/backbones/fake_backbone.py").write_text(
        BACKBONE_SOURCE
    )
    (tmp_path / "rscd/models/decoderheads/fake_head.py").write_text(HEAD_SOURCE)
    return tmp_path


def test_two_class_conversion_preserves_the_change_probability() -> None:
    """sigmoid(l1 - l0) must equal softmax([l0, l1])[1] exactly."""

    generator = torch.Generator().manual_seed(3407)
    logits = torch.randn(4, 2, 8, 8, generator=generator)
    converted = binary_logit_from_two_class(logits)

    softmax_change = torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)
    torch.testing.assert_close(torch.sigmoid(converted), softmax_change)

    # And the 0.5 decision boundary agrees with an argmax, which is what
    # rschange itself uses at inference.
    torch.testing.assert_close(
        (converted > 0).squeeze(1),
        logits.argmax(dim=1).bool(),
    )


def test_taking_channel_one_alone_would_be_wrong() -> None:
    """Guards the specific mistake the conversion exists to prevent."""

    logits = torch.tensor([[[[2.0]], [[1.0]]]])  # argmax says "no change"
    assert logits.argmax(dim=1).item() == 0
    naive = torch.sigmoid(logits[:, 1]).item()
    correct = torch.sigmoid(binary_logit_from_two_class(logits)).item()
    assert naive > 0.5, "the naive reading would call this a change"
    assert correct < 0.5, "the correct conversion agrees with argmax"


def test_package_init_files_are_never_executed(checkout: Path) -> None:
    """BIT must not require every Mamba dependency in the toolbox."""

    model = RSChangeDetector(checkout, BACKBONE, _head(), name="fake")
    assert model.backbone.__class__.__name__ == "FakeBackbone"


def test_head_receives_backbone_output_as_one_argument(checkout: Path) -> None:
    """Unpacking [f1, f2] into two arguments is the bug this pins."""

    model = RSChangeDetector(checkout, BACKBONE, _head(), name="fake")
    outputs = model(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32))
    assert outputs["change_logits"].shape == (2, 1, 32, 32)


def test_checkout_relative_weights_load_and_cwd_is_restored(
    checkout: Path,
) -> None:
    weights = checkout / "rscd/models/backbones/review_pretrain"
    weights.mkdir()
    (weights / "fake.pth").write_text("weights")
    before = os.getcwd()
    model = RSChangeDetector(
        checkout,
        {**BACKBONE, "weights": "rscd/models/backbones/review_pretrain/fake.pth"},
        _head(),
    )
    assert model.backbone.loaded == "weights"
    assert os.getcwd() == before


def test_global_cudnn_flags_survive_construction(checkout: Path) -> None:
    """The frozen localizer shares the process and must keep its numerics."""

    cudnn = torch.backends.cudnn
    before = (cudnn.enabled, cudnn.benchmark, cudnn.deterministic)
    RSChangeDetector(checkout, BACKBONE, _head())
    assert (cudnn.enabled, cudnn.benchmark, cudnn.deterministic) == before


def test_deep_supervision_tuple_takes_the_primary_output(checkout: Path) -> None:
    model = RSChangeDetector(checkout, BACKBONE, _head(mode="tuple"))
    outputs = model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    assert outputs["change_logits"].shape == (1, 1, 32, 32)


def test_low_resolution_head_is_upsampled_to_the_input(checkout: Path) -> None:
    model = RSChangeDetector(checkout, BACKBONE, _head(mode="lowres"))
    outputs = model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    assert outputs["change_logits"].shape == (1, 1, 32, 32)


def test_mask_classification_output_is_rejected_not_read_as_logits(
    checkout: Path,
) -> None:
    """CDMask's first output is a normalised probability, not a logit."""

    model = RSChangeDetector(
        checkout, BACKBONE, _head(mode="mask_classification"), name="CDMask"
    )
    with pytest.raises(NotImplementedError, match="mask-classification"):
        model(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))


def test_unexpected_channel_count_is_rejected(checkout: Path) -> None:
    model = RSChangeDetector(checkout, BACKBONE, _head(out_channels=5))
    with pytest.raises(ValueError, match="expected 1 or 2"):
        model(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))


def test_unknown_component_asks_for_a_module(checkout: Path) -> None:
    with pytest.raises(ImportError, match="module"):
        RSChangeDetector(checkout, {"type": "NotInRschange"}, _head())


def test_missing_checkout_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rschange checkout"):
        RSChangeDetector(tmp_path / "nowhere", BACKBONE, _head())


# Amendment B: stochastic inference (item 7 of the wrapper's docstring).


def test_the_preregistered_inference_seed_is_the_project_seed() -> None:
    assert INFERENCE_SEED == 3407


def test_evaluation_forward_of_a_stochastic_detector_is_repeatable(
    checkout: Path,
) -> None:
    """CD-Lamba samples Gumbel noise even at inference."""

    model = RSChangeDetector(checkout, NOISY_BACKBONE, _head()).eval()
    image_t1, image_t2 = _pair()
    first = model(image_t1, image_t2)["change_logits"]
    second = model(image_t1, image_t2)["change_logits"]
    assert torch.equal(first, second)


def test_zero_offset_raw_equals_aligned_for_a_stochastic_detector(
    checkout: Path,
) -> None:
    """The smoke guard the first CD-Lamba run failed: both paths share one draw."""

    detector = RSChangeDetector(checkout, NOISY_BACKBONE, _head())
    adapter = DualPathAdapter(_NoLocalizer(), detector).eval()
    image_t1, image_t2 = _pair(size=32)
    zero = torch.zeros(2, 2)

    shared = adapter.forward_with_forced_offset(image_t1, image_t2, zero)
    assert torch.equal(shared["raw_change_logits"], shared["aligned_change_logits"])

    # Without the private stream each path draws its own noise: the failure
    # the server smoke reported on 2026-09-15.
    detector.inference_seed = None
    independent = adapter.forward_with_forced_offset(image_t1, image_t2, zero)
    assert not torch.equal(
        independent["raw_change_logits"], independent["aligned_change_logits"]
    )


def test_training_forward_keeps_the_published_randomness(checkout: Path) -> None:
    model = RSChangeDetector(checkout, NOISY_BACKBONE, _head()).train()
    image_t1, image_t2 = _pair()
    first = model(image_t1, image_t2)["change_logits"]
    second = model(image_t1, image_t2)["change_logits"]
    assert not torch.equal(first, second)


def test_the_seed_fixes_the_draw_rather_than_removing_the_noise(
    checkout: Path,
) -> None:
    model = RSChangeDetector(checkout, NOISY_BACKBONE, _head()).eval()
    image_t1, image_t2 = _pair()
    first = model(image_t1, image_t2)["change_logits"]
    model.inference_seed = INFERENCE_SEED + 1
    second = model(image_t1, image_t2)["change_logits"]
    assert not torch.equal(first, second)


def test_evaluation_forward_leaves_the_global_random_stream_untouched(
    checkout: Path,
) -> None:
    model = RSChangeDetector(checkout, NOISY_BACKBONE, _head()).eval()
    image_t1, image_t2 = _pair()
    before = torch.get_rng_state()
    model(image_t1, image_t2)
    assert torch.equal(torch.get_rng_state(), before)


def test_deterministic_detector_is_numerically_unchanged(checkout: Path) -> None:
    """BIT and CDxLSTM draw no random numbers at inference, so the BIT row stands."""

    model = RSChangeDetector(checkout, BACKBONE, _head()).eval()
    image_t1, image_t2 = _pair()
    with_stream = model(image_t1, image_t2)["change_logits"]
    model.inference_seed = None
    without_stream = model(image_t1, image_t2)["change_logits"]
    assert torch.equal(with_stream, without_stream)


# Amendment C: native input size (item 8 of the wrapper's docstring).


def _manual_tiles(model: RSChangeDetector, image_t1, image_t2, size: int) -> torch.Tensor:
    """Stitch whole-input forwards of each window: the reference for tiling."""

    tile_size, model.inference_tile_size = model.inference_tile_size, None
    height, width = image_t1.shape[-2:]
    expected = torch.zeros(image_t1.shape[0], 1, height, width)
    for top in range(0, height, size):
        for left in range(0, width, size):
            window = (Ellipsis, slice(top, top + size), slice(left, left + size))
            expected[window] = model(image_t1[window], image_t2[window])["change_logits"]
    model.inference_tile_size = tile_size
    return expected


def test_detectors_run_on_whole_inputs_unless_they_declare_a_tile_size(
    checkout: Path,
) -> None:
    """BIT and CD-Lamba declare nothing, so their evaluation is unchanged."""

    assert RSChangeDetector(checkout, BACKBONE, _head()).inference_tile_size is None


def test_tiled_inference_stitches_per_tile_outputs(checkout: Path) -> None:
    model = RSChangeDetector(
        checkout, BACKBONE, _head(), inference_tile_size=16
    ).eval()
    image_t1, image_t2 = _pair(size=32)
    tiled = model(image_t1, image_t2)["change_logits"]
    assert tiled.shape == (2, 1, 32, 32)
    assert torch.equal(tiled, _manual_tiles(model, image_t1, image_t2, 16))
    # Tiling is not a no-op: the convolution sees tile borders.
    model.inference_tile_size = None
    assert not torch.equal(tiled, model(image_t1, image_t2)["change_logits"])


def test_tiled_inference_uses_smaller_windows_at_the_edges(checkout: Path) -> None:
    model = RSChangeDetector(
        checkout, BACKBONE, _head(), inference_tile_size=16
    ).eval()
    image_t1 = torch.rand(1, 3, 20, 28)
    image_t2 = torch.rand(1, 3, 20, 28)
    tiled = model(image_t1, image_t2)["change_logits"]
    assert tiled.shape == (1, 1, 20, 28)
    assert torch.equal(tiled, _manual_tiles(model, image_t1, image_t2, 16))


def test_training_forward_is_never_tiled(checkout: Path) -> None:
    model = RSChangeDetector(
        checkout, BACKBONE, _head(), inference_tile_size=16
    ).train()
    image_t1, image_t2 = _pair(size=32)
    tiled_setting = model(image_t1, image_t2)["change_logits"]
    model.inference_tile_size = None
    whole = model(image_t1, image_t2)["change_logits"]
    assert torch.equal(tiled_setting, whole)


def test_zero_offset_raw_equals_aligned_with_tiled_inference(checkout: Path) -> None:
    detector = RSChangeDetector(checkout, NOISY_BACKBONE, _head(), inference_tile_size=16)
    adapter = DualPathAdapter(_NoLocalizer(), detector).eval()
    image_t1, image_t2 = _pair(size=32)
    outputs = adapter.forward_with_forced_offset(image_t1, image_t2, torch.zeros(2, 2))
    assert torch.equal(outputs["raw_change_logits"], outputs["aligned_change_logits"])


def test_invalid_tile_size_is_rejected(checkout: Path) -> None:
    with pytest.raises(ValueError, match="inference_tile_size"):
        RSChangeDetector(checkout, BACKBONE, _head(), inference_tile_size=0)


def test_config_tile_size_reaches_the_wrapper(checkout: Path) -> None:
    from rs_cd.runtime import build_model

    model = build_model(
        {
            "model": {
                "name": "rschange_detector",
                "repo_root": str(checkout),
                "backbone": BACKBONE,
                "decoderhead": _head(),
                "inference_tile_size": 16,
            }
        }
    )
    assert model.inference_tile_size == 16
