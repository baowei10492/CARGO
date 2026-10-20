"""Use rschange model definitions as detectors inside this project.

P21 needs several published change detectors evaluated under one protocol.
The rschange toolbox (https://github.com/xwmaxwma/rschange) already carries
faithful implementations of BIT, CD-Lamba, CDxLSTM and others, so we reuse
its ``nn.Module`` definitions rather than re-implementing each paper.

We deliberately do NOT adopt its training framework. rschange pins
torch 2.0.1 with pytorch-lightning, which conflicts with this project, and
using someone else's training loop would make the cross-detector comparison
incomparable with our frozen LEVIR-CD results. Only the model code is
imported; the optimiser, schedule, sampling, tiling, evaluation border and
paired-support metrics all stay ours. That is the point of the experiment:
every detector must face an identical protocol.

Checked against the rschange source (main, pushed 2026-04-20) before the
first server run. Each item below would otherwise fail late or, worse,
produce plausible numbers:

1. *Package imports.* ``rscd/models/backbones/__init__.py`` imports every
   model in the toolbox, including Mamba variants that need ``mamba_ssm``
   and compiled kernels. Importing BIT through the package would require the
   whole zoo to install. Components are therefore loaded from their own
   files, with the parent packages registered as empty namespaces so no
   package ``__init__`` runs.
2. *Head call convention.* rschange's ``myModel.forward`` passes the backbone
   output to the head as ONE argument (``decoderhead(backbone_outputs)``),
   usually a ``[features_t1, features_t2]`` list. Unpacking it into two
   arguments raises for BIT and silently mis-wires any head that accepts
   varargs.
3. *Working directory.* SeaFormer (the CD-Lamba and CDxLSTM backbone) loads
   ``rscd/models/backbones/review_pretrain/...`` by a path relative to the
   checkout, so components are constructed with the checkout as the cwd.
4. *Global cudnn flags.* ``cdloma.py`` sets ``cudnn.benchmark`` and
   ``cudnn.deterministic`` at import time, which would change the numerics
   of the frozen localizer running in the same process. The flags are
   restored after construction.
5. *Head output.* rschange heads emit two-class logits and take an argmax;
   this project uses a single-channel logit with a sigmoid. The conversion
   is ``logit = logits[:, 1] - logits[:, 0]``, which is the exact binary
   logit whose sigmoid equals the softmax probability of the change class.
   Anything else (taking channel 1 alone, or a softmax followed by a logit)
   changes the decision threshold and would shift every reported F1.
   Mask-classification heads (CDMask) return a per-image min-max normalised
   probability plus query outputs and are trained with a Hungarian-matched
   loss; they are rejected explicitly rather than read as logits.
6. *Deep supervision.* Heads returning several scales yield the first
   element, rschange's primary full-resolution output, upsampled if needed.
7. *Stochastic inference* (found by the first CD-Lamba smoke, 2026-09-15).
   CD-Lamba's ``cdloma.py`` picks its local scan windows with Gumbel top-k
   sampling on every forward pass, in eval mode too, so one image pair gives
   different logits on every call and the raw and aligned paths of the
   dual-path adapter draw different windows even at zero offset. Under
   protocol amendment B every evaluation-mode forward runs on a private
   random stream seeded with ``INFERENCE_SEED`` and the global stream is
   restored afterwards: each forward is one fixed draw of the published
   stochastic inference, shared by both paths. Training-mode forwards keep
   the published randomness. Detectors that draw no random numbers at
   inference (BIT, CDxLSTM) are numerically unchanged.
8. *Native input size* (found by the CDxLSTM instrument-guard diagnosis,
   2026-09-15). CDxLSTM has components tied to the absolute input size: a
   fixed 12x12 adaptive max-pool builds the cross-attention keys in LHBlock,
   and its xLSTM sequences grow with the input. Trained on 256-pixel crops,
   the same weights scored validation F1 0.4748 on whole 1024-pixel tiles
   and 0.8864 on 256-pixel tiles. Under protocol amendment C a detector may
   declare ``inference_tile_size``: every evaluation-mode forward then runs
   on non-overlapping tiles of that size (smaller windows at the right and
   bottom edges) and the logits are stitched back. Training-mode forwards are
   untouched, and a detector that declares nothing runs on whole inputs.

Usage requires a local checkout whose path is given in the config; nothing
is vendored into this repository.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

# Where each component type lives in the rschange checkout. File names do
# not match class names, and the package __init__ files must not be used
# (see item 1 of the module docstring). A config may override with "module".
KNOWN_MODULES: dict[str, str] = {
    "BIT_Backbone": "rscd.models.backbones.BIT",
    "cdlamba": "rscd.models.backbones.cdlamba",
    "Base": "rscd.models.backbones.backbone",
    "BASE_Transformer": "rscd.models.decoderheads.BIThead",
    "changedetector": "rscd.models.decoderheads.detector",
    "CDXLSTM": "rscd.models.decoderheads.xformer",
    "STNet": "rscd.models.decoderheads.stnet",
    "CDMask": "rscd.models.decoderheads.cdmask",
    # P32: SNUNet, the pure-convolution row (docs/P32_PLUGIN_EXTENSION.md).
    "UNetpp": "rscd.models.backbones.SNUNet_UNetpp",
    "ECAM": "rscd.models.decoderheads.SNUNet_ECAM",
}

_NAMESPACE_PACKAGES = (
    "rscd",
    "rscd.models",
    "rscd.models.backbones",
    "rscd.models.decoderheads",
)

# Protocol amendment B (docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md, section 10):
# the seed every evaluation-mode forward starts from. Written down before any
# CD-Lamba number was read; it is never tuned.
INFERENCE_SEED = 3407


def _checkout_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not (root / "rscd").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like an rschange checkout: no rscd/ found. "
            "Clone https://github.com/xwmaxwma/rschange and point "
            "model.repo_root at it."
        )
    return root


def _register_namespaces(root: Path) -> None:
    """Make ``rscd.models.*`` importable without running any package init."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for name in _NAMESPACE_PACKAGES:
        directory = root.joinpath(*name.split(".")).resolve()
        existing = sys.modules.get(name)
        if existing is not None:
            paths = {
                str(Path(path).resolve())
                for path in getattr(existing, "__path__", [])
            }
            if str(directory) not in paths:
                raise ImportError(
                    f"{name} is already imported from {sorted(paths)}; "
                    f"refusing to mix it with the checkout at {root}"
                )
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
    importlib.invalidate_caches()


@contextlib.contextmanager
def _inside_checkout(root: Path) -> Iterator[None]:
    """Construct with the checkout as cwd and leave global state untouched."""

    cudnn = torch.backends.cudnn
    saved_flags = (cudnn.enabled, cudnn.benchmark, cudnn.deterministic)
    saved_cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        cudnn.enabled, cudnn.benchmark, cudnn.deterministic = saved_flags


@contextlib.contextmanager
def _seeded_random_stream(seed: int) -> Iterator[None]:
    """Run on a private random stream and restore the global one afterwards.

    Only CUDA devices that are already initialised are seeded and restored:
    seeding an uninitialised CUDA runtime is deferred by torch and would fire
    later, outside this block, and change the global stream after all.
    """

    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.default_generator.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _build_component(spec: dict[str, Any]) -> nn.Module:
    """Instantiate ``spec['type']`` from its own module file."""

    parameters = dict(spec)
    type_name = str(parameters.pop("type"))
    module_path = parameters.pop("module", None) or KNOWN_MODULES.get(type_name)
    if module_path is None:
        raise ImportError(
            f"No module known for rschange component '{type_name}'. Add a "
            "'module' key naming the file, e.g. "
            "module: rscd.models.backbones.BIT"
        )
    module = importlib.import_module(str(module_path))
    component = getattr(module, type_name, None)
    if component is None:
        raise ImportError(f"{module_path} defines no '{type_name}'")
    return component(**parameters)


def binary_logit_from_two_class(logits: torch.Tensor) -> torch.Tensor:
    """Convert two-class logits to the equivalent single-channel logit.

    ``sigmoid(l1 - l0) == softmax([l0, l1])[1]`` exactly, so the decision at
    probability 0.5 is preserved and the project's threshold semantics carry
    over unchanged.
    """

    if logits.shape[1] != 2:
        raise ValueError(
            f"expected two-class logits, got {logits.shape[1]} channels"
        )
    return (logits[:, 1] - logits[:, 0]).unsqueeze(1)


class RSChangeDetector(nn.Module):
    """A published detector wrapped to this project's detector interface."""

    def __init__(
        self,
        repo_root: str | Path,
        backbone: dict[str, Any],
        decoderhead: dict[str, Any],
        *,
        name: str = "rschange",
        inference_seed: int | None = INFERENCE_SEED,
        inference_tile_size: int | None = None,
    ) -> None:
        super().__init__()
        if inference_tile_size is not None and int(inference_tile_size) < 1:
            raise ValueError("inference_tile_size must be a positive integer")
        root = _checkout_root(repo_root)
        _register_namespaces(root)
        self.name = name
        # None disables the private stream; only the tests use that.
        self.inference_seed = inference_seed
        # None runs evaluation on whole inputs (item 8, amendment C).
        self.inference_tile_size = (
            None if inference_tile_size is None else int(inference_tile_size)
        )
        with _inside_checkout(root):
            self.backbone = _build_component(backbone)
            self.decoderhead = _build_component(decoderhead)

    def _primary(self, outputs: Any, size: torch.Size) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise ValueError(f"{self.name} returned no logits")
            if len(outputs) > 1 and isinstance(outputs[1], dict):
                raise NotImplementedError(
                    f"{self.name} is a mask-classification head: it returns a "
                    "normalised probability map plus query outputs and needs "
                    "a Hungarian-matched loss. Its first output is not a "
                    "logit, and P21 has no amendment covering it."
                )
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                f"{self.name} returned {type(outputs).__name__}, expected a tensor"
            )
        if outputs.shape[-2:] != size:
            outputs = nn.functional.interpolate(
                outputs, size=size, mode="bilinear", align_corners=False
            )
        return outputs

    def _published_logits(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> torch.Tensor:
        features = self.backbone(image_t1, image_t2)
        # One argument, exactly as rschange's myModel.forward calls the head.
        return self._primary(self.decoderhead(features), image_t1.shape[-2:])

    def _tiled_logits(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> torch.Tensor:
        size = int(self.inference_tile_size)
        height, width = image_t1.shape[-2:]
        stitched: torch.Tensor | None = None
        for top in range(0, height, size):
            for left in range(0, width, size):
                rows = slice(top, top + size)
                columns = slice(left, left + size)
                piece = self._published_logits(
                    image_t1[..., rows, columns], image_t2[..., rows, columns]
                )
                if stitched is None:
                    stitched = piece.new_zeros(
                        (*piece.shape[:-2], height, width)
                    )
                stitched[..., rows, columns] = piece
        if stitched is None:
            raise ValueError(f"{self.name} received an empty input")
        return stitched

    def _logits(self, image_t1: torch.Tensor, image_t2: torch.Tensor) -> torch.Tensor:
        if self.training or self.inference_tile_size is None:
            return self._published_logits(image_t1, image_t2)
        # Item 8 of the module docstring (amendment C).
        return self._tiled_logits(image_t1, image_t2)

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if self.training or self.inference_seed is None:
            logits = self._logits(image_t1, image_t2)
        else:
            # Item 7 of the module docstring (amendment B).
            with _seeded_random_stream(self.inference_seed):
                logits = self._logits(image_t1, image_t2)
        if logits.shape[1] == 2:
            logits = binary_logit_from_two_class(logits)
        elif logits.shape[1] != 1:
            raise ValueError(
                f"{self.name} produced {logits.shape[1]} channels; expected 1 or 2"
            )
        return {"change_logits": logits, "aux": None}
