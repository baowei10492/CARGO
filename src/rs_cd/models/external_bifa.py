"""Use the published BiFA model as a downstream detector (P32).

BiFA (TGRS 2024, https://github.com/zmoka-zht/BiFA) is the one detector in
this paper that already aligns the two dates itself: every SegFormer stage
feeds each date the other one as a condition, and a learned flow warps one
side before the difference is taken. P32 adds it to check whether that
in-network alignment makes an explicit front-end redundant.

As with rschange (external_rschange.py) only the ``nn.Module`` is borrowed;
the optimiser, schedule, sampling, loss, tiling and evaluation border stay
this project's. Five things in the author's implementation would otherwise
fail late on the server, or -- worse -- run and produce plausible numbers:

1. *ImageNet initialisation by an absolute path.* ``BiFA.__init__`` runs
   ``torch.load(r"E:\\pertrain_weight\\segformer\\mit_b0.pth")``
   unconditionally, a path on the author's own machine, and the file itself
   is distributed only through a Baidu link. Rather than edit their source,
   ``torch.load`` is intercepted for the duration of the construction and
   returns the state dict named by ``pretrained_backbone``; the file is
   produced from the public ImageNet checkpoint by tools/p32_convert_mit_b0.py.
   Their code pops ``head.weight`` and ``head.bias`` from it afterwards, so
   both keys must be present.
2. *mmcv / mmseg imports.* ``bifa.py`` imports ``mmcv.runner.load_checkpoint``
   and ``mmseg.utils.get_root_logger`` at module level. Neither is used on the
   path we run (nothing calls ``init_weights``), and mmcv-full 1.6.2 has no
   wheel for this project's torch. Both symbols are shimmed, so mmcv is never
   installed.
3. *The unused segmentation head.* ``BiFA.__init__`` builds a ``SegFormerHead``
   that ``BiFA.forward`` never calls -- the prediction comes from the implicit
   function ``fpn_ifa``. That head is the only reason mmseg's decode-head stack
   is imported, and its parameters would otherwise take gradient and weight
   decay while contributing nothing. ``models.seghead.segformer_head`` is
   therefore replaced with a stub whose ``forward`` raises, which is visible
   at once if a future BiFA version starts using it.
4. *Silent partial loading.* Their line is
   ``self.segformer.load_state_dict(self.ckpt, False)``: a renamed or reshaped
   key leaves that parameter at its random initialisation and prints nothing,
   the same failure mode as main record 85. Every tensor handed in is compared
   with the constructed backbone afterwards and any mismatch raises.
5. *Memory at tile resolution.* The backbone runs at stride 4, so a 1024-pixel
   tile puts 65536 tokens through attention twice per stage. Evaluation
   therefore runs on non-overlapping tiles of ``inference_tile_size`` (256,
   the size BiFA trains on), as protocol amendment C does for CDxLSTM.

BiFA draws no random numbers in eval mode, so unlike CD-Lamba it needs no
private random stream (amendment B).

Usage requires a local checkout whose path is given in the config; nothing is
vendored into this repository.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn

from .external_rschange import binary_logit_from_two_class

# Packages of the checkout, registered as empty namespaces so that no
# ``__init__`` of the author's model zoo runs (models/__init__.py imports
# detectors this project does not have the dependencies for).
_NAMESPACE_PACKAGES = (
    "models",
    "models.backbone",
    "models.bifa_help",
    "models.seghead",
)

# The two keys BiFA.__init__ pops from the checkpoint after loading it.
HEAD_KEYS = ("head.weight", "head.bias")

# BiFA's own default is mit_b5, which its __init__ never builds: only the
# mit_b0 branch exists. The published experiments are mit_b0.
BACKBONE = "mit_b0"


def _checkout_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not (root / "models" / "bifa.py").is_file():
        raise FileNotFoundError(
            f"{root} does not look like a BiFA checkout: no models/bifa.py. "
            "Clone https://github.com/zmoka-zht/BiFA and point "
            "model.repo_root at it."
        )
    return root


def _register_namespaces(root: Path) -> None:
    """Make ``models.*`` of the checkout importable without running any init."""

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


def _stub(name: str, **members: Any) -> None:
    """Provide ``name`` with ``members`` unless the real module is present."""

    existing = sys.modules.get(name)
    if existing is not None:
        missing = [key for key in members if not hasattr(existing, key)]
        if missing:
            raise ImportError(
                f"{name} is already imported but defines none of {missing}; "
                "the BiFA shim would change what it means"
            )
        return
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        if parent not in sys.modules:
            _stub(parent)
        setattr(sys.modules[parent], child, module)


def _unused(symbol: str):
    def refuse(*args: Any, **kwargs: Any):
        raise NotImplementedError(
            f"{symbol} is shimmed for BiFA and is not expected to be called; "
            "this project loads no mmcv/mmseg checkpoint"
        )

    return refuse


class _UnusedSegFormerHead(nn.Module):
    """Stand-in for the head BiFA builds and never calls (item 3)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.arguments = kwargs

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "BiFA.forward does not use self.head in the published source; "
            "this build replaced it with a stub. Restore the real "
            "SegFormerHead before using a version that needs it."
        )


def _install_shims() -> None:
    """mmcv, mmseg and the unused head (items 2 and 3)."""

    _stub("mmcv.runner", load_checkpoint=_unused("mmcv.runner.load_checkpoint"))
    _stub("mmseg.utils", get_root_logger=lambda *a, **k: logging.getLogger("bifa"))
    _stub("models.seghead.segformer_head", SegFormerHead=_UnusedSegFormerHead)


def load_backbone_state(path: str | Path) -> dict[str, torch.Tensor]:
    """The ImageNet state dict BiFA's constructor expects (item 1)."""

    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(
            f"missing ImageNet weights for BiFA's backbone: {file}. "
            "Produce them with tools/p32_convert_mit_b0.py."
        )
    payload = torch.load(file, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
    if not isinstance(payload, dict):
        raise ValueError(f"{file} does not hold a state dict")
    missing = [key for key in HEAD_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"{file} lacks {missing}; BiFA.__init__ pops those keys after "
            "loading and would raise KeyError"
        )
    return {key: value for key, value in payload.items()}


@contextlib.contextmanager
def _intercepted_torch_load(state: dict[str, torch.Tensor]) -> Iterator[None]:
    """Answer the constructor's hard-coded ``torch.load`` with ``state``."""

    original = torch.load

    def load(*args: Any, **kwargs: Any):
        return dict(state)   # their code pops from it; keep ours intact

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def _verify_backbone_loaded(
    backbone: nn.Module, state: dict[str, torch.Tensor]
) -> dict[str, int]:
    """Item 4: prove every handed-in tensor reached the constructed backbone."""

    parameters = dict(backbone.state_dict())
    wrong: list[str] = []
    checked = 0
    for key, value in state.items():
        if key in HEAD_KEYS:
            continue
        current = parameters.get(key)
        if current is None:
            wrong.append(f"{key}: absent from the backbone")
        elif tuple(current.shape) != tuple(value.shape):
            wrong.append(
                f"{key}: shape {tuple(current.shape)} != {tuple(value.shape)}"
            )
        elif not torch.equal(current.cpu(), value.cpu()):
            wrong.append(f"{key}: value did not reach the backbone")
        else:
            checked += 1
    if wrong:
        raise ValueError(
            "BiFA's backbone did not take the ImageNet weights "
            f"({len(wrong)} of {len(state)} keys); load_state_dict(..., False) "
            "would have hidden this. First offenders: " + "; ".join(wrong[:5])
        )
    if checked == 0:
        raise ValueError("no ImageNet tensor was checked against the backbone")
    return {"checked": checked, "backbone_tensors": len(parameters)}


class BiFADetector(nn.Module):
    """BiFA wrapped to this project's detector interface."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        pretrained_backbone: str | Path,
        name: str = "BiFA",
        backbone: str = BACKBONE,
        inference_tile_size: int | None = 256,
    ) -> None:
        super().__init__()
        if inference_tile_size is not None and int(inference_tile_size) < 1:
            raise ValueError("inference_tile_size must be a positive integer")
        if backbone != BACKBONE:
            raise ValueError(
                f"BiFA.__init__ only builds the {BACKBONE} branch; "
                f"'{backbone}' would leave self.ckpt unset"
            )
        root = _checkout_root(repo_root)
        _register_namespaces(root)
        _install_shims()
        self.name = name
        self.inference_tile_size = (
            None if inference_tile_size is None else int(inference_tile_size)
        )
        state = load_backbone_state(pretrained_backbone)
        module = importlib.import_module("models.bifa")
        with _intercepted_torch_load(state):
            self.model = getattr(module, "BiFA")(backbone=backbone)
        self.pretrained = _verify_backbone_loaded(self.model.segformer, state)
        # The constructor keeps the checkpoint on the module; it is not part
        # of the state dict, only of resident memory.
        self.model.ckpt = None

    def _published_logits(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.model(image_t1, image_t2)
        if isinstance(outputs, (tuple, list)):
            if not outputs:
                raise ValueError(f"{self.name} returned no logits")
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                f"{self.name} returned {type(outputs).__name__}, expected a tensor"
            )
        size = image_t1.shape[-2:]
        if outputs.shape[-2:] != size:
            outputs = nn.functional.interpolate(
                outputs, size=size, mode="bilinear", align_corners=False
            )
        return outputs

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
                    stitched = piece.new_zeros((*piece.shape[:-2], height, width))
                stitched[..., rows, columns] = piece
        if stitched is None:
            raise ValueError(f"{self.name} received an empty input")
        return stitched

    def forward(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
        targets: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if self.training or self.inference_tile_size is None:
            logits = self._published_logits(image_t1, image_t2)
        else:
            # Item 5: tile at the size BiFA trains on.
            logits = self._tiled_logits(image_t1, image_t2)
        if logits.shape[1] == 2:
            logits = binary_logit_from_two_class(logits)
        elif logits.shape[1] != 1:
            raise ValueError(
                f"{self.name} produced {logits.shape[1]} channels; expected 1 or 2"
            )
        return {"change_logits": logits, "aux": None}
