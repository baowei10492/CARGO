"""ImageNet weights for BiFA's backbone, from a public file instead of a Baidu link.

BiFA's constructor loads ``mit_b0.pth`` in the original SegFormer naming
(``patch_embed1.proj.weight``, ``block1.0.attn.kv.weight``, ...). The authors
distribute that file only through a Baidu share, which has no checksum and
cannot be scripted. The same weights are public as the Hugging Face model
``nvidia/mit-b0`` (ImageNet-1k classification, the checkpoint SegFormer itself
was initialised from), under different key names, so this tool renames them:

    python -m tools.p32_convert_mit_b0 --download \
        --repo-root /root/autodl-tmp/rs_cd_work/BiFA \
        --output external/mit_b0_imagenet.pth

Two renames are not one-to-one and are the reason this is a script rather than
a regular expression:

* the original attention fuses key and value into one ``kv`` projection whose
  output is read as ``reshape(B, N, 2, heads, head_dim)``, so ``kv`` is
  ``cat([key, value])`` along the output dimension, in that order;
* the classifier is ``head.*`` here and ``classifier.*`` there. BiFA pops both
  keys after loading, so they must be present even though they are unused.

The result is checked against a real ``mit_b0`` built from the BiFA checkout:
every renamed key must exist in that module with the same shape, and the only
parameters left uncovered may be the ones BiFA itself adds to the backbone
(see ``ADDED_BY_BIFA``), which have no ImageNet counterpart. A silent partial
match is the failure this project has already been bitten by (main record 85),
so --no-verify exists only for a machine without the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path
from typing import Any

import torch

SOURCE_URL = "https://huggingface.co/nvidia/mit-b0/resolve/main/pytorch_model.bin"
PREFIX = "segformer.encoder."

# BiFA does not use SegFormer's backbone unchanged: every block conditions its
# attention on the other date (``attn.cond``) and carries an extra channel
# branch (``attn_realchannel``). Those modules have no ImageNet counterpart and
# stay at their random initialisation -- in the author's code too, which is why
# it loads with ``strict=False``. Any other uncovered parameter would be a real
# gap and still fails.
ADDED_BY_BIFA = ("attn.cond.", "attn_realchannel.")

# Inside one transformer block: Hugging Face name -> original name.
BLOCK_RENAMES = {
    "layer_norm_1": "norm1",
    "layer_norm_2": "norm2",
    "attention.self.query": "attn.q",
    "attention.self.sr": "attn.sr",
    "attention.self.layer_norm": "attn.norm",
    "attention.output.dense": "attn.proj",
    "mlp.dense1": "mlp.fc1",
    "mlp.dense2": "mlp.fc2",
    "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, destination)
    print(f"{destination}: {destination.stat().st_size} bytes, sha256 {sha256(destination)}")
    return destination


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} does not hold a state dict")
    return {str(key): value for key, value in payload.items()}


def renamed(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Hugging Face SegFormer names -> the original names BiFA expects."""

    out: dict[str, torch.Tensor] = {}
    pending_kv: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        name = key[len("segformer."):] if key.startswith("segformer.") else key
        if name.startswith("classifier."):
            out["head." + name.split(".", 1)[1]] = value
            continue
        if not name.startswith("encoder."):
            continue                      # nothing else belongs to the backbone
        name = name[len("encoder."):]
        if name.startswith("patch_embeddings."):
            index, rest = name[len("patch_embeddings."):].split(".", 1)
            rest = rest.replace("layer_norm.", "norm.")
            out[f"patch_embed{int(index) + 1}.{rest}"] = value
            continue
        if name.startswith("layer_norm."):
            index, rest = name[len("layer_norm."):].split(".", 1)
            out[f"norm{int(index) + 1}.{rest}"] = value
            continue
        if not name.startswith("block."):
            raise SystemExit(f"unexpected key {key}")
        stage, block, rest = name[len("block."):].split(".", 2)
        head = f"block{int(stage) + 1}.{block}"
        member, _, parameter = rest.rpartition(".")
        if member in ("attention.self.key", "attention.self.value"):
            # Fused into one projection by the original implementation.
            pending_kv.setdefault(f"{head}|{parameter}", {})[
                member.rsplit(".", 1)[1]
            ] = value
            continue
        if member not in BLOCK_RENAMES:
            raise SystemExit(f"unexpected key {key}")
        out[f"{head}.{BLOCK_RENAMES[member]}.{parameter}"] = value
    for slot, halves in pending_kv.items():
        head, parameter = slot.split("|")
        if set(halves) != {"key", "value"}:
            raise SystemExit(f"{head}: attention has {sorted(halves)}, need key and value")
        out[f"{head}.attn.kv.{parameter}"] = torch.cat(
            [halves["key"], halves["value"]], dim=0
        )
    return out


def verify(state: dict[str, torch.Tensor], repo_root: str) -> dict[str, int]:
    """Every renamed key must fit a real mit_b0 from the checkout, and cover it."""

    from rs_cd.models.external_bifa import (
        HEAD_KEYS,
        _checkout_root,
        _install_shims,
        _register_namespaces,
    )
    import importlib

    _register_namespaces(_checkout_root(repo_root))
    _install_shims()
    backbone = importlib.import_module("models.bifa").mit_b0()
    reference = dict(backbone.state_dict())
    problems = []
    for key, value in state.items():
        if key in HEAD_KEYS:
            continue
        if key not in reference:
            problems.append(f"{key}: not a parameter of mit_b0")
        elif tuple(reference[key].shape) != tuple(value.shape):
            problems.append(
                f"{key}: {tuple(value.shape)} != {tuple(reference[key].shape)}"
            )
    uncovered = [key for key in reference if key not in state and key not in HEAD_KEYS]
    unexpected = [key for key in uncovered if not any(part in key for part in ADDED_BY_BIFA)]
    if problems or unexpected:
        raise SystemExit(
            "the conversion does not match mit_b0: "
            + "; ".join([*problems[:5], *(f"{key}: uncovered" for key in unexpected[:5])])
            + f" ({len(problems)} mismatched, {len(unexpected)} uncovered)"
        )
    added = sorted({key.split(".", 2)[2].rsplit(".", 1)[0] for key in uncovered})
    return {"tensors": len(state), "backbone_tensors": len(reference),
            "randomly_initialised": len(uncovered), "added_by_bifa": added}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="external/hf_mit_b0/pytorch_model.bin",
                        help="the Hugging Face checkpoint; downloaded there with --download")
    parser.add_argument("--download", action="store_true", help=f"fetch {SOURCE_URL}")
    parser.add_argument("--repo-root", default=None, help="the BiFA checkout, for verification")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output", default="external/mit_b0_imagenet.pth")
    args = parser.parse_args()

    source = Path(args.source)
    if args.download:
        fetch(SOURCE_URL, source)
    if not source.is_file():
        raise SystemExit(f"{source} does not exist; pass --download or point --source at it")
    state = renamed(load_state(source))
    if not args.no_verify:
        if args.repo_root is None:
            raise SystemExit("pass --repo-root (the BiFA checkout) or --no-verify")
        print(verify(state, args.repo_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(f"wrote {output} with {len(state)} tensors; sha256 {sha256(output)}")


if __name__ == "__main__":
    main()
