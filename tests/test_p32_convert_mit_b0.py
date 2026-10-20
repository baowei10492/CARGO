"""The one renaming in tools/p32_convert_mit_b0.py that can be silently wrong.

Every other key is a literal rename that the tool verifies against a real
mit_b0. The attention is not: the original implementation fuses key and value
into one ``kv`` projection and reads it back with a reshape, so a fused tensor
in the wrong order has the right shape, loads without complaint, and quietly
swaps the two halves of attention.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.p32_convert_mit_b0 import renamed

HEADS, CHANNELS, TOKENS = 2, 8, 5


def hugging_face_block(generator: torch.Generator) -> dict[str, torch.Tensor]:
    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    return {
        "segformer.encoder.patch_embeddings.0.proj.weight": tensor(CHANNELS, 3, 7, 7),
        "segformer.encoder.patch_embeddings.0.layer_norm.weight": tensor(CHANNELS),
        "segformer.encoder.block.0.0.layer_norm_1.weight": tensor(CHANNELS),
        "segformer.encoder.block.0.0.attention.self.query.weight": tensor(CHANNELS, CHANNELS),
        "segformer.encoder.block.0.0.attention.self.key.weight": tensor(CHANNELS, CHANNELS),
        "segformer.encoder.block.0.0.attention.self.key.bias": tensor(CHANNELS),
        "segformer.encoder.block.0.0.attention.self.value.weight": tensor(CHANNELS, CHANNELS),
        "segformer.encoder.block.0.0.attention.self.value.bias": tensor(CHANNELS),
        "segformer.encoder.block.0.0.attention.output.dense.weight": tensor(CHANNELS, CHANNELS),
        "segformer.encoder.block.0.0.mlp.dwconv.dwconv.weight": tensor(CHANNELS, 1, 3, 3),
        "segformer.encoder.layer_norm.0.weight": tensor(CHANNELS),
        "classifier.weight": tensor(1000, CHANNELS),
        "classifier.bias": tensor(1000),
    }


def test_names_follow_the_original_implementation() -> None:
    state = renamed(hugging_face_block(torch.Generator().manual_seed(3407)))

    assert set(state) == {
        "patch_embed1.proj.weight",
        "patch_embed1.norm.weight",
        "block1.0.norm1.weight",
        "block1.0.attn.q.weight",
        "block1.0.attn.kv.weight",
        "block1.0.attn.kv.bias",
        "block1.0.attn.proj.weight",
        "block1.0.mlp.dwconv.dwconv.weight",
        "norm1.weight",
        "head.weight",
        "head.bias",
    }


def test_the_fused_projection_splits_back_into_key_then_value() -> None:
    """Pins the order against the original ``reshape(B, N, 2, heads, dim)``."""

    generator = torch.Generator().manual_seed(3407)
    source = hugging_face_block(generator)
    fused = renamed(source)["block1.0.attn.kv.weight"]
    fused_bias = renamed(source)["block1.0.attn.kv.bias"]
    x = torch.randn(1, TOKENS, CHANNELS, generator=generator)

    kv = F.linear(x, fused, fused_bias)
    kv = kv.reshape(1, -1, 2, HEADS, CHANNELS // HEADS).permute(2, 0, 3, 1, 4)

    for index, member in enumerate(("key", "value")):
        separate = F.linear(
            x,
            source[f"segformer.encoder.block.0.0.attention.self.{member}.weight"],
            source[f"segformer.encoder.block.0.0.attention.self.{member}.bias"],
        ).reshape(1, TOKENS, HEADS, CHANNELS // HEADS).permute(0, 2, 1, 3)
        torch.testing.assert_close(kv[index], separate)
