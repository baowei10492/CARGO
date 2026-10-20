"""Time per 256 px pair of DiffRegCD's released test path (final measurements, record 192.66).

Runs in the DiffRegCD environment with the LEVIR-CD weights of the P27 DiffRegCD config. Only what the released
test path computes is timed: diffusion features of both images at the configured timesteps and the change
head (model/cd_model_256.py, test()); the flow head, which that path computes only for its loss, is left out,
which favours DiffRegCD. Batch 1, CUDA synchronised, after warm-up; the median is reported, as in the
efficiency table of the proposed pipeline.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

from tools.run_p27_diffregcd_f1 import audit_weights, load_config, weight_paths


def measure(forward: Callable[[torch.Tensor, torch.Tensor], Any], make_pair: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
            *, warmup: int, repeats: int, synchronize: Callable[[], None]) -> dict[str, float]:
    for index in range(warmup):
        forward(*make_pair(index))
    synchronize()
    times = []
    for index in range(repeats):
        a, b = make_pair(warmup + index)
        synchronize()
        start = time.perf_counter()
        forward(a, b)
        synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(times)
    return {"median_ms": statistics.median(times), "mean_ms": statistics.fmean(times),
            "p10_ms": ordered[len(ordered) // 10], "p90_ms": ordered[(9 * len(ordered)) // 10], "repeats": repeats}


def official_forward(adapter) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """The released test path: diffusion features, then netCD(features_A, features_B) (no flow)."""

    @torch.inference_mode()
    def forward(image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        adapter.diffusion.feed_data({"A": image_a, "B": image_b})
        features_a, features_b = [], []
        for timestep in adapter.opt["model_cd"]["t"]:
            encoder_a, decoder_a, encoder_b, decoder_b = adapter.diffusion.get_feats(t=int(timestep))
            if adapter.opt["model_cd"]["feat_type"] == "dec":
                features_a.append(decoder_a)
                features_b.append(decoder_b)
            else:
                features_a.append(encoder_a)
                features_b.append(encoder_b)
        return adapter.joint.netCD(features_a, features_b)

    return forward


def parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p27_diffregcd.yaml")
    parser.add_argument("--dataset", default="LEVIR-CD")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config, _ = load_config(Path(args.config))
    root = Path(config["external"]["root"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    audit = audit_weights(config, args.dataset)

    from tools.evaluate_diffregcd_external import ExternalDiffRegCD

    adapter = ExternalDiffRegCD(root, config_path=root / config["datasets"][args.dataset]["config_json"],
                                checkpoint_paths=weight_paths(config, args.dataset))
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(0)
    pool = [(torch.rand(1, 3, 256, 256, generator=generator) * 2 - 1, torch.rand(1, 3, 256, 256, generator=generator) * 2 - 1)
            for _ in range(8)]
    pairs = [(a.to(device), b.to(device)) for a, b in pool]
    timing = measure(official_forward(adapter), lambda i: pairs[i % len(pairs)], warmup=args.warmup,
                     repeats=args.repeats, synchronize=torch.cuda.synchronize)
    result = {
        "method": "DiffRegCD released test path (diffusion features + change head; flow head excluded)",
        "dataset_weights": args.dataset, "weights": audit, "external_commit": commit,
        "device": torch.cuda.get_device_name(0), "batch": 1, "crop_size": 256,
        "timesteps": list(adapter.opt["model_cd"]["t"]),
        "parameters": {"diffusion_backbone": parameters(adapter.diffusion.netG), "change_head": parameters(adapter.joint.netCD),
                       "registration_head": parameters(adapter.joint.netReg)},
        "timing": timing,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
