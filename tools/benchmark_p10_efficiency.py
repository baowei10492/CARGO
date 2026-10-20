from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from rs_cd.geometry import warp_reference_from_target
from rs_cd.runtime import build_model, save_json, seed_everything, select_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_count(batch_size: int, coverage: float) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be in [0, 1]")
    return min(batch_size, max(0, round(batch_size * coverage)))


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("No timing values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _router_parameter_count(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"Router checkpoint lacks model_state: {path}")
    parameters = sum(
        int(value.numel())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "parameters": parameters,
        "feature_count": len(checkpoint.get("feature_names", [])),
        "probability_threshold": checkpoint.get("probability_threshold"),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    batch_size: int,
    percentiles: list[float],
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    _sync(device)
    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync(device)
        started = time.perf_counter()
        output = function()
        _sync(device)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Benchmark function must return a Tensor")
    mean_ms = statistics.fmean(samples_ms)
    return {
        "iterations": iterations,
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "images_per_second": batch_size / (mean_ms / 1000.0),
        "percentiles_ms": {
            f"p{int(probability * 100):02d}": _percentile(
                samples_ms,
                probability,
            )
            for probability in percentiles
        },
    }


def _count_flops(function: Callable[[], torch.Tensor]) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        output = function()
        if not isinstance(output, torch.Tensor):
            raise TypeError("FLOP function must return a Tensor")
    return int(counter.get_total_flops())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark P10 raw, registration, constant dual-path and "
            "registration-veto conditional execution under one controlled "
            "protocol. Task-veto is reported as constant dual-path because "
            "its task-disagreement features require both logits."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p10_efficiency.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-router", default=None)
    parser.add_argument("--registration-router", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    seed_everything(int(protocol["seed"]))
    device = select_device(args.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if not callable(getattr(model, "estimate_registration", None)):
        raise TypeError("Checkpoint model lacks estimate_registration")
    if not callable(getattr(model, "forward_detector_only", None)):
        raise TypeError("Checkpoint model lacks forward_detector_only")

    amp_enabled = bool(protocol["amp"]) and device.type == "cuda"
    image_size = int(protocol["image_size"])
    warmup = int(protocol["warmup_iterations"])
    iterations = int(protocol["measurement_iterations"])
    percentiles = [
        float(value) for value in protocol["report_percentiles"]
    ]
    coverages = [
        float(value) for value in protocol["conditional_coverages"]
    ]

    def amp_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )

    results: dict[str, Any] = {}
    for batch_size_value in protocol["batch_sizes"]:
        batch_size = int(batch_size_value)
        generator = torch.Generator().manual_seed(
            int(protocol["seed"]) + batch_size
        )
        t1 = torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            generator=generator,
        ).to(device)
        t2 = torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            generator=generator,
        ).to(device)

        def raw_path() -> torch.Tensor:
            with amp_context():
                return model.forward_detector_only(t1, t2)

        def registration_path() -> torch.Tensor:
            with amp_context():
                registration = model.estimate_registration(t1, t2)
            return registration["applied_offset_pixels"]

        def full_dual_path() -> torch.Tensor:
            with amp_context():
                registration = model.estimate_registration(t1, t2)
                offsets = registration["applied_offset_pixels"]
                raw = model.forward_detector_only(t1, t2)
                corrected = warp_reference_from_target(t2, offsets)
                aligned = model.forward_detector_only(t1, corrected)
            return raw + aligned

        raw = _measure(
            raw_path,
            device=device,
            warmup=warmup,
            iterations=iterations,
            batch_size=batch_size,
            percentiles=percentiles,
        )
        registration = _measure(
            registration_path,
            device=device,
            warmup=warmup,
            iterations=iterations,
            batch_size=batch_size,
            percentiles=percentiles,
        )
        full_dual = _measure(
            full_dual_path,
            device=device,
            warmup=warmup,
            iterations=iterations,
            batch_size=batch_size,
            percentiles=percentiles,
        )
        conditional: list[dict[str, Any]] = []
        for requested_coverage in coverages:
            selected = _selected_count(batch_size, requested_coverage)

            def conditional_path() -> torch.Tensor:
                with amp_context():
                    registration_output = model.estimate_registration(t1, t2)
                    offsets = registration_output["applied_offset_pixels"]
                    raw_logits = model.forward_detector_only(t1, t2)
                    if selected == 0:
                        return raw_logits
                    corrected = warp_reference_from_target(
                        t2[:selected],
                        offsets[:selected],
                    )
                    aligned = model.forward_detector_only(
                        t1[:selected],
                        corrected,
                    )
                    output = raw_logits.clone()
                    output[:selected] = aligned
                    return output

            timing = _measure(
                conditional_path,
                device=device,
                warmup=warmup,
                iterations=iterations,
                batch_size=batch_size,
                percentiles=percentiles,
            )
            timing.update(
                {
                    "requested_coverage": requested_coverage,
                    "selected_samples": selected,
                    "effective_coverage": selected / batch_size,
                    "speedup_vs_constant_dual": (
                        full_dual["mean_ms"] / timing["mean_ms"]
                    ),
                    "overhead_vs_raw": (
                        timing["mean_ms"] / raw["mean_ms"]
                    ),
                }
            )
            conditional.append(timing)
        results[str(batch_size)] = {
            "raw_detector": raw,
            "registration_only": registration,
            "constant_dual_path_task_veto_core": full_dual,
            "registration_veto_conditional_curve": conditional,
            "constant_dual_overhead_vs_raw": (
                full_dual["mean_ms"] / raw["mean_ms"]
            ),
        }

    flop_counts: dict[str, Any] = {"status": "disabled"}
    if bool(protocol.get("count_flops", True)):
        batch_size = int(protocol.get("flop_batch_size", 1))
        t1 = torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            device=device,
        )
        t2 = torch.rand_like(t1)

        def flop_raw() -> torch.Tensor:
            with amp_context():
                return model.forward_detector_only(t1, t2)

        def flop_registration() -> torch.Tensor:
            with amp_context():
                result = model.estimate_registration(t1, t2)
            return result["applied_offset_pixels"]

        def flop_dual() -> torch.Tensor:
            with amp_context():
                result = model.estimate_registration(t1, t2)
                offsets = result["applied_offset_pixels"]
                raw = model.forward_detector_only(t1, t2)
                aligned = model.forward_detector_only(
                    t1,
                    warp_reference_from_target(t2, offsets),
                )
            return raw + aligned

        flop_counts = {
            "status": "completed",
            "scope": (
                "PyTorch supported-operator FLOPs; grid sampling and "
                "unsupported operators may be omitted"
            ),
            "batch_size": batch_size,
            "raw_detector": _count_flops(flop_raw),
            "registration_only": _count_flops(flop_registration),
            "constant_dual_path": _count_flops(flop_dual),
        }

    total_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    registration_parameters = sum(
        int(parameter.numel())
        for parameter in model.registration.parameters()
    )
    output = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "torch": torch.__version__,
        "amp": amp_enabled,
        "image_size": image_size,
        "parameters": {
            "complete_model": total_parameters,
            "registration_module": registration_parameters,
            "frozen_detector_and_other": (
                total_parameters - registration_parameters
            ),
            "task_router": _router_parameter_count(
                Path(args.task_router).expanduser().resolve()
                if args.task_router
                else None
            ),
            "registration_router": _router_parameter_count(
                Path(args.registration_router).expanduser().resolve()
                if args.registration_router
                else None
            ),
        },
        "flops": flop_counts,
        "timing": results,
        "interpretation_constraints": {
            "task_veto": (
                "Requires Raw and Aligned task disagreement; use constant "
                "dual-path timing, not the conditional curve."
            ),
            "registration_veto": (
                "Can decide before Aligned detector execution; conditional "
                "curve is an executable early-exit schedule."
            ),
            "absolute_latency": (
                "Host/vGPU dependent. Compare ratios only within this run."
            ),
        },
    }
    save_json(args.output, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
