"""Server smoke test for P21 detectors, run before any training.

Answers, per detector config, the questions that otherwise surface hours
into a run: does the rschange module import inside THIS project's
environment, does it forward at every tile size the matrix uses, do
gradients reach its parameters, and is its inference repeatable. The last
one matters because the dual-path router features compare two passes of the
same detector; a nondeterministic kernel would add noise to every feature.

With --localizer-checkpoint it also checks the adapter against the frozen
P7 model at logit level (main record section 85: a new pipeline must first
reproduce a known number before any new number is read):

  * estimate_registration through the adapter equals the P7 model's own;
  * wrapping the P7 model's own detector, raw and aligned logits equal
    P7.forward_with_forced_offset exactly for a non-zero offset;
  * for each external detector, a zero offset gives raw == aligned exactly.

Random tensors only: no dataset, label or split is read.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rs_cd.models.dual_path_adapter import DualPathAdapter
from rs_cd.runtime import build_model, load_config, seed_everything, select_device
from tools.build_p21_dualpath_checkpoint import localizer_config_from_payload

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _pair(batch: int, size: int, device: torch.device, seed: int):
    generator = torch.Generator().manual_seed(seed)
    images = [torch.rand(batch, 3, size, size, generator=generator) for _ in range(2)]
    return tuple(((image - _MEAN) / _STD).to(device) for image in images)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda")


def _logits(model: nn.Module, image_t1: torch.Tensor, image_t2: torch.Tensor):
    outputs = model(image_t1, image_t2)
    return outputs["change_logits"] if isinstance(outputs, dict) else outputs


def _max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.float() - second.float()).abs().max())


def check_detector(
    config: dict[str, Any],
    *,
    device: torch.device,
    sizes: list[int],
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {"status": "failed", "stage": "build", "warnings": []}
    try:
        start = time.perf_counter()
        model = build_model(config).to(device)
        record["build_seconds"] = round(time.perf_counter() - start, 2)
        record["parameters"] = int(sum(p.numel() for p in model.parameters()))

        model.eval()
        record["forward"] = {}
        for size in sizes:
            record["stage"] = f"forward@{size}"
            image_t1, image_t2 = _pair(batch_size, size, device, seed=size)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad(), _autocast(device, amp):
                _sync(device)
                start = time.perf_counter()
                logits = _logits(model, image_t1, image_t2)
                _sync(device)
                elapsed = time.perf_counter() - start
            expected = (batch_size, 1, size, size)
            if tuple(logits.shape) != expected:
                raise ValueError(f"logits {tuple(logits.shape)} != {expected}")
            if not torch.isfinite(logits.float()).all():
                raise ValueError("non-finite logits")
            record["forward"][str(size)] = {
                "milliseconds": round(elapsed * 1000.0, 1),
                "peak_memory_gib": (
                    round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                    if device.type == "cuda"
                    else None
                ),
            }

        record["stage"] = "repeatability"
        image_t1, image_t2 = _pair(1, sizes[0], device, seed=7)
        with torch.no_grad(), _autocast(device, amp):
            first = _logits(model, image_t1, image_t2)
            second = _logits(model, image_t1, image_t2)
        record["repeat_max_abs_diff"] = _max_abs_diff(first, second)
        if record["repeat_max_abs_diff"] != 0.0:
            record["warnings"].append(
                "inference is not bit-repeatable; dual-path features would "
                "carry kernel noise"
            )

        record["stage"] = "backward"
        model.train()
        image_t1, image_t2 = _pair(batch_size, sizes[0], device, seed=11)
        generator = torch.Generator().manual_seed(13)
        target = (
            torch.rand(batch_size, 1, sizes[0], sizes[0], generator=generator) > 0.9
        ).float().to(device)
        with _autocast(device, amp):
            logits = _logits(model, image_t1, image_t2)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target)
        loss.backward()
        without_gradient = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        record["trainable_parameters_without_gradient"] = len(without_gradient)
        record["examples_without_gradient"] = without_gradient[:10]
        if without_gradient:
            record["warnings"].append("some trainable parameters receive no gradient")

        record["stage"] = "done"
        record["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - the report is the product
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc(limit=12)
    return record


def _load_localizer(payload: Any, device: torch.device) -> nn.Module:
    model = build_model({"model": localizer_config_from_payload(payload)})
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


class _P7DetectorView(nn.Module):
    """The P7 model's own frozen detector, exposed as a plain detector."""

    def __init__(self, p7: nn.Module) -> None:
        super().__init__()
        self.p7 = p7

    def forward(self, image_t1: torch.Tensor, image_t2: torch.Tensor):
        return {"change_logits": self.p7.forward_detector_only(image_t1, image_t2)}


def check_p7_anchor(
    payload: Any, *, device: torch.device, size: int, amp: bool
) -> dict[str, Any]:
    record: dict[str, Any] = {"status": "failed"}
    try:
        adapted = _load_localizer(payload, device)
        reference = _load_localizer(payload, device)
        adapter = DualPathAdapter(adapted, _P7DetectorView(adapted)).to(device).eval()
        image_t1, image_t2 = _pair(1, size, device, seed=21)
        offset = torch.tensor([[3.0, -5.0]], device=device)
        with torch.no_grad(), _autocast(device, amp):
            ours_registration = adapter.estimate_registration(image_t1, image_t2)
            reference_registration = reference.estimate_registration(image_t1, image_t2)
            ours = adapter.forward_with_forced_offset(image_t1, image_t2, offset)
            theirs = reference.forward_with_forced_offset(image_t1, image_t2, offset)
        differences = {
            f"registration.{key}": _max_abs_diff(ours_registration[key], value)
            for key, value in reference_registration.items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        }
        for key in ("raw_change_logits", "aligned_change_logits"):
            differences[key] = _max_abs_diff(ours[key], theirs[key])
        record["max_abs_diff"] = differences
        record["raw_differs_from_aligned"] = _max_abs_diff(
            theirs["raw_change_logits"], theirs["aligned_change_logits"]
        ) > 0.0
        exact = all(value == 0.0 for value in differences.values())
        record["status"] = "ok" if exact and record["raw_differs_from_aligned"] else "mismatch"
    except Exception as error:  # noqa: BLE001
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc(limit=12)
    return record


def check_zero_offset(
    config: dict[str, Any],
    payload: Any,
    *,
    device: torch.device,
    size: int,
    amp: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {"status": "failed"}
    try:
        adapter = DualPathAdapter(
            _load_localizer(payload, device), build_model(config).to(device)
        ).to(device).eval()
        image_t1, image_t2 = _pair(1, size, device, seed=31)
        with torch.no_grad(), _autocast(device, amp):
            registration = adapter.estimate_registration(image_t1, image_t2)
            forced = adapter.forward_with_forced_offset(
                image_t1, image_t2, torch.zeros(1, 2, device=device)
            )
        record["conditional_offset_present"] = (
            "conditional_hard_offset_pixels" in registration
        )
        record["raw_minus_aligned_max_abs"] = _max_abs_diff(
            forced["raw_change_logits"], forced["aligned_change_logits"]
        )
        record["status"] = (
            "ok"
            if record["raw_minus_aligned_max_abs"] == 0.0
            and record["conditional_offset_present"]
            else "mismatch"
        )
    except Exception as error:  # noqa: BLE001
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc(limit=12)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--localizer-checkpoint")
    parser.add_argument("--anchor-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(3407, deterministic=True)
    device = select_device(args.device)
    amp = not args.no_amp
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": amp,
        "detectors": {},
    }
    payload = None
    if args.localizer_checkpoint:
        payload = torch.load(
            args.localizer_checkpoint, map_location="cpu", weights_only=False
        )
        report["localizer_checkpoint"] = args.localizer_checkpoint
        report["p7_anchor"] = check_p7_anchor(
            payload, device=device, size=args.anchor_size, amp=amp
        )

    for path in args.config:
        config = load_config(path)
        record = check_detector(
            config,
            device=device,
            sizes=list(args.sizes),
            batch_size=args.batch_size,
            amp=amp,
        )
        record["detector_name"] = config["model"].get("detector_name")
        if payload is not None and record["status"] == "ok":
            record["dual_path_zero_offset"] = check_zero_offset(
                config, payload, device=device, size=args.anchor_size, amp=amp
            )
        report["detectors"][path] = record
        if device.type == "cuda":
            torch.cuda.empty_cache()

    statuses = [record["status"] for record in report["detectors"].values()]
    statuses += [
        record["dual_path_zero_offset"]["status"]
        for record in report["detectors"].values()
        if "dual_path_zero_offset" in record
    ]
    if "p7_anchor" in report:
        statuses.append(report["p7_anchor"]["status"])
    report["all_ok"] = all(status == "ok" for status in statuses)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for path, record in report["detectors"].items():
        line = f"{record.get('detector_name') or path}: {record['status']}"
        if record["status"] != "ok":
            line += f" at {record['stage']} -> {record.get('error')}"
        elif "dual_path_zero_offset" in record:
            line += f", zero-offset {record['dual_path_zero_offset']['status']}"
        print(line)
        for warning in record.get("warnings", []):
            print(f"  warning: {warning}")
    if "p7_anchor" in report:
        print(f"P7 anchor: {report['p7_anchor']['status']}")
    print(f"all_ok={report['all_ok']}  wrote {output}")


if __name__ == "__main__":
    main()
