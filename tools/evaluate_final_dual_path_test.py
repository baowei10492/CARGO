from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.evaluate_safe_alignment_transfer import (
    evaluate_transfer_validation,
    load_safe_classifier_bundle,
    merge_detector_state,
)
from tools.train_evaluate_safe_alignment_classifier import _sha256_of_file


def final_test_summary(
    rows: list[dict[str, Any]],
    *,
    primary_method: str,
) -> dict[str, Any]:
    table = {
        (str(row["scenario"]), str(row["method"])): row for row in rows
    }
    clean = "clean:s0:common"
    shifts = ["shift:s1:t2", "shift:s2:t2", "shift:s3:t2"]
    required = [
        (scenario, method)
        for scenario in [clean, *shifts]
        for method in ("raw", primary_method)
    ]
    missing = [key for key in required if key not in table]
    if missing:
        raise ValueError(f"Final test rows lack required entries: {missing}")

    def f1(scenario: str, method: str) -> float:
        return float(table[(scenario, method)]["f1"])

    shift_gains = [
        f1(scenario, primary_method) - f1(scenario, "raw")
        for scenario in shifts
    ]
    return {
        "status": "completed",
        "reporting_only_no_gate": True,
        "primary_method": primary_method,
        "clean": {
            "raw_f1": f1(clean, "raw"),
            "primary_f1": f1(clean, primary_method),
            "primary_minus_raw": (
                f1(clean, primary_method) - f1(clean, "raw")
            ),
        },
        "shift": {
            scenario: {
                "raw_f1": f1(scenario, "raw"),
                "primary_f1": f1(scenario, primary_method),
                "primary_minus_raw": gain,
            }
            for scenario, gain in zip(shifts, shift_gains, strict=True)
        },
        "mean_shift_gain": sum(shift_gains) / len(shift_gains),
        "test_result_used_for_model_selection": False,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot reporting-only final test of the frozen Raw/Aligned "
            "experts and safe routers. No gate or model selection is made."
        )
    )
    parser.add_argument("--p7-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--classifier-registration", required=True)
    parser.add_argument("--classifier-task", required=True)
    parser.add_argument(
        "--protocol",
        default="configs/p12_levir_final_test.yaml",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not bool(protocol.get("reporting_only_no_selection", False)):
        raise ValueError("Final test protocol must be reporting-only")
    if str(protocol["split"]) != "test":
        raise ValueError("Final evaluation split must be test")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    p7_path = Path(args.p7_checkpoint).expanduser().resolve()
    detector_path = Path(args.detector_checkpoint).expanduser().resolve()
    classifier_paths = {
        "safe_alignment_registration": Path(
            args.classifier_registration
        ).expanduser().resolve(),
        "safe_alignment_task": Path(
            args.classifier_task
        ).expanduser().resolve(),
    }
    sources = {
        "p7_checkpoint": str(p7_path),
        "p7_checkpoint_sha256": _sha256_of_file(p7_path),
        "detector_checkpoint": str(detector_path),
        "detector_checkpoint_sha256": _sha256_of_file(detector_path),
        **{
            f"{name}_checkpoint": str(path)
            for name, path in classifier_paths.items()
        },
        **{
            f"{name}_sha256": _sha256_of_file(path)
            for name, path in classifier_paths.items()
        },
    }

    device = select_device(args.device)
    seed_everything(int(protocol["seed"]))
    p7_checkpoint = torch.load(
        p7_path,
        map_location="cpu",
        weights_only=False,
    )
    detector_checkpoint = torch.load(
        detector_path,
        map_location="cpu",
        weights_only=False,
    )
    config = copy.deepcopy(detector_checkpoint["config"])
    config["data"]["dataset_root"] = args.dataset_root
    config["data"]["manifest"] = args.manifest
    model_config = copy.deepcopy(p7_checkpoint["config"]["model"])
    model_config["registration_only"] = False
    model_config["freeze_registration"] = True
    model_config["offset_decoding"] = "hard"
    model_config["alignment_gate"] = "threshold_nonzero"
    config["model"] = model_config
    model = build_model(config)
    merged_state, replaced = merge_detector_state(
        p7_checkpoint["model"],
        detector_checkpoint["model"],
    )
    model.load_state_dict(merged_state, strict=True)
    model = model.to(device)
    model.eval()

    classifiers = {
        name: load_safe_classifier_bundle(path)
        for name, path in classifier_paths.items()
    }
    dataset = PairedChangeDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split="test",
        image_size=None,
        normalization=config["data"].get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(config["data"].get("mask_threshold", 127)),
        cache_mode=str(config["data"].get("cache_mode", "none")),
    )
    rows, _ = evaluate_transfer_validation(
        model,
        dataset,
        classifiers,
        protocol=protocol,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    summary = final_test_summary(
        rows,
        primary_method=str(protocol["primary_method"]),
    )
    results = {
        **summary,
        "split": "test",
        "test_consumed": True,
        "sources": sources,
        "hybrid_detector_parameters_replaced": replaced,
        "scenario_results": rows,
    }
    save_json(output_dir / "final_test_summary.json", summary)
    save_json(output_dir / "results.json", results)
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved one-shot final test to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
