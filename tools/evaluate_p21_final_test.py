"""One-shot LEVIR-CD test of one frozen P21 detector (protocol amendment D).

Reporting only: every cell verdict stays with its validation report
(docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md, section 12), and nothing computed
here selects a model, a router, a threshold or a column.

What runs is exactly what the validation cell was judged with:

* the cell's dual-path checkpoint (frozen localizer plus the stage-1
  best-epoch detector), loaded strictly once its sha256 matches the
  validation freeze manifest written by tools/freeze_p21_validation.py;
* both safe-alignment classifiers with the thresholds calibrated on
  validation, checked the same way;
* ``evaluate_transfer_validation``, the evaluator the P12 test used. On the
  same inputs it produces the same rows as the validation evaluator in
  tools/train_evaluate_safe_alignment_classifier.py, which
  tests/test_p21_final_test.py pins on synthetic data.

The consumed marker is written after the model and classifiers have loaded
and before the first test image is read. It is never removed, so a second
run for the same detector refuses to start.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import yaml

from rs_cd.datasets import PairedChangeDataset
from rs_cd.runtime import build_model, save_json, seed_everything, select_device
from tools.evaluate_safe_alignment_transfer import (
    evaluate_transfer_validation,
    load_safe_classifier_bundle,
)
from tools.freeze_p21_validation import (
    CLASSIFIER_FILES,
    TEST_INPUT_KEYS,
    sha256_of_file,
)
from tools.summarize_p21_matrix import (
    SCENARIOS,
    SHIFT_LABELS,
    SHIFT_SCENARIOS,
    break_even_fraction,
)

FIXED_REPORTED_METHODS = ("raw", "always_aligned")
ORACLE_METHOD = "loss_oracle"
DETECTORS = ("BIT", "CD-Lamba", "CDxLSTM")

# What tools/train_evaluate_safe_alignment_classifier.py main() writes onto
# the checkpoint config before building the model for validation. On a
# dual-path config these keys sit beside the localizer and detector blocks
# and change nothing, but copying them keeps both paths literally alike;
# tests/test_p21_final_test.py checks the validation tool still sets them.
VALIDATION_MODEL_OVERRIDES: dict[str, Any] = {
    "registration_only": False,
    "freeze_registration": True,
    "offset_decoding": "hard",
    "alignment_gate": "threshold_nonzero",
}


def check_protocol(protocol: dict[str, Any], *, detector: str) -> None:
    if not bool(protocol.get("reporting_only_no_selection", False)):
        raise ValueError("The P21 test protocol must be reporting-only")
    if str(protocol.get("split")) != "test":
        raise ValueError("The P21 test protocol must read the test split")
    if str(protocol.get("dataset")) != "LEVIR-CD":
        raise ValueError("Amendment D tests LEVIR-CD only")
    if detector not in protocol.get("detectors", []):
        raise ValueError(f"{detector!r} is not a detector of amendment D")
    expected = [*FIXED_REPORTED_METHODS, str(protocol["primary_method"])]
    if list(protocol.get("reported_methods", [])) != expected:
        raise ValueError(f"Amendment D fixes the reported methods to {expected}")


def verified_test_inputs(
    manifest: dict[str, Any],
    detector: str,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Re-hash every test input of ``detector`` against the freeze manifest."""

    if manifest.get("status") != "frozen":
        raise SystemExit("The validation freeze manifest is not frozen")
    entry = manifest.get("test_inputs", {}).get(detector)
    if entry is None:
        raise SystemExit(f"The freeze manifest has no test inputs for {detector}")
    problems: list[str] = []
    paths: dict[str, Path] = {}
    for key in TEST_INPUT_KEYS:
        record = entry[key]
        path = project_root / record["path"]
        if not path.is_file():
            problems.append(f"{detector}: {record['path']} is missing")
            continue
        actual = sha256_of_file(path)
        if actual != record["sha256"]:
            problems.append(
                f"{detector}: {record['path']} hashes to {actual}, "
                f"frozen as {record['sha256']}"
            )
            continue
        paths[key] = path
    if problems:
        raise SystemExit(
            "Test inputs differ from the frozen validation matrix:\n  "
            + "\n  ".join(problems)
        )
    return {
        "paths": paths,
        "report_dir": entry["report_dir"],
        "rschange_commit": entry["rschange_commit"],
    }


def verify_rschange_checkout(config: dict[str, Any], expected_commit: str) -> str:
    """The detector code must be the commit the validation cell ran."""

    repo_root = config["model"]["detector"].get("repo_root")
    if repo_root is None:
        raise SystemExit("The dual-path detector names no rschange repo_root")
    head = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_commit:
        raise SystemExit(
            f"rschange at {repo_root} is at {head}; the validation cell ran "
            f"{expected_commit}"
        )
    modified = subprocess.check_output(
        [
            "git", "-C", str(repo_root), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if modified:
        raise SystemExit(f"rschange at {repo_root} has modified tracked files")
    return head


def load_cell_model(
    payload: dict[str, Any],
    *,
    dataset_root: str,
    manifest: str,
    builder: Callable[[dict[str, Any]], torch.nn.Module] = build_model,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = copy.deepcopy(payload["config"])
    if config.get("model", {}).get("name") != "dual_path_adapter":
        raise ValueError("Expected a P21 dual-path checkpoint")
    config["data"]["dataset_root"] = dataset_root
    config["data"]["manifest"] = manifest
    config["model"].update(VALIDATION_MODEL_OVERRIDES)
    model = builder(config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, config


def claim_test_marker(marker: Path, *, detector: str, output_dir: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "started_irreversible",
        "detector": detector,
        "dataset": "LEVIR-CD",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "allowed_reruns": 0,
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, indent=2) + "\n")
    except FileExistsError as error:
        raise SystemExit(
            f"The LEVIR-CD test of {detector} was already started or consumed:\n"
            + marker.read_text(encoding="utf-8")
        ) from error


def complete_test_marker(marker: Path, summary_path: Path) -> None:
    record = json.loads(marker.read_text(encoding="utf-8"))
    record.update(
        status="completed",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        summary=str(summary_path),
    )
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def summarize_test_rows(
    rows: Iterable[dict[str, Any]],
    *,
    reported_methods: Iterable[str],
    primary_method: str,
) -> dict[str, Any]:
    """The summary amendment D fixed before any test image was read."""

    reported = list(reported_methods)
    table = {(str(row["scenario"]), str(row["method"])): row for row in rows}
    required = [*reported, ORACLE_METHOD]
    missing = [
        (scenario, method)
        for scenario in SCENARIOS
        for method in required
        if (scenario, method) not in table
    ]
    if missing:
        raise ValueError(f"Test rows lack required entries: {missing}")

    def f1(scenario: str, method: str) -> float:
        return float(table[(scenario, method)]["f1"])

    def shift_mean(method: str) -> float:
        return sum(f1(scenario, method) for scenario in SHIFT_SCENARIOS) / len(
            SHIFT_SCENARIOS
        )

    clean = SCENARIOS[0]
    raw_clean = f1(clean, "raw")
    raw_shift_mean = shift_mean("raw")
    methods = {
        method: {
            "clean_f1": f1(clean, method),
            "shift_f1": {
                SHIFT_LABELS[scenario]: f1(scenario, method)
                for scenario in SHIFT_SCENARIOS
            },
            "shift_mean_f1": shift_mean(method),
            "clean_minus_raw": f1(clean, method) - raw_clean,
            "shift_mean_gain_over_raw": shift_mean(method) - raw_shift_mean,
        }
        for method in reported
    }
    selection_rate = {
        ("clean" if scenario == clean else SHIFT_LABELS[scenario]): table[
            (scenario, primary_method)
        ].get("aligned_selection_rate")
        for scenario in SCENARIOS
    }
    per_scenario = {}
    for scenario in SHIFT_SCENARIOS:
        available = f1(scenario, ORACLE_METHOD) - f1(scenario, "raw")
        captured = f1(scenario, primary_method) - f1(scenario, "raw")
        per_scenario[SHIFT_LABELS[scenario]] = {
            "available_gain": available,
            "captured_gain": captured,
            "capture_rate": captured / available if available > 0 else None,
        }
    mean_available = shift_mean(ORACLE_METHOD) - raw_shift_mean
    mean_captured = shift_mean(primary_method) - raw_shift_mean
    return {
        "status": "completed",
        "split": "test",
        "reporting_only_no_gate": True,
        "test_result_used_for_model_selection": False,
        "primary_method": primary_method,
        "reported_methods": reported,
        "methods": methods,
        "primary_aligned_selection_rate": selection_rate,
        "capture_descriptive": {
            "per_scenario": per_scenario,
            "mean_available_gain": mean_available,
            "mean_captured_gain": mean_captured,
            "mean_capture_rate": (
                mean_captured / mean_available if mean_available > 0 else None
            ),
        },
        "break_even_clean_fraction": break_even_fraction(
            f1(clean, primary_method),
            shift_mean(primary_method),
            f1(clean, "always_aligned"),
            shift_mean("always_aligned"),
        ),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot LEVIR-CD test of one frozen P21 detector."
    )
    parser.add_argument("--detector", required=True, choices=DETECTORS)
    parser.add_argument(
        "--freeze-manifest", default="reports/p21/VALIDATION_FROZEN.json"
    )
    parser.add_argument("--protocol", default="configs/p21_levir_final_test.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/datasets/levir_cd.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--consumed-marker", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    protocol_path = Path(args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    check_protocol(protocol, detector=args.detector)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"{output_dir} already exists")

    freeze_path = Path(args.freeze_manifest).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    inputs = verified_test_inputs(freeze, args.detector, project_root=project_root)
    paths = inputs["paths"]

    # Everything that can fail without reading test happens before the marker.
    device = select_device(args.device)
    seed_everything(int(protocol["seed"]))
    payload = torch.load(paths["dual_path"], map_location="cpu", weights_only=False)
    model, config = load_cell_model(
        payload, dataset_root=args.dataset_root, manifest=args.manifest
    )
    rschange_commit = verify_rschange_checkout(config, inputs["rschange_commit"])
    model = model.to(device)
    classifiers = {
        name: load_safe_classifier_bundle(paths[name]) for name in CLASSIFIER_FILES
    }

    marker = Path(args.consumed_marker)
    claim_test_marker(marker, detector=args.detector, output_dir=output_dir)
    output_dir.mkdir(parents=True)
    data = config["data"]
    dataset = PairedChangeDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split="test",
        image_size=None,
        normalization=data.get("normalization", "imagenet"),
        augment=False,
        mask_threshold=int(data.get("mask_threshold", 127)),
        cache_mode=str(data.get("cache_mode", "none")),
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
    for stem in ("scenario_metrics", "sample_decisions"):
        written = output_dir / f"transfer_{stem}.csv"
        if written.exists():
            written.rename(output_dir / f"test_{stem}.csv")

    summary = {
        "detector": args.detector,
        "dataset": "LEVIR-CD",
        "test_samples": len(dataset),
        **summarize_test_rows(
            rows,
            reported_methods=protocol["reported_methods"],
            primary_method=str(protocol["primary_method"]),
        ),
    }
    sources = {
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_of_file(freeze_path),
        "validation_report_dir": inputs["report_dir"],
        "rschange_commit": rschange_commit,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_of_file(protocol_path),
        **{f"{key}_checkpoint": str(path) for key, path in paths.items()},
        **{f"{key}_sha256": sha256_of_file(path) for key, path in paths.items()},
    }
    save_json(output_dir / "final_test_summary.json", summary)
    save_json(
        output_dir / "results.json",
        {
            **summary,
            "test_consumed": True,
            "sources": sources,
            "scenario_results": rows,
        },
    )
    (output_dir / "protocol_resolved.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    complete_test_marker(marker, output_dir / "final_test_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved the one-shot LEVIR-CD test of {args.detector} to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
