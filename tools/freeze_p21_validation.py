"""Freeze the P21 validation matrix before any test image is read.

Protocol section 7 allows test consumption only once every cell has a
completed validation report. Amendment D (section 12) turns "frozen" into a
file: this tool checks every cell of configs/p21_matrix.yaml, recomputes the
hashes the cell recorded in its SHA256SUMS.txt, adds the hashes of the two
safe-alignment classifiers the cell did not record, and writes
reports/p21/VALIDATION_FROZEN.json exactly once.

scripts/run_p21_final_test.sh refuses to start without that file, and
tools/evaluate_p21_final_test.py re-hashes every test input against it, so a
checkpoint that changes after freezing cannot reach the test split.

    python -m tools.freeze_p21_validation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TEST_DATASET = "LEVIR-CD"
CELLS_ROOT = "reports/p21/cells"
SCENARIO_METRICS = "safe_alignment_experiment/validation_scenario_metrics.csv"
CLASSIFIER_FILES = {
    "safe_alignment_registration": (
        "safe_alignment_experiment/safe_alignment_registration.pth"
    ),
    "safe_alignment_task": "safe_alignment_experiment/safe_alignment_task.pth",
}
TEST_INPUT_KEYS = ("dual_path", *CLASSIFIER_FILES)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_hashes(sums_path: Path) -> dict[str, str]:
    """Read a ``sha256sum`` listing into {path: hash}."""

    hashes: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        hashes[name.lstrip("*").replace("\\", "/")] = digest
    return hashes


def _recorded_config(hashes: dict[str, str], report_dir: Path) -> dict[str, str] | None:
    """The stage-1 config this cell hashed when it ran."""

    matches = {
        name: digest
        for name, digest in hashes.items()
        if name.startswith("configs/p21/") and name.endswith(".yaml")
    }
    if len(matches) != 1:
        return None
    path, digest = next(iter(matches.items()))
    return {"path": path, "sha256": digest}


def _recorded_hash(
    hashes: dict[str, str], report_dir: Path, relative: str
) -> str | None:
    # run_p21_cell.sh hashes paths relative to the project root, and the
    # timestamped report directory name makes the suffix unique.
    suffix = f"{report_dir.name}/{relative}"
    matches = [
        digest for name, digest in hashes.items() if name.endswith(suffix)
    ]
    return matches[0] if len(matches) == 1 else None


def freeze_matrix(
    project_root: Path,
    matrix: dict[str, Any],
    *,
    test_dataset: str = TEST_DATASET,
) -> dict[str, Any]:
    """Check every cell and return the manifest; list all problems at once."""

    problems: list[str] = []
    cells: dict[str, Any] = {}
    test_inputs: dict[str, Any] = {}
    for detector, detector_entry in matrix["detectors"].items():
        for dataset, dataset_entry in matrix["datasets"].items():
            name = f"{detector_entry['slug']}_{dataset_entry['slug']}"
            latest = project_root / CELLS_ROOT / name / "LATEST"
            if not latest.is_file():
                problems.append(
                    f"{name}: no LATEST, so no completed validation report"
                )
                continue
            report_text = latest.read_text(encoding="utf-8").strip()
            report_dir = project_root / report_text
            sums_path = report_dir / "SHA256SUMS.txt"
            if not sums_path.is_file():
                problems.append(f"{name}: {report_text}/SHA256SUMS.txt is missing")
                continue
            hashes = recorded_hashes(sums_path)
            is_test_cell = dataset == test_dataset

            recorded = {
                "cell_summary": "cell_summary.json",
                "validation_scenario_metrics": SCENARIO_METRICS,
            }
            if is_test_cell:
                recorded["dual_path"] = "dual_path.pth"
            files: dict[str, dict[str, str]] = {}
            for key, relative in recorded.items():
                path = report_dir / relative
                expected = _recorded_hash(hashes, report_dir, relative)
                if not path.is_file():
                    problems.append(f"{name}: {relative} is missing")
                    continue
                if expected is None:
                    problems.append(
                        f"{name}: {relative} is not listed in SHA256SUMS.txt"
                    )
                    continue
                actual = sha256_of_file(path)
                if actual != expected:
                    problems.append(
                        f"{name}: {relative} hashes to {actual}, but "
                        f"SHA256SUMS.txt records {expected}"
                    )
                    continue
                files[key] = {"path": f"{report_text}/{relative}", "sha256": actual}

            commit: str | None = None
            if is_test_cell:
                # The cell never hashed its classifiers; they are recorded here,
                # before the test, so they cannot change afterwards.
                for key, relative in CLASSIFIER_FILES.items():
                    path = report_dir / relative
                    if not path.is_file():
                        problems.append(f"{name}: {relative} is missing")
                        continue
                    files[key] = {
                        "path": f"{report_text}/{relative}",
                        "sha256": sha256_of_file(path),
                    }
                commit_path = report_dir / "RSCHANGE_COMMIT"
                if commit_path.is_file():
                    commit = commit_path.read_text(encoding="utf-8").strip()
                else:
                    problems.append(f"{name}: RSCHANGE_COMMIT is missing")

            verdict = None
            raw_clean_f1 = None
            if "cell_summary" in files:
                summary = json.loads(
                    (report_dir / "cell_summary.json").read_text(encoding="utf-8")
                )
                verdict = summary.get("verdict")
                raw_clean_f1 = summary.get("preconditions", {}).get("raw_clean_f1")
                # A detector that predicts nothing still produces a verdict:
                # displacement cannot damage an F1 of zero, so the summary
                # reads no_step_needed (main record 180). The instrument
                # guard does not catch it either, because DSIFN-CD and GZ-CD
                # inherit the detector's LEVIR-CD guard.
                invalid = (
                    raw_clean_f1 is None
                    or not math.isfinite(float(raw_clean_f1))
                    or float(raw_clean_f1) <= 0.0
                )
                if invalid:
                    problems.append(
                        f"{name}: raw clean F1 is {raw_clean_f1}; the run is "
                        "numerically invalid and must not be frozen as a verdict"
                    )
            stage1_best_f1 = None
            guard_path = report_dir / "instrument_guard.json"
            if guard_path.is_file():
                guard = json.loads(guard_path.read_text(encoding="utf-8"))
                stage1_best_f1 = guard.get("stage1_best_val_f1")
                if not guard.get("passed"):
                    problems.append(f"{name}: the instrument guard did not pass")
            else:
                problems.append(f"{name}: instrument_guard.json is missing")
            stage1_config = _recorded_config(hashes, report_dir)
            if stage1_config is None:
                problems.append(
                    f"{name}: SHA256SUMS.txt records no single configs/p21/*.yaml"
                )
            cells[name] = {
                "detector": detector,
                "dataset": dataset,
                "report_dir": report_text,
                "verdict": verdict,
                "raw_clean_f1": raw_clean_f1,
                "stage1_best_val_f1": stage1_best_f1,
                # The recipe the cell actually ran. A cell whose config
                # differs from the shared one is then visible in the
                # manifest rather than only in someone's memory (182).
                "stage1_config": stage1_config,
                "files": files,
            }
            if is_test_cell and commit and all(key in files for key in TEST_INPUT_KEYS):
                test_inputs[detector] = {
                    "cell": name,
                    "report_dir": report_text,
                    "rschange_commit": commit,
                    **{key: files[key] for key in TEST_INPUT_KEYS},
                }
    if problems:
        raise SystemExit(
            "The P21 validation matrix cannot be frozen:\n  "
            + "\n  ".join(problems)
        )
    return {
        "status": "frozen",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "test_dataset": test_dataset,
        "cells": cells,
        "test_inputs": test_inputs,
    }


def write_frozen_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    except FileExistsError as error:
        raise SystemExit(
            f"{path} already exists: the validation matrix is frozen once"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the P21 validation matrix (protocol amendment D)."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--matrix", default="configs/p21_matrix.yaml")
    parser.add_argument(
        "--protocol-doc", default="docs/P21_DETECTOR_AGNOSTIC_PROTOCOL.md"
    )
    parser.add_argument("--output", default="reports/p21/VALIDATION_FROZEN.json")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = root / args.output
    if output.exists():
        raise SystemExit(
            f"{output} already exists: the validation matrix is frozen once"
        )
    protocol_doc = root / args.protocol_doc
    if "修订 D" not in protocol_doc.read_text(encoding="utf-8"):
        raise SystemExit(f"Amendment D is missing from {protocol_doc}")
    matrix_path = root / args.matrix
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))

    manifest = freeze_matrix(root, matrix)
    manifest["matrix"] = {"path": args.matrix, "sha256": sha256_of_file(matrix_path)}
    manifest["protocol_doc"] = {
        "path": args.protocol_doc,
        "sha256": sha256_of_file(protocol_doc),
    }
    write_frozen_manifest(output, manifest)
    for name, cell in manifest["cells"].items():
        print(f"{name:<16} {cell['verdict']}")
    for detector, entry in manifest["test_inputs"].items():
        print(f"test input {detector:<9} {entry['dual_path']['path']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
