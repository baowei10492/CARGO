from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rs_cd.external_diffregcd import official_path_freeze_decision


EXPECTED_COMMIT = "6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}")
    return value


def _assert_protocol_pair(
    validation: dict[str, Any],
    test: dict[str, Any],
) -> None:
    if validation["status"] != "preregistered":
        raise ValueError("Validation protocol is not preregistered")
    if test["status"] != "frozen_external_test":
        raise ValueError("Test protocol is not frozen_external_test")
    if validation["split"] != "val" or test["split"] != "test":
        raise ValueError("Validation/test split pairing is invalid")
    if test["evaluation"]["inference_paths"] != [
        "official_unmodified"
    ]:
        raise ValueError("Only official_unmodified may enter external test")
    for key in ("external_commit", "seed", "scenarios", "input", "weights"):
        if validation[key] != test[key]:
            raise ValueError(f"Protocol field changed before test: {key}")
    for key in (
        "probability_threshold",
        "interior_border_pixels",
        "use_corruption_valid_mask",
        "metrics",
    ):
        if validation["evaluation"][key] != test["evaluation"][key]:
            raise ValueError(
                f"Evaluation field changed before test: {key}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the public DiffRegCD inference path for one-shot test "
            "using clean reproduction only, never shift performance."
        )
    )
    parser.add_argument("--validation-results", required=True)
    parser.add_argument(
        "--validation-protocol",
        default="configs/p13_diffregcd_validation.yaml",
    )
    parser.add_argument(
        "--test-protocol",
        default="configs/p13_diffregcd_final_test.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    validation_results = Path(args.validation_results).resolve()
    validation_protocol = Path(args.validation_protocol).resolve()
    test_protocol = Path(args.test_protocol).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    result = json.loads(
        validation_results.read_text(encoding="utf-8")
    )
    validation_config = _load_yaml(validation_protocol)
    test_config = _load_yaml(test_protocol)
    _assert_protocol_pair(validation_config, test_config)
    if result.get("split") != "val":
        raise ValueError("Freeze source is not validation")
    if bool(result.get("test_consumed", True)):
        raise ValueError("Validation result unexpectedly consumed test")
    if result.get("external_commit") != EXPECTED_COMMIT:
        raise ValueError("External source commit changed")

    freeze = official_path_freeze_decision(result)
    manifest = {
        **freeze,
        "status": (
            "frozen_external_test"
            if freeze["status"] == "passed"
            else "freeze_failed"
        ),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "external_method": "DiffRegCD",
        "external_commit": EXPECTED_COMMIT,
        "validation_results": str(validation_results),
        "validation_results_sha256": _sha256(validation_results),
        "validation_protocol": str(validation_protocol),
        "validation_protocol_sha256": _sha256(validation_protocol),
        "test_protocol": str(test_protocol),
        "test_protocol_sha256": _sha256(test_protocol),
        "test_policy": {
            "reporting_only_no_selection": True,
            "allowed_runs": 1,
            "result_may_not_change_project_method": True,
        },
    }
    (output_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if freeze["status"] != "passed":
        raise SystemExit("External test freeze gate failed")
    (output_dir / "EXTERNAL_TEST_FROZEN").write_text(
        "DiffRegCD official_unmodified frozen for one-shot test.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

