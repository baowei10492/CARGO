from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


AUDITED_FILES = (
    "model/cd_model_256.py",
    "config/levir_test.json",
    "data/CMUCDFlowDataset.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_source(
    external_root: Path,
    *,
    expected_commit: str,
    actual_commit: str,
) -> dict[str, Any]:
    root = external_root.resolve()
    files = {}
    for relative in AUDITED_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing DiffRegCD source: {path}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    source = (root / "model/cd_model_256.py").read_text(
        encoding="utf-8"
    )
    signatures = {
        "training_warps_features": (
            "warped_feats_B = warp_features_all(" in source
            and "self.pred_cm = self.netCD(self.feats_A, warped_feats_B)"
            in source
        ),
        "test_uses_unwarped_features": (
            "self.pred_cm = self.netCD(self.feats_A, self.feats_B)"
            in source
        ),
        "test_writes_pred_corase_flow": (
            "self.pred_corase_flow, self.logits = self.netReg(" in source
        ),
        "visuals_read_pred_flow_coarse": (
            "self.pred_flow_coarse.detach()" in source
        ),
    }
    required = {
        "training_warps_features",
        "test_uses_unwarped_features",
        "test_writes_pred_corase_flow",
        "visuals_read_pred_flow_coarse",
    }
    wiring_ambiguity_detected = all(signatures[key] for key in required)
    commit_matches = actual_commit == expected_commit
    return {
        "status": (
            "completed"
            if commit_matches and wiring_ambiguity_detected
            else "failed"
        ),
        "reporting_only": True,
        "external_root": str(root),
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "commit_matches": commit_matches,
        "files": files,
        "source_signatures": signatures,
        "inference_wiring_ambiguity_detected": wiring_ambiguity_detected,
        "required_validation_paths": [
            "official_unmodified",
            "intended_warped",
        ],
    }


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the DiffRegCD external source and document the official "
            "test/training inference-wiring discrepancy."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--expected-commit",
        default="6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.external_root).expanduser().resolve()
    result = audit_source(
        root,
        expected_commit=str(args.expected_commit),
        actual_commit=_git_commit(root),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
