"""On-disk memo of pairwise registration estimates (P33).

A matching baseline (rs_cd.p27_baseline_localizers.SimilarityBaseline) calls its estimator with the pair key of the
two images (SHA-1 of both arrays). The key depends on the image content only, so the same (reference, target) pair
seen by another detector, another process or a later run returns the stored estimate: an expensive matcher such as
RoMa runs once per pair. The file is append-only JSON lines; a torn last line (a killed process) is skipped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rs_cd.affine_registration import SimilarityEstimate

Estimator = Callable[[np.ndarray, np.ndarray, str], SimilarityEstimate]


class DiskCachedEstimator:
    def __init__(self, estimator: Estimator, path: Path | str) -> None:
        self.estimator = estimator
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.entries[entry["key"]] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def __len__(self) -> int:
        return len(self.entries)

    def close(self) -> None:
        self._handle.close()

    def __call__(self, reference: np.ndarray, target: np.ndarray, key: str) -> SimilarityEstimate:
        entry = self.entries.get(key)
        if entry is None:
            estimate = self.estimator(reference, target, key)
            fraction = estimate.inlier_fraction
            entry = {"key": key, "matrix": np.asarray(estimate.matrix, dtype=np.float64).reshape(-1).tolist(),
                     "converged": bool(estimate.converged), "inliers": int(estimate.inliers),
                     "inlier_fraction": None if fraction is None or not np.isfinite(fraction) else float(fraction)}
            self._handle.write(json.dumps(entry) + "\n")
            self._handle.flush()
            self.entries[key] = entry
            self.misses += 1
        else:
            self.hits += 1
        return SimilarityEstimate(matrix=np.array(entry["matrix"], dtype=np.float64).reshape(2, 3),
                                  converged=bool(entry["converged"]), inliers=int(entry["inliers"]),
                                  inlier_fraction=entry["inlier_fraction"])
