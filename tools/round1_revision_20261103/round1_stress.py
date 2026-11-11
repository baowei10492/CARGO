"""Frozen WHU-CD stress evaluation; geometry-only common support, no training.

Reuses the original round1 evaluator without modifying its pinned source.
All source-coverage decisions are made before loading model checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import round1_formal as base


def continuous_field_sample(field, x, y, derivatives=False):
    """Exact bilinear field interpolation with replicated boundary values.

    cv2.remap(INTER_LINEAR) quantizes fractional coordinates to 1/32 pixel;
    that discontinuity must not be used inside a subpixel inverse solver.
    Image generation still uses the original Geometry.make() unchanged.
    """
    height, width = field.shape[:2]
    clipped_x, clipped_y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    ix, iy = clipped_x.astype(np.intp), clipped_y.astype(np.intp)
    ix1, iy1 = np.minimum(ix + 1, width - 1), np.minimum(iy + 1, height - 1)
    fx, fy = (clipped_x - ix)[..., None], (clipped_y - iy)[..., None]
    v00, v10 = field[iy, ix], field[iy, ix1]
    v01, v11 = field[iy1, ix], field[iy1, ix1]
    upper = v00 + (v10 - v00) * fx
    lower = v01 + (v11 - v01) * fx
    value = upper + (lower - upper) * fy
    if not derivatives:
        return value
    gx = (v10 - v00) * (1 - fy) + (v11 - v01) * fy
    gy = lower - upper
    gx *= ((x >= 0) & (x < width - 1))[..., None]
    gy *= ((y >= 0) & (y < height - 1))[..., None]
    return value, gx, gy


class StressGeometry(base.Geometry):
    """Same generated images, with continuous Newton inversion for scoring."""

    def forward_crop_grid(self):
        if self.field is None:
            return super().forward_crop_grid()
        if hasattr(self, "_continuous_grid"):
            return self._continuous_grid
        y, x = np.mgrid[:self.size, :self.size].astype(np.float64)
        mx, my = base.affine_xy(self.tile_matrix, x + self.item["left"], y + self.item["top"])
        mx, my = mx.ravel(), my.ravel()
        qx, qy = mx.copy(), my.copy()
        active = np.arange(mx.size)
        tolerance = 1e-5
        for iteration in range(32):
            displacement, gx, gy = continuous_field_sample(self.field, qx[active], qy[active], True)
            rx = qx[active] + displacement[:, 0] - mx[active]
            ry = qy[active] + displacement[:, 1] - my[active]
            pending = np.hypot(rx, ry) > tolerance
            if not pending.any():
                break
            active, rx, ry, gx, gy = active[pending], rx[pending], ry[pending], gx[pending], gy[pending]
            a, b, c, d = 1 + gx[:, 0], gy[:, 0], gx[:, 1], 1 + gy[:, 1]
            determinant = a * d - b * c
            if np.any(determinant <= 1e-8):
                raise RuntimeError(f"nonpositive local-map Jacobian: {self.item['crop_id']}")
            step_x, step_y = (d * rx - b * ry) / determinant, (a * ry - c * rx) / determinant
            factor = np.minimum(1, 32 / np.maximum(np.hypot(step_x, step_y), 1e-30))
            qx[active] -= factor * step_x
            qy[active] -= factor * step_y
        displacement = continuous_field_sample(self.field, qx, qy)
        residual = np.hypot(qx + displacement[:, 0] - mx, qy + displacement[:, 1] - my)
        maximum = float(residual.max())
        if not np.isfinite(maximum) or maximum > tolerance:
            raise RuntimeError(f"continuous local-map inverse did not converge: {self.item['crop_id']} residual={maximum}")
        self.inverse_stats = {"iterations": iteration + 1, "max_residual_px": maximum,
                              "solver": "continuous bilinear field; bounded Newton; unchanged sampled images"}
        self._continuous_grid = ((qx.reshape(self.size, self.size) - self.item["left"]),
                                 (qy.reshape(self.size, self.size) - self.item["top"]))
        return self._continuous_grid


def source_support(matrix, field, item, size=256, tile_size=512):
    """Require real source content over the entire raw detector input.

    Affine extrema occur at corners. For nonlinear sampling every pixel is
    checked, using the same backward map as Geometry.make().
    """
    if field is None:
        x = np.array([0, size - 1, 0, size - 1], dtype=np.float64) + item["left"]
        y = np.array([0, 0, size - 1, size - 1], dtype=np.float64) + item["top"]
    else:
        y, x = np.mgrid[:size, :size].astype(np.float32)
        x += item["left"]
        y += item["top"]
        d = field[item["top"]:item["top"] + size, item["left"]:item["left"] + size]
        x += d[..., 0]
        y += d[..., 1]
    inverse = cv2.invertAffineTransform(base.to_tile(matrix, item["top"], item["left"]))
    sx, sy = base.affine_xy(inverse, x, y)
    bounds = [float(sx.min()), float(sx.max()), float(sy.min()), float(sy.max())]
    covered = bounds[0] >= 0 and bounds[2] >= 0 and bounds[1] <= tile_size - 1 and bounds[3] <= tile_size - 1
    return bool(covered), bounds


def require_pinned_base(protocol, runner_dir):
    path = runner_dir / "round1_formal.py"
    if base.digest(path) != protocol["base_runner_sha256"]:
        raise RuntimeError("original round1_formal.py hash differs; do not overwrite the running/original script")


class StressEvaluation(base.Evaluation):
    def condition_geometry(self, item):
        from rs_cd.affine_registration import AffineRanges
        unit_angle = base.affine_parameters(item["crop_id"], 3, AffineRanges(3.0, .80, 1.25, 25.0, 3),
                                           self.protocol["seed"])[0] / 3
        unit = base.smooth_unit_field(self.spec["tile_size"], item["crop_id"], self.protocol["seed"],
                                      self.settings["field_grid"])
        result = {}
        for condition in self.settings["conditions"]:
            matrix = self.matrix(item, condition["base_level"], unit_angle * condition["rotation_limit_deg"])
            rms = condition["backward_field_rms_px"]
            result[condition["name"]] = (matrix, unit * rms if rms else None)
        return result

    def select_geometry_support(self):
        all_crops = self.crops
        selected, excluded, valid_counts = [], [], []
        for index, item in enumerate(all_crops, 1):
            geometries = self.condition_geometry(item)
            outside = {}
            for name, (matrix, field) in geometries.items():
                covered, bounds = source_support(matrix, field, item, self.size, self.spec["tile_size"])
                if not covered:
                    outside[name] = bounds
            if outside:
                excluded.append({"crop_id": item["crop_id"], "reason": "raw_input_requires_out_of_tile_content",
                                 "source_bounds_by_condition": outside})
            else:
                valid = base.scoring_mask(self.standard_matrices(item), self.size)
                for matrix, field in geometries.values():
                    geometry = base.Geometry(None, item, matrix, self.size, field)
                    valid &= geometry.inside_crop()
                if not valid.any():
                    excluded.append({"crop_id": item["crop_id"], "reason": "empty_common_scoring_support"})
                else:
                    selected.append(item)
                    valid_counts.append(int(valid.sum()))
            if index % 200 == 0 or index == len(all_crops):
                print(f"GEOMETRY {index}/{len(all_crops)} eligible={len(selected)} excluded={len(excluded)}", flush=True)
        if not selected:
            raise RuntimeError("no crops have common real source coverage")
        fingerprint = hashlib.sha256("\n".join(sorted(c["crop_id"] for c in selected)).encode()).hexdigest()
        expected = self.settings["expected_geometry_selection"]
        if len(selected) != expected["crops"] or fingerprint != expected["fingerprint"]:
            raise RuntimeError("geometry selection differs from the pre-inference local audit; no model inference started")
        report = {
            "identity": self.identity["id"], "selection_stage": "before model checkpoint loading/inference",
            "rule": self.settings["coverage_policy"], "attempted_crops": len(all_crops),
            "attempted_tiles": len({c['sample_id'] for c in all_crops}),
            "eligible_crops": len(selected), "eligible_tiles": len({c['sample_id'] for c in selected}),
            "excluded_crops": len(excluded), "selection_fingerprint": fingerprint, "items": selected, "excluded": excluded,
            "common_scored_pixels_per_crop": {"minimum": min(valid_counts), "median": float(np.median(valid_counts)),
                                              "maximum": max(valid_counts), "total": sum(valid_counts)},
            "interpretation": "Matched geometry-eligible subset of the original WHU test crops; not the full main-table score. Sampling is conditional on real source coverage."
        }
        path = self.root / "geometry_selection.json"
        if path.exists() and json.loads(path.read_text()) != report:
            raise RuntimeError("saved geometry selection differs; preserve this output and use a fresh output directory")
        base.atomic_json(path, report)
        self.crops = selected
        self.selection_report = report
        print(f"SELECTION_LOCKED attempted={len(all_crops)} eligible={len(selected)} excluded={len(excluded)}; no model outputs used", flush=True)

    def summarize(self, conditions, paths):
        final_marker = self.done
        self.done = self.root / "SUMMARY_CORE_WRITTEN.json"
        try:
            super().summarize(conditions, paths)
        finally:
            self.done = final_marker
        path = self.root / "summary.json"
        report = json.loads(path.read_text())
        report["geometry_selection"] = {k: v for k, v in self.selection_report.items() if k not in ("items", "excluded")}
        baseline = report["conditions"]["ts_level3"]["f1"]
        for condition in report["conditions"].values():
            condition["difference_from_ts_level3_pp"] = {p: 100 * (condition["f1"][p] - baseline[p]) for p in paths}
        base.atomic_json(path, report)
        with (self.root / "summary.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"Geometry-only common subset: {self.selection_report['eligible_crops']}/{self.selection_report['attempted_crops']} crops; {self.selection_report['excluded_crops']} excluded before inference.\n")
            handle.write("No training, threshold changes, or test-outcome selection. This subset/mask differs from main-table scoring.\n")
        # The final marker is emitted only after the complete summary is written.
        base.atomic_json(self.done, {"stage": self.args.stage, "crops": len(self.crops),
                                    "identity": self.identity["id"], "status": "completed"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.project, args.protocol, args.output = args.project.resolve(), args.protocol.resolve(), args.output.resolve()
    args.runner = Path(__file__).resolve()
    args.stage = "E2"
    protocol = json.loads(args.protocol.read_text())
    require_pinned_base(protocol, args.runner.parent)
    if protocol["E2"]["inverse_solver"] != "continuous_bilinear_bounded_newton_v1":
        raise RuntimeError("inverse solver protocol mismatch")
    os.chdir(args.project)
    sys.path[:0] = [str(args.project), str(args.project / "src")]
    base.runtime_imports()
    # Only this process uses the stress solver. The original source is untouched.
    base.Geometry = StressGeometry
    evaluator = StressEvaluation(args)
    if evaluator.done.exists():
        print("KEPT_COMPLETE", evaluator.root, flush=True)
        return
    evaluator.select_geometry_support()
    evaluator.load_models()
    with base.torch.inference_mode():
        evaluator.run_e2()
    print("COMPLETE E2_STRESS", evaluator.root, flush=True)


if __name__ == "__main__":
    main()
