"""Frozen CARGO round-1 experiments; no training and no original test-marker changes.

E3/E2 use every specified test crop. E1/E7 export preselected cases; E1 does
not create independent geometric ground truth. All runtime project imports
are delayed so numerical geometry can be checked locally without PyTorch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_number(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def affine_xy(matrix, x, y):
    return matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2], matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]


def to_tile(matrix, top, left):
    m = np.asarray(matrix, dtype=np.float64).copy()
    m[:, 2] += np.array([left, top]) - m[:, 0] * left - m[:, 1] * top
    return m


def smooth_unit_field(tile_size, crop_id, seed, grid):
    rng = np.random.default_rng(stable_number(f"{seed}|round1-local-field|{crop_id}"))
    field = cv2.resize(rng.standard_normal((grid, grid, 2)).astype(np.float32),
                       (tile_size, tile_size), interpolation=cv2.INTER_CUBIC)
    field -= field.mean(axis=(0, 1), keepdims=True)
    rms = float(np.sqrt(np.square(field.astype(np.float64)).sum(axis=2).mean()))
    if rms <= 0:
        raise ValueError("degenerate local field")
    return field / rms


class Geometry:
    """Forward M without field; with field, backward map q -> inv(M)(q+d(q))."""

    def __init__(self, tile, item, matrix, size, field=None):
        self.tile, self.item, self.size = tile, item, size
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.tile_matrix = to_tile(matrix, item["top"], item["left"])
        self.field = field
        self.observed = None
        self.content = None

    def crop(self, array):
        y, x = self.item["top"], self.item["left"]
        return np.ascontiguousarray(array[y:y + self.size, x:x + self.size])

    def make(self):
        if self.observed is not None:
            return
        height, width = self.tile.shape[:2]
        if self.field is None:
            if np.array_equal(self.matrix, np.eye(2, 3)):
                self.observed = self.tile
                self.content = np.ones((height, width), dtype=np.float32)
            else:
                self.observed = cv2.warpAffine(self.tile, self.tile_matrix, (width, height), flags=cv2.INTER_LINEAR)
                self.content = cv2.warpAffine(np.ones((height, width), np.float32), self.tile_matrix,
                                              (width, height), flags=cv2.INTER_LINEAR)
        else:
            y, x = np.mgrid[:height, :width].astype(np.float32)
            inverse = cv2.invertAffineTransform(self.tile_matrix)
            sx, sy = affine_xy(inverse, x + self.field[..., 0], y + self.field[..., 1])
            self.observed = cv2.remap(self.tile, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR)
            self.content = ((sx >= 0) & (sx <= width - 1) & (sy >= 0) & (sy <= height - 1)).astype(np.float32)
        if not np.all(self.crop(self.content) >= 0.99999):
            raise RuntimeError(f"raw crop requires padding: {self.item['crop_id']}")

    def raw(self):
        self.make()
        return self.crop(self.observed)

    def corrected(self, estimate):
        self.make()
        h, w = self.tile.shape[:2]
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        m = to_tile(estimate, self.item["top"], self.item["left"])
        aligned = cv2.warpAffine(self.observed, m, (w, h), flags=flags)
        covered = cv2.warpAffine(self.content, m, (w, h), flags=flags)
        return self.crop(aligned), int(np.count_nonzero(self.crop(covered) < 0.99999))

    def forward_crop_grid(self):
        y, x = np.mgrid[:self.size, :self.size].astype(np.float64)
        px, py = x + self.item["left"], y + self.item["top"]
        mx, my = affine_xy(self.tile_matrix, px, py)
        qx, qy = mx.copy(), my.copy()
        if self.field is not None:
            for _ in range(24):
                displacement = cv2.remap(self.field, qx.astype(np.float32), qy.astype(np.float32),
                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                nx, ny = mx - displacement[..., 0], my - displacement[..., 1]
                delta = max(float(np.max(np.abs(nx - qx))), float(np.max(np.abs(ny - qy))))
                qx, qy = nx, ny
                if delta < 0.01:
                    break
            displacement = cv2.remap(self.field, qx.astype(np.float32), qy.astype(np.float32),
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            residual = np.hypot(qx + displacement[..., 0] - mx, qy + displacement[..., 1] - my)
            if float(residual.max()) > 0.05:
                raise RuntimeError("local-map inverse failed to converge")
        return qx - self.item["left"], qy - self.item["top"]

    def inside_crop(self):
        x, y = self.forward_crop_grid()
        return (x >= 0) & (x <= self.size - 1) & (y >= 0) & (y <= self.size - 1)


def output_remap(probability, matrix):
    h, w = probability.shape
    return cv2.warpAffine(probability, np.asarray(matrix, dtype=np.float64), (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def binary_counts(probability, truth, valid, threshold):
    predicted = probability >= threshold
    return [int(np.count_nonzero(valid & predicted & truth)),
            int(np.count_nonzero(valid & predicted & ~truth)),
            int(np.count_nonzero(valid & ~predicted & truth)),
            int(np.count_nonzero(valid & ~predicted & ~truth))]


def f1(counts):
    c = np.asarray(counts, dtype=np.float64)
    denominator = 2 * c[..., 0] + c[..., 1] + c[..., 2]
    return np.divide(2 * c[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)


def runtime_imports():
    global torch, yaml, TileCache, select_margin_crops, affine_parameters, perturbation_matrix
    global scoring_mask, normalize, TrainedLocalizer, load_localizer, open_detector, harness, open_dataset
    global feature_matrix, selector_probability, registration_features, task_features, TASK_FEATURES
    import torch
    import yaml
    from rs_cd.affine_crops import TileCache, select_margin_crops
    from rs_cd.affine_registration import affine_parameters, perturbation_matrix
    from rs_cd.affine_f1 import scoring_mask
    from rs_cd.datasets.localizer_pairs import normalize
    from rs_cd.localizer_training import TrainedLocalizer, load_localizer
    from rs_cd.p27_detectors import open_detector
    from rs_cd.p27_selector import feature_matrix, selector_probability, registration_features, task_features, TASK_FEATURES
    from tools.run_p27_selector import harness, open_dataset


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        self.settings = self.protocol[args.stage]
        self.name = self.settings["dataset"]
        self.spec = {**self.protocol["datasets"][self.name], "name": self.name}
        self.root = args.output / args.stage
        self.root.mkdir(parents=True, exist_ok=True)
        self.size = self.protocol["crop_size"]
        self.evaluation_path = Path(self.protocol["evaluation_config"])
        evaluation = yaml.safe_load(self.evaluation_path.read_text(encoding="utf-8"))
        self.h = harness(evaluation)
        if self.h["size"] != self.size or self.h["seed"] != self.protocol["seed"]:
            raise RuntimeError("original evaluation protocol changed")
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for formal inference")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(4)
        torch.manual_seed(self.protocol["seed"])
        np.random.seed(self.protocol["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.data = open_dataset(self.spec, "test")
        self.crops = select_margin_crops(self.data.rows, self.spec["tile_size"], self.size, self.protocol["crop_margin"])
        if len(self.crops) != self.spec["expected_test_crops"]:
            raise RuntimeError(f"unexpected crop count: {len(self.crops)}")
        self.cache = TileCache(self.data, self.spec["tile_size"])
        self.files = {"runner": args.runner, "protocol": args.protocol,
                      "evaluation": self.evaluation_path, "manifest": Path(self.spec["manifest"]),
                      "localizer": Path(self.protocol["localizer"]["checkpoint"]),
                      "detector": Path(self.spec["detector"]), "selector": Path(self.spec["selector"])}
        self.hashes = {key: digest(path) for key, path in self.files.items()}
        for key, expected in (("localizer", self.protocol["localizer"]["sha256"]),
                              ("detector", self.spec["detector_sha256"]), ("selector", self.spec["selector_sha256"])):
            if self.hashes[key] != expected:
                raise RuntimeError(f"{key} hash mismatch")
        dependencies = ["src/rs_cd/p27_selector.py", "src/rs_cd/p30_geometry.py", "src/rs_cd/affine_registration.py",
                        "src/rs_cd/affine_crops.py", "src/rs_cd/affine_f1.py", "src/rs_cd/localizer_training.py",
                        "src/rs_cd/models/change_aware_similarity_localizer.py", "src/rs_cd/dual_path.py",
                        "src/rs_cd/runtime.py", "tools/run_p27_selector.py"]
        self.hashes.update({p: digest(p) for p in dependencies})
        identity = {"stage": args.stage, "hashes": self.hashes,
                    "torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                    "device": torch.cuda.get_device_name(0), "tf32": False, "localizer_amp": True, "detector_amp": False}
        identity["id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text(encoding="utf-8"))
            if previous != identity:
                raise RuntimeError("resume identity differs; preserve this output and use a new --output directory")
        else:
            atomic_json(identity_path, identity)
            atomic_json(self.root / "protocol.json", self.protocol)
        self.identity = identity
        self.started = time.time()
        self.done = self.root / "COMPLETE.json"

    def load_models(self):
        info = {}
        for key in ("cpu.max", "memory.max"):
            path = Path("/sys/fs/cgroup") / key
            info[key] = path.read_text().strip() if path.is_file() else None
        info.update(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), opencv=cv2.__version__,
                    numpy=np.__version__, gpu=command_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv"]),
                    git=command_output(["git", "rev-parse", "HEAD"]), threads=4)
        atomic_json(self.root / "environment.json", info)
        self.localizer = TrainedLocalizer(load_localizer(str(self.files["localizer"]), self.device), self.device, amp=True)
        self.detector, detector_config, _, source = open_detector(
            {"checkpoint": str(self.files["detector"]), "sha256": self.spec["detector_sha256"]}, self.device)
        self.threshold = float(detector_config["evaluation"].get("threshold", 0.5))
        self.bundle = torch.load(self.files["selector"], map_location="cpu", weights_only=False)
        self.tau = math.inf if self.bundle["tau"] is None else float(self.bundle["tau"])
        if self.tau != self.spec["tau"] or len(self.bundle["feature_names"]) != 19:
            raise RuntimeError("selector threshold/features do not match fixed protocol")
        atomic_json(self.root / "detector_effective_config.json", detector_config)
        atomic_json(self.root / "assets.json", {"files": {k: str(v) for k, v in self.files.items()},
                    "hashes": self.hashes, "selector_tau": self.tau, "detector_source": source})
        torch.cuda.reset_peak_memory_stats()
        print(f"MODEL_READY {self.name} stage={self.args.stage} tau={self.tau}", flush=True)

    def standard_matrices(self, item):
        return [self.matrix(item, k) for k in self.h["levels"]]

    def matrix(self, item, level, angle=None):
        rotation, scale, tx, ty = affine_parameters(item["crop_id"], level, self.h["ranges"], self.h["seed"])
        return perturbation_matrix(rotation if angle is None else angle, scale, tx, ty, width=self.size, height=self.size)

    def evaluate(self, t1, mask, valid, geometry, *, oracle=False, remap=False, arrays=False):
        raw = geometry.raw()
        forward = self.localizer.estimate_many([t1], [raw])[0]
        backward = self.localizer.estimate_many([raw], [t1])[0]
        if not (np.isfinite(forward.matrix).all() and np.isfinite(backward.matrix).all()):
            raise RuntimeError("non-finite registration matrix")
        aligned, missing = geometry.corrected(forward.matrix)
        images = [raw, aligned]
        if oracle:
            if geometry.field is not None:
                raise ValueError("no affine oracle for a nonlinear field")
            images.append(geometry.corrected(geometry.matrix)[0])
        target_batch = torch.stack([normalize(image) for image in images]).to(self.device)
        reference_batch = normalize(t1).unsqueeze(0).repeat(len(images), 1, 1, 1).to(self.device)
        logits = self.detector(reference_batch, target_batch)["change_logits"].float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("non-finite detector logits")
        inside = torch.from_numpy(scoring_mask([forward.matrix], self.size).astype(np.float32))[None, None].to(self.device)
        task = task_features(logits[0:1], logits[1:2], inside, self.threshold).cpu().tolist()[0]
        features = registration_features(forward.matrix, backward.matrix, forward.misaligned_probability,
                                         self.h["points"], self.size)
        features.update(zip(TASK_FEATURES, task))
        probability = float(selector_probability(self.bundle, feature_matrix([features], self.bundle["feature_names"]))[0])
        adopted = probability >= self.tau
        probabilities = torch.sigmoid(logits).cpu().numpy()[:, 0]
        truth = mask > self.spec["mask_threshold"]
        counts = {"raw": binary_counts(probabilities[0], truth, valid, self.threshold),
                  "estimated": binary_counts(probabilities[1], truth, valid, self.threshold)}
        counts["cargo"] = list(counts["estimated" if adopted else "raw"])
        if oracle:
            counts["oracle_input"] = binary_counts(probabilities[2], truth, valid, self.threshold)
        if remap:
            counts["output_remap"] = binary_counts(output_remap(probabilities[0], geometry.matrix), truth, valid, self.threshold)
        record = {"counts": counts, "adopted": adopted, "selector_score": probability,
                  "estimate": forward.matrix.tolist(), "backward_estimate": backward.matrix.tolist(),
                  "synthetic_matrix": geometry.matrix.tolist(), "features": features,
                  "valid_pixels": int(valid.sum()), "estimated_missing_pixels": missing}
        if arrays:
            return record, {"t1": t1, "t2": raw, "registered_t2": aligned, "label": truth.astype(np.uint8),
                            "valid": valid.astype(np.uint8), "raw_probability": probabilities[0],
                            "registered_probability": probabilities[1],
                            "cargo_probability": probabilities[1 if adopted else 0]}
        return record

    def path(self, item):
        return self.root / "records" / (hashlib.sha256(item["crop_id"].encode()).hexdigest()[:24] + ".json")

    def save_record(self, item, conditions):
        atomic_json(self.path(item), {"tile_id": item["sample_id"], "crop_id": item["crop_id"],
                    "top": item["top"], "left": item["left"], "conditions": conditions})

    def progress(self, index, total, newly_done):
        if index % 100 == 0 or index == total:
            elapsed = time.time() - self.started
            eta = (total - index) * elapsed / max(newly_done, 1) / 60
            memory = torch.cuda.max_memory_allocated() / 2**20
            print(f"PROGRESS {self.args.stage} {index}/{total} new={newly_done} elapsed_min={elapsed/60:.1f} eta_min={eta:.1f} peak_MiB={memory:.0f}", flush=True)

    def run_e3(self):
        newly_done = 0
        for index, item in enumerate(self.crops, 1):
            if not self.path(item).exists():
                t1, _, mask = self.cache.crop(item, self.size)
                valid = scoring_mask(self.standard_matrices(item), self.size)
                conditions = {}
                for level in self.settings["levels"]:
                    geometry = Geometry(self.cache.arrays[1], item, self.matrix(item, level), self.size)
                    conditions[f"level{level}"] = self.evaluate(t1, mask, valid, geometry, oracle=True, remap=True)
                self.save_record(item, conditions)
                newly_done += 1
            self.progress(index, len(self.crops), newly_done)
        self.summarize([f"level{k}" for k in self.settings["levels"]], self.settings["paths"])

    def run_e2(self):
        newly_done = 0
        from rs_cd.affine_registration import AffineRanges
        angle_range = AffineRanges(3.0, 0.80, 1.25, 25.0, 3)
        for index, item in enumerate(self.crops, 1):
            if not self.path(item).exists():
                t1, _, mask = self.cache.crop(item, self.size)
                unit = smooth_unit_field(self.spec["tile_size"], item["crop_id"], self.protocol["seed"], self.settings["field_grid"])
                angle3 = affine_parameters(item["crop_id"], 3, angle_range, self.protocol["seed"])[0]
                geometries = {}
                valid = scoring_mask(self.standard_matrices(item), self.size)
                for condition in self.settings["conditions"]:
                    angle = angle3 * condition["rotation_limit_deg"] / 3
                    matrix = self.matrix(item, condition["base_level"], angle)
                    rms = condition["backward_field_rms_px"]
                    geometry = Geometry(self.cache.arrays[1], item, matrix, self.size, unit * rms if rms else None)
                    valid &= geometry.inside_crop()
                    geometries[condition["name"]] = (geometry, condition)
                if not valid.any():
                    raise RuntimeError("empty common scoring region")
                conditions = {}
                for name, (geometry, condition) in geometries.items():
                    result = self.evaluate(t1, mask, valid, geometry)
                    result["rotation_deg"] = angle3 * condition["rotation_limit_deg"] / 3
                    rms = condition["backward_field_rms_px"]
                    result["field_backward_rms_px"] = rms
                    if rms:
                        inv = cv2.invertAffineTransform(geometry.matrix)
                        field = geometry.field.astype(np.float64)
                        dx = inv[0, 0] * field[..., 0] + inv[0, 1] * field[..., 1]
                        dy = inv[1, 0] * field[..., 0] + inv[1, 1] * field[..., 1]
                        norms = np.hypot(dx, dy)
                        result["field_source_rms_px"] = float(np.sqrt(np.square(norms).mean()))
                        result["field_source_max_px"] = float(norms.max())
                    conditions[name] = result
                self.save_record(item, conditions)
                newly_done += 1
            self.progress(index, len(self.crops), newly_done)
        self.summarize([c["name"] for c in self.settings["conditions"]], self.settings["paths"])

    def chosen_crops(self, count):
        groups = {}
        for crop in self.crops:
            groups.setdefault(crop["sample_id"], []).append(crop)
        rank = sorted(groups, key=lambda tile: stable_number(f"{self.protocol['seed']}|round1-case-tile|{tile}"))[:count]
        chosen = []
        for tile in rank:
            items = groups[tile]
            chosen.append(items[stable_number(f"{self.protocol['seed']}|round1-case-crop|{tile}") % len(items)])
        return chosen

    def export_cases(self):
        count = self.settings["native_pairs"] if self.args.stage == "E1" else self.settings["tiles"]
        chosen = self.chosen_crops(count)
        # Written before any inference/inspection; later presentation cannot alter this diagnostic sample.
        atomic_json(self.root / "selection.json", {"selection": self.settings["selection"], "items": chosen})
        if self.args.stage == "E1" and not (self.root / "tiepoints_to_annotate.csv").exists():
            with (self.root / "tiepoints_to_annotate.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["case", "crop_id", "point", "t1_x", "t1_y", "t2_x", "t2_y", "unchanged_verified", "notes"])
                for index, item in enumerate(chosen, 1):
                    for point in range(1, self.settings["tiepoint_rows_per_pair"] + 1):
                        writer.writerow([f"case{index:03d}", item["crop_id"], point, "", "", "", "", "", ""])
        levels = [0] if self.args.stage == "E1" else self.settings["levels"]
        for index, item in enumerate(chosen, 1):
            t1, _, mask = self.cache.crop(item, self.size)
            valid = scoring_mask(self.standard_matrices(item), self.size)
            for level in levels:
                folder = self.root / "cases" / f"case{index:03d}_L{level}"
                if (folder / "COMPLETE.json").exists():
                    continue
                geometry = Geometry(self.cache.arrays[1], item, self.matrix(item, level), self.size)
                record, arrays = self.evaluate(t1, mask, valid, geometry, arrays=True)
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "arrays.npz", **arrays)
                for key in ("t1", "t2", "registered_t2"):
                    if not cv2.imwrite(str(folder / f"{key}.png"), cv2.cvtColor(arrays[key], cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("image export failed")
                for key in ("label", "valid"):
                    cv2.imwrite(str(folder / f"{key}.png"), arrays[key] * 255)
                for key in ("raw", "registered", "cargo"):
                    prediction = (arrays[f"{key}_probability"] >= self.threshold).astype(np.uint8) * 255
                    cv2.imwrite(str(folder / f"{key}_prediction.png"), prediction)
                atomic_json(folder / "COMPLETE.json", {"crop": item, "level": level, "result": record,
                            "tiepoint_status": "not annotated" if self.args.stage == "E1" else "not applicable"})
            print(f"CASE_DONE {self.args.stage} {index}/{len(chosen)} {item['crop_id']}", flush=True)
        atomic_json(self.done, {"stage": self.args.stage, "exported_cases": len(chosen) * len(levels),
                    "independent_geometry_validated": False,
                    "note": "E1 still needs manual unchanged tiepoints; these cases are not a full-test-set F1 estimate",
                    "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20})

    def summarize(self, conditions, paths):
        records = []
        for item in self.crops:
            record = json.loads(self.path(item).read_text(encoding="utf-8"))
            if record["crop_id"] != item["crop_id"] or set(record["conditions"]) != set(conditions):
                raise RuntimeError("invalid resumed record")
            records.append(record)
        tiles = sorted({r["tile_id"] for r in records})
        tile_index = {tile: i for i, tile in enumerate(tiles)}
        counts = np.zeros((len(tiles), len(conditions), len(paths), 4), dtype=np.int64)
        adopted = np.zeros(len(conditions), dtype=np.float64)
        for record in records:
            for ci, condition in enumerate(conditions):
                result = record["conditions"][condition]
                adopted[ci] += result["adopted"]
                for pi, path in enumerate(paths):
                    counts[tile_index[record["tile_id"]], ci, pi] += result["counts"][path]
        pooled = f1(counts.sum(axis=0))
        rng = np.random.default_rng(self.protocol["seed"])
        repetitions = self.protocol["bootstrap_repetitions"]
        bootstrap = np.empty((repetitions, len(conditions), len(paths)), dtype=np.float64)
        for repeat in range(repetitions):
            draws = rng.integers(0, len(tiles), size=len(tiles))
            bootstrap[repeat] = f1(counts[draws].sum(axis=0))
        result = {"stage": self.args.stage, "dataset": self.name, "crops": len(records), "tiles": len(tiles),
                  "identity": self.identity["id"], "conditions": {},
                  "bootstrap": {"repetitions": repetitions, "seed": self.protocol["seed"],
                                "unit": "source tile; shared draws across all paths and conditions; fixed checkpoints"},
                  "peak_gpu_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                  "this_invocation_elapsed_seconds": time.time() - self.started}
        text = [f"{self.args.stage} {self.name}: {len(records)} crops / {len(tiles)} tiles", "F1 in percent; gains and CIs in percentage points."]
        for ci, condition in enumerate(conditions):
            entry = {"f1": dict(zip(paths, pooled[ci].tolist())), "adoption_rate": float(adopted[ci] / len(records)), "differences": {}}
            reference_paths = [p for p in paths if p != "cargo"]
            for baseline in reference_paths:
                a, b = paths.index("cargo"), paths.index(baseline)
                difference = bootstrap[:, ci, a] - bootstrap[:, ci, b]
                entry["differences"][f"cargo_minus_{baseline}"] = {
                    "estimate": float(pooled[ci, a] - pooled[ci, b]),
                    "ci95": np.quantile(difference, [0.025, 0.975]).tolist()}
            result["conditions"][condition] = entry
            text.append(condition + ": " + ", ".join(f"{p}={pooled[ci,pi]*100:.2f}" for pi, p in enumerate(paths)) + f"; adopted={entry['adoption_rate']*100:.1f}%")
        atomic_json(self.root / "summary.json", result)
        (self.root / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        atomic_json(self.done, {"stage": self.args.stage, "crops": len(records), "identity": self.identity["id"]})
        print("\n".join(text), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("E3", "E2", "E1", "E7"), required=True)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    args.runner = Path(__file__).resolve()
    os.chdir(args.project)
    sys.path[:0] = [str(args.project / "src"), str(args.project)]
    runtime_imports()
    evaluator = Evaluation(args)
    if evaluator.done.exists():
        print(f"KEPT_COMPLETE {args.stage} {evaluator.root}", flush=True)
        return
    evaluator.load_models()
    with torch.inference_mode():
        if args.stage == "E3":
            evaluator.run_e3()
        elif args.stage == "E2":
            evaluator.run_e2()
        else:
            evaluator.export_cases()
    print(f"COMPLETE {args.stage} {evaluator.root}", flush=True)


if __name__ == "__main__":
    main()
