"""Run P23 on WHU-CD (docs/P23_WHU_STRONG_LOCALIZER_PROTOCOL.md).

screen  Label-free localizer screen. Every generic candidate first passes a
        self-check (a train window shifted against itself); P7C, the
        reference, must reproduce the P22 offsets files row for row. Then
        every candidate estimates clean plus six shifted scenarios on the 60
        P22 Gate 0 train blocks and the 174 P22 validation tiles, on the same
        3072 px windows P22 used. The verdict and the selected localizer
        follow the preregistered rule and use no label. Validation labels are
        read only for the descriptive change-fraction column (P22 Gate 1 has
        already read them).
gate1   Validation tiles with labels, the frozen P8 WHU detector, P7C and the
        selected candidate. The four P7C paths must reproduce the P22 Gate 1
        F1 table before the candidate's paths are reported.

The WHU test raster is never opened: raster loading refuses it.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from rs_cd.geometry import interior_valid_mask, warp_reference_from_target
from rs_cd.metrics.binary import BinaryChangeMetrics
from rs_cd.physical_shift import (
    CLEAN_KEY,
    box_downsample,
    crop,
    gate1_decision,
    p22_scenarios,
    recovery_fraction,
    scenario_translation,
    val_tiles,
    window_origin,
)
from rs_cd.runtime import save_json, seed_everything, select_device
from rs_cd.strong_localizers import (
    REFERENCE_ROLE,
    LightGlueDiskLocalizer,
    PhaseCorrelationLocalizer,
    RomaOutdoorLocalizer,
    SiftLocalizer,
    TranslationEstimate,
    anchor_offsets,
    applied_offset,
    candidate_verdict,
    change_bin_summary,
    clean_agreement,
    compare_f1_tables,
    group_rows,
    row_error,
    scenario_metrics,
    select_candidate,
    selfcheck_summary,
    triggered,
)
from tools.build_whu_scale_probe import blocks_for_scale
from tools.run_p22_whu_physical_shift import (
    _sha256,
    _shift,
    _write_csv,
    estimate_offset,
    label_tile,
    load_detector,
    load_localizer,
    load_rasters,
    normalize_cpu,
    normalize_device,
    unit_region,
)

P7C_PATHS = ("raw", "oracle", "p7c_estimated", "p7c_gated")
CANDIDATE_PATHS = ("estimated", "estimated_gated")
P22_PATH_MAP = {
    "raw": "raw",
    "oracle": "oracle",
    "estimated": "p7c_estimated",
    "estimated_gated": "p7c_gated",
}


class P7CLocalizer:
    """The frozen LEVIR localizer exactly as P22 ran it (6x box view, q gate)."""

    def __init__(self, model: torch.nn.Module, *, factor: int, amp: bool) -> None:
        self.model = model
        self.factor = int(factor)
        self.amp = bool(amp)
        self.last_conditional: torch.Tensor | None = None

    def estimate(
        self, t1: torch.Tensor, t2: torch.Tensor, *, sample_key: str
    ) -> TranslationEstimate:
        coarse_t1 = normalize_device(box_downsample(t1.unsqueeze(0), self.factor))
        conditional, _, gate = estimate_offset(
            self.model, coarse_t1, t2, factor=self.factor, amp=self.amp
        )
        self.last_conditional = conditional
        return TranslationEstimate(
            dx=float(conditional[0]),
            dy=float(conditional[1]),
            converged=True,
            gate=bool(gate),
        )


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    p23 = yaml.safe_load(path.read_text(encoding="utf-8"))
    if p23.get("status") != "preregistered":
        raise SystemExit("P23 config is not marked preregistered")
    p22_path = Path(p23["p22_config"])
    digest = _sha256(p22_path)
    if digest != p23["p22_config_sha256"]:
        raise SystemExit(
            f"{p22_path} has sha256 {digest}, P23 expects {p23['p22_config_sha256']}"
        )
    p22 = yaml.safe_load(p22_path.read_text(encoding="utf-8"))
    if int(p22["seed"]) != int(p23["seed"]):
        raise SystemExit("P23 and P22 seeds differ")
    return p23, p22, digest


def build_candidates(
    p23: dict[str, Any],
    p22: dict[str, Any],
    device: torch.device,
    names: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    estimation = p23["estimation"]
    common = {
        "consensus_threshold": float(estimation["consensus_threshold_working_px"]),
        "min_matches": int(estimation["min_matches"]),
        "max_abs_shift_native": float(estimation["max_abs_shift_native_px"]),
    }
    candidates: dict[str, Any] = {}
    roles: dict[str, str] = {}
    for spec in p23["candidates"]:
        name = str(spec["name"])
        if names is not None and name not in names:
            continue
        kind = spec["kind"]
        if kind == "p7c":
            model, _, _ = load_localizer(
                Path(p22["checkpoints"]["localizer"]),
                p22["checkpoints"].get("localizer_sha256"),
                device,
            )
            localizer: Any = P7CLocalizer(
                model,
                factor=int(p22["geometry"]["downsample_factor"]),
                amp=bool(p22.get("amp", True)),
            )
        elif kind == "phase_correlation":
            localizer = PhaseCorrelationLocalizer(
                working_factor=int(spec["working_factor"]),
                max_abs_shift_native=common["max_abs_shift_native"],
            )
        elif kind == "sift":
            localizer = SiftLocalizer(
                working_factor=int(spec["working_factor"]),
                max_features=int(spec["max_features"]),
                ratio=float(spec["ratio"]),
                **common,
            )
        elif kind == "lightglue_disk":
            localizer = LightGlueDiskLocalizer(
                working_factor=int(spec["working_factor"]),
                max_keypoints=int(spec["max_keypoints"]),
                device=device,
                **common,
            )
        elif kind == "roma_outdoor":
            if spec["use_custom_corr"]:
                require_fused_local_corr(str(spec["fused_local_corr_version"]))
            localizer = RomaOutdoorLocalizer(
                working_factor=int(spec["working_factor"]),
                num_matches=int(spec["num_matches"]),
                device=device,
                seed=int(p23["seed"]),
                model_kwargs={"use_custom_corr": bool(spec["use_custom_corr"])},
                **common,
            )
        else:
            raise SystemExit(f"unknown candidate kind {kind!r}")
        candidates[name] = localizer
        roles[name] = str(spec.get("role", "candidate"))
    if names is not None and set(candidates) != set(names):
        raise SystemExit(f"candidates not in config: {sorted(set(names) - set(candidates))}")
    return candidates, roles


def require_fused_local_corr(expected_version: str) -> None:
    """RoMa's fused local-correlation kernel, pinned to the build for torch 2.5.1.

    Every RoMa result of this line uses the same kernel; a missing or
    different build stops the run instead of silently falling back.
    """

    try:
        installed = importlib.metadata.version("fused-local-corr")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit("fused-local-corr is not installed") from None
    if installed != expected_version:
        raise SystemExit(
            f"fused-local-corr {installed} is installed, the protocol requires {expected_version}"
        )
    importlib.import_module("local_corr")


def weights_manifest() -> dict[str, Any]:
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    files = sorted(path for path in hub.glob("*") if path.is_file()) if hub.exists() else []
    versions: dict[str, Any] = {"torch": torch.__version__}
    for module in ("kornia", "romatch", "cv2"):
        try:
            versions[module] = getattr(importlib.import_module(module), "__version__", "unknown")
        except ImportError:
            versions[module] = None
    for distribution in ("romatch", "kornia", "fused-local-corr"):
        try:
            versions[f"dist:{distribution}"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[f"dist:{distribution}"] = None
    return {
        "hub_checkpoints": str(hub),
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
        "versions": versions,
    }


def timed(device: torch.device, function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def screen_items(p22: dict[str, Any], rasters: dict[str, Any]) -> list[dict[str, Any]]:
    """The P22 Gate 0 train blocks and the P22 Gate 1 validation tiles."""

    geometry = p22["geometry"]
    window = int(geometry["window_size"])
    tile_size = int(geometry["tile_size"])
    region = geometry["train_region"]
    blocks, _ = blocks_for_scale(
        int(region["x_max"]),
        int(region["y_max"]),
        window,
        max_blocks=int(p22["gate0"]["blocks"]),
    )
    items = [
        {
            "set": "train_blocks",
            "id": f"whu_p22_block_y{top:05d}_x{left:05d}",
            "window_left": left,
            "window_top": top,
            "change_fraction": None,
        }
        for left, top in blocks
    ]
    height, width = rasters["t1"].shape[:2]
    for tile in val_tiles(list(geometry["val_columns"]), int(geometry["val_rows"]), tile_size):
        left, top = window_origin(
            tile["left"], tile["top"], tile_size=tile_size, window_size=window,
            raster_width=width, raster_height=height,
        )
        fraction = None
        if "label" in rasters:
            region_label = rasters["label"][top : top + window, left : left + window]
            fraction = float((region_label > 127).mean())
        items.append(
            {
                "set": "val_tiles",
                "id": tile["sample_id"],
                "window_left": left,
                "window_top": top,
                "change_fraction": fraction,
            }
        )
    return items


def estimate_row(
    *,
    item: dict[str, Any],
    name: str,
    scenario: Any,
    gt: tuple[int, int],
    estimate: TranslationEstimate,
    seconds: float,
    trigger_threshold: float,
) -> dict[str, Any]:
    row = {
        "set": item["set"],
        "id": item["id"],
        "candidate": name,
        "scenario": scenario.key,
        "family": scenario.family,
        "severity": scenario.severity,
        "gt_dx": float(gt[0]),
        "gt_dy": float(gt[1]),
        "est_dx": float(estimate.dx),
        "est_dy": float(estimate.dy),
        "converged": bool(estimate.converged),
        "trigger": triggered(estimate, trigger_threshold),
        "support": int(estimate.support),
        "inlier_fraction": estimate.inlier_fraction,
        "seconds": float(seconds),
        "change_fraction": item.get("change_fraction"),
    }
    row["native_error"] = row_error(row)
    return row


def run_selfcheck(
    candidates: dict[str, Any],
    roles: dict[str, str],
    items: list[dict[str, Any]],
    rasters: dict[str, Any],
    *,
    p23: dict[str, Any],
    window: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = p23["selfcheck"]
    blocks = [item for item in items if item["set"] == "train_blocks"][: int(settings["blocks"])]
    rows = []
    for name, localizer in candidates.items():
        if roles[name] == REFERENCE_ROLE:
            continue  # P7C is verified by the offsets anchor instead
        for item in blocks:
            t1 = unit_region(rasters["t1"], item["window_left"], item["window_top"], window, device)
            for dx, dy in settings["shifts"]:
                shifted = _shift(t1, int(dx), int(dy))
                estimate, seconds = timed(
                    device, localizer.estimate, t1, shifted,
                    sample_key=f"selfcheck|{item['id']}|{dx}|{dy}",
                )
                rows.append(
                    {
                        "candidate": name,
                        "id": item["id"],
                        "gt_dx": float(dx),
                        "gt_dy": float(dy),
                        "est_dx": estimate.dx,
                        "est_dy": estimate.dy,
                        "converged": estimate.converged,
                        "seconds": seconds,
                    }
                )
        print(f"[selfcheck] {name} done")
    return selfcheck_summary(rows, tolerance=float(settings["tolerance_native_px"])), rows


def screen_candidate(
    name: str,
    localizer: Any,
    items: list[dict[str, Any]],
    rasters: dict[str, Any],
    *,
    scenarios: list[Any],
    seed: int,
    window: int,
    trigger_threshold: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(items, start=1):
        t1 = unit_region(rasters["t1"], item["window_left"], item["window_top"], window, device)
        t2 = unit_region(rasters["t2"], item["window_left"], item["window_top"], window, device)
        for scenario in scenarios:
            dx, dy = scenario_translation(item["id"], scenario, seed)
            estimate, seconds = timed(
                device, localizer.estimate, t1, _shift(t2, dx, dy),
                sample_key=f"{item['id']}|{scenario.key}",
            )
            rows.append(
                estimate_row(
                    item=item, name=name, scenario=scenario, gt=(dx, dy),
                    estimate=estimate, seconds=seconds,
                    trigger_threshold=trigger_threshold,
                )
            )
        if index % 20 == 0 or index == len(items):
            print(f"[screen {name} {index}/{len(items)}] {item['set']} {item['id']}")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_screen(
    p23: dict[str, Any],
    p22: dict[str, Any],
    p22_config_sha: str,
    device: torch.device,
    output_dir: Path,
    *,
    p22_gate0_dir: Path,
    p22_gate1_dir: Path,
) -> dict[str, Any]:
    screen = p23["screen"]
    geometry = p22["geometry"]
    window = int(geometry["window_size"])
    seed = int(p23["seed"])
    trigger_threshold = float(p23["estimation"]["trigger_threshold_native_px"])
    references = {
        "train_blocks": p22_gate0_dir / "run" / "gate0_offsets.csv",
        "val_tiles": p22_gate1_dir / "run" / "gate1_offsets.csv",
    }
    for path in references.values():
        if not path.is_file():
            raise SystemExit(f"missing P22 reference offsets {path}")

    rasters = load_rasters(
        Path(p22["raw_root"]), "train",
        with_label=bool(screen["describe_change_fraction"]),
    )
    items = screen_items(p22, rasters)
    candidates, roles = build_candidates(p23, p22, device)
    reference_names = [name for name, role in roles.items() if role == REFERENCE_ROLE]
    if reference_names != ["p7c"]:
        raise SystemExit(f"exactly one reference named p7c is required, got {reference_names}")
    manifest = weights_manifest()
    save_json(output_dir / "weights_manifest.json", manifest)
    base = {
        "gate": "p23_screen",
        "p22_config_sha256": p22_config_sha,
        "reference_offsets": {
            key: {"path": str(path), "sha256": _sha256(path)} for key, path in references.items()
        },
        "labels_read": "validation window labels, change_fraction column only (descriptive)",
        "test_raster_touched": False,
        "thresholds": dict(screen),
        "estimation": dict(p23["estimation"]),
    }

    selfcheck, selfcheck_rows = run_selfcheck(
        candidates, roles, items, rasters, p23=p23, window=window, device=device
    )
    _write_csv(output_dir / "selfcheck_offsets.csv", selfcheck_rows)
    save_json(output_dir / "selfcheck.json", selfcheck)
    if not selfcheck["passed"]:
        decision = {**base, "verdict": "selfcheck_failed", "selfcheck": selfcheck}
        save_json(output_dir / "screen_decision.json", decision)
        return decision

    scenarios = p22_scenarios(int(geometry["downsample_factor"]))
    rows: list[dict[str, Any]] = []
    order = ["p7c"] + [name for name in candidates if name != "p7c"]
    anchor: dict[str, Any] = {}
    for name in order:
        rows.extend(
            screen_candidate(
                name, candidates[name], items, rasters, scenarios=scenarios,
                seed=seed, window=window, trigger_threshold=trigger_threshold,
                device=device,
            )
        )
        if name == "p7c":
            for set_name, path in references.items():
                observed = [r for r in rows if r["candidate"] == "p7c" and r["set"] == set_name]
                anchor[set_name] = anchor_offsets(
                    observed, _read_csv(path), tolerance=float(screen["anchor_tolerance_px"])
                )
            anchor["passed"] = all(entry["passed"] for entry in anchor.values())
            save_json(output_dir / "anchor.json", anchor)
            print(json.dumps(anchor, indent=2))
            if not anchor["passed"]:
                _write_csv(output_dir / "screen_offsets.csv", rows)
                decision = {**base, "verdict": "anchor_failed", "anchor": anchor, "selfcheck": selfcheck}
                save_json(output_dir / "screen_decision.json", decision)
                return decision
    _write_csv(output_dir / "screen_offsets.csv", rows)

    summaries: dict[str, Any] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for (name, set_name), members in group_rows(rows, "candidate", "set").items():
        by_scenario = {
            key: scenario_metrics(scenario_rows)
            for (key,), scenario_rows in group_rows(members, "scenario").items()
        }
        summaries.setdefault(name, {"role": roles[name]})[set_name] = by_scenario
        if set_name == "val_tiles":
            verdicts[name] = candidate_verdict(by_scenario, screen)
            summaries[name]["change_bins"] = change_bin_summary(members)
        else:
            summaries[name]["train_verdict"] = candidate_verdict(by_scenario, screen)
    selected = select_candidate(verdicts, roles, margin=float(screen["ranking_margin"]))
    tracking = [
        name for name, verdict in verdicts.items()
        if verdict["tracks"] and roles[name] != REFERENCE_ROLE
    ]
    decision = {
        **base,
        "verdict": "proceed_to_gate1" if selected else "stop_no_candidate_tracks",
        "selected": selected,
        "tracking_candidates": tracking,
        "val_verdicts": verdicts,
        "candidates": summaries,
        "clean_agreement": clean_agreement(
            [row for row in rows if row["set"] == "val_tiles"], ["p7c", *tracking]
        ),
        "anchor": anchor,
        "selfcheck": selfcheck,
        "items": {
            "train_blocks": sum(1 for item in items if item["set"] == "train_blocks"),
            "val_tiles": sum(1 for item in items if item["set"] == "val_tiles"),
        },
    }
    save_json(output_dir / "screen_decision.json", decision)
    return decision


@torch.no_grad()
def run_gate1(
    p23: dict[str, Any],
    p22: dict[str, Any],
    p22_config_sha: str,
    device: torch.device,
    output_dir: Path,
    *,
    selected: str,
    authorisation: dict[str, Any],
    anchor_tolerance: float,
    p22_gate1_dir: Path,
    gate_name: str = "p23_gate1",
    row_groups: list[tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    reference_path = p22_gate1_dir / "gate1_decision.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if "f1" not in reference:
        raise SystemExit(f"{reference_path} has no main-measurement F1 table")

    geometry = p22["geometry"]
    window = int(geometry["window_size"])
    tile_size = int(geometry["tile_size"])
    seed = int(p23["seed"])
    trigger_threshold = float(p23["estimation"]["trigger_threshold_native_px"])
    rounded = bool(p23["estimation"]["round_to_integer_px"])
    border = int(p22["gate1"]["evaluation_border_pixels"])
    rasters = load_rasters(Path(p22["raw_root"]), "train", with_label=True)
    height, width = rasters["t1"].shape[:2]
    tiles = val_tiles(list(geometry["val_columns"]), int(geometry["val_rows"]), tile_size)
    detector, detector_config, detector_sha = load_detector(
        Path(p22["checkpoints"]["detector"]), device
    )
    threshold = float(detector_config["evaluation"].get("threshold", 0.5))
    candidates, _ = build_candidates(p23, p22, device, names={"p7c", selected})
    p7c, chosen = candidates["p7c"], candidates[selected]
    scenarios = p22_scenarios(int(geometry["downsample_factor"]))
    if border < max(scenario.pixels for scenario in scenarios):
        raise SystemExit("evaluation border is smaller than the largest injected shift")
    valid = interior_valid_mask(1, tile_size, tile_size, border, device=device)
    paths = P7C_PATHS + CANDIDATE_PATHS
    metrics = {
        scenario.key: {path: BinaryChangeMetrics(threshold=threshold) for path in paths}
        for scenario in scenarios
    }
    # Descriptive only: the same F1 table per band of validation rows.
    group_metrics = {
        label: {
            scenario.key: {path: BinaryChangeMetrics(threshold=threshold) for path in paths}
            for scenario in scenarios
        }
        for label, _, _ in (row_groups or [])
    }
    group_tiles = {label: 0 for label in group_metrics}
    offset_rows = []
    for index, tile in enumerate(tiles, start=1):
        left, top = tile["left"], tile["top"]
        group = row_group(top // tile_size, row_groups)
        if group is not None:
            group_tiles[group] += 1
        window_left, window_top = window_origin(
            left, top, tile_size=tile_size, window_size=window,
            raster_width=width, raster_height=height,
        )
        box = dict(tile_left=left, tile_top=top, window_left=window_left,
                   window_top=window_top, tile_size=tile_size)
        item = {"set": "val_tiles", "id": tile["sample_id"], "change_fraction": None}
        t1_window = unit_region(rasters["t1"], window_left, window_top, window, device)
        t2_window = unit_region(rasters["t2"], window_left, window_top, window, device)
        t1_tile = normalize_cpu(crop(t1_window, **box))
        mask = label_tile(rasters["label"], left, top, tile_size).unsqueeze(0).to(device)
        for scenario in scenarios:
            dx, dy = scenario_translation(tile["sample_id"], scenario, seed)
            shifted = _shift(t2_window, dx, dy)
            oracle_window = _shift(shifted, -dx, -dy)
            key = f"{tile['sample_id']}|{scenario.key}"

            # P7C paths: the P22 Gate 1 computation, unchanged.
            p7c_estimate = p7c.estimate(t1_window, shifted, sample_key=key)
            conditional = p7c.last_conditional
            p7c_aligned = warp_reference_from_target(
                shifted.unsqueeze(0), conditional.unsqueeze(0).to(shifted)
            )[0]
            first = {
                "raw": crop(shifted, **box),
                "oracle": crop(oracle_window, **box),
                "p7c_estimated": crop(p7c_aligned, **box),
            }
            first["p7c_gated"] = first["p7c_estimated"] if p7c_estimate.gate else first["raw"]

            estimate, seconds = timed(device, chosen.estimate, t1_window, shifted, sample_key=key)
            offset = torch.tensor([applied_offset(estimate, rounded=rounded)], dtype=shifted.dtype, device=shifted.device)
            aligned = warp_reference_from_target(shifted.unsqueeze(0), offset)[0]
            second = {"estimated": crop(aligned, **box)}
            second["estimated_gated"] = (
                second["estimated"] if triggered(estimate, trigger_threshold) else first["raw"]
            )

            for batch, batch_paths in ((first, P7C_PATHS), (second, CANDIDATE_PATHS)):
                t2_batch = normalize_cpu(torch.stack([batch[path] for path in batch_paths]))
                t1_batch = t1_tile.unsqueeze(0).repeat(len(batch_paths), 1, 1, 1)
                logits = detector(t1_batch.to(device), t2_batch.to(device))["change_logits"]
                for position, path in enumerate(batch_paths):
                    targets = [metrics[scenario.key][path]]
                    if group is not None:
                        targets.append(group_metrics[group][scenario.key][path])
                    for target in targets:
                        target.update(
                            logits[position : position + 1], mask, from_logits=True, valid_mask=valid
                        )
            offset_rows.append(
                estimate_row(
                    item=item, name=selected, scenario=scenario, gt=(dx, dy),
                    estimate=estimate, seconds=seconds,
                    trigger_threshold=trigger_threshold,
                )
            )
        print(f"[{gate_name} {index}/{len(tiles)}] {tile['sample_id']}")

    table = {key: {path: metrics[key][path].compute() for path in paths} for key in metrics}
    support = {
        int(v["tp"] + v["fp"] + v["fn"] + v["tn"])
        for by_path in table.values() for v in by_path.values()
    }
    if len(support) != 1:
        raise SystemExit(f"supports differ across scenarios or paths: {support}")
    f1 = {key: {path: float(values["f1"]) for path, values in by_path.items()} for key, by_path in table.items()}
    anchor = compare_f1_tables(
        f1, reference["f1"], P22_PATH_MAP, tolerance=float(anchor_tolerance)
    )
    anchor["reference"] = {"path": str(reference_path), "sha256": _sha256(reference_path)}
    save_json(output_dir / "gate1_anchor.json", anchor)
    base = {
        "gate": gate_name,
        "selected": selected,
        "authorisation": authorisation,
        "p22_config_sha256": p22_config_sha,
        "detector_sha256": detector_sha,
        "anchor": anchor,
        "test_raster_touched": False,
    }
    if not anchor["passed"]:
        # The candidate's numbers are not reported when the harness does not
        # reproduce P22.
        decision = {**base, "verdict": "anchor_failed"}
        save_json(output_dir / "gate1_decision.json", decision)
        return decision

    rows = [
        {"scenario": key, "path": path, **values}
        for key, by_path in table.items() for path, values in by_path.items()
    ]
    _write_csv(output_dir / "gate1_scenario_metrics.csv", rows)
    _write_csv(output_dir / "gate1_offsets.csv", offset_rows)
    candidate_f1 = {
        key: {
            "raw": values["raw"],
            "oracle": values["oracle"],
            "estimated": values["estimated"],
            "estimated_gated": values["estimated_gated"],
        }
        for key, values in f1.items()
    }
    decision = gate1_decision(candidate_f1, anchor_passed=True, thresholds=p22["gate1"])
    by_scenario = {
        key: scenario_metrics(scenario_rows)
        for (key,), scenario_rows in group_rows(offset_rows, "scenario").items()
    }
    decision.update(
        {
            **base,
            "f1": f1,
            "clean_change": {
                path: f1[CLEAN_KEY][path] - f1[CLEAN_KEY]["raw"]
                for path in ("estimated", "estimated_gated", "p7c_estimated", "p7c_gated")
            },
            "validation_offsets": by_scenario,
            "tiles": len(tiles),
            "support": {"border_pixels": border, "valid_pixels_per_scenario": support.pop()},
            "row_groups": row_group_summary(group_metrics, group_tiles),
        }
    )
    save_json(output_dir / "gate1_decision.json", decision)
    return decision


def row_group(row: int, groups: list[tuple[str, int, int]] | None) -> str | None:
    for label, first, last in groups or []:
        if first <= row <= last:
            return label
    return None


def row_group_summary(
    group_metrics: dict[str, dict[str, dict[str, Any]]], group_tiles: dict[str, int]
) -> dict[str, Any]:
    """Descriptive per-band F1 and recovery; never part of a verdict."""

    summary = {}
    for label, by_scenario in group_metrics.items():
        f1 = {
            key: {path: float(metric.compute()["f1"]) for path, metric in by_path.items()}
            for key, by_path in by_scenario.items()
        }
        summary[label] = {
            "tiles": group_tiles[label],
            "f1": f1,
            "recovery": {
                key: recovery_fraction(values["raw"], values["estimated"], values["oracle"])
                for key, values in f1.items()
                if key.startswith("physical_shift")
            },
            "p7c_recovery": {
                key: recovery_fraction(values["raw"], values["p7c_estimated"], values["oracle"])
                for key, values in f1.items()
                if key.startswith("physical_shift")
            },
        }
    return summary


def selected_from_screen(screen_dir: Path) -> tuple[str, dict[str, Any]]:
    """P23: Gate 1 runs only on the candidate a passing screen selected."""

    path = screen_dir / "screen_decision.json"
    screen = json.loads(path.read_text(encoding="utf-8"))
    if screen.get("verdict") != "proceed_to_gate1" or not screen.get("selected"):
        raise SystemExit(f"screen does not authorise Gate 1: {screen.get('verdict')}")
    return str(screen["selected"]), {
        "by": "p23_screen", "screen": str(screen_dir), "screen_sha256": _sha256(path),
    }


def selected_for_p24(p24: dict[str, Any], screen_dir: Path) -> tuple[str, dict[str, Any]]:
    """P24: RoMa is named by the protocol, written after the P23 screen stopped.

    The P23 screen must carry exactly the verdict P24 was written against, so
    the follow-up cannot be pointed at any other screen.
    """

    path = screen_dir / "screen_decision.json"
    screen = json.loads(path.read_text(encoding="utf-8"))
    if screen.get("verdict") != p24["p23_screen_verdict"]:
        raise SystemExit(
            f"P24 follows a P23 screen with verdict {p24['p23_screen_verdict']}, "
            f"got {screen.get('verdict')}"
        )
    if screen_dir.name != p24["p23_screen_run"]:
        raise SystemExit(f"P24 follows screen run {p24['p23_screen_run']}, got {screen_dir.name}")
    return str(p24["selected"]), {
        "by": "p24_protocol",
        "protocol": p24["protocol_doc"],
        "p23_screen": str(screen_dir),
        "p23_screen_sha256": _sha256(path),
        "p23_screen_verdict": screen.get("verdict"),
    }


def load_p24_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    p24 = yaml.safe_load(path.read_text(encoding="utf-8"))
    if p24.get("status") != "preregistered":
        raise SystemExit("P24 config is not marked preregistered")
    p23_path = Path(p24["p23_config"])
    digest = _sha256(p23_path)
    if digest != p24["p23_config_sha256"]:
        raise SystemExit(
            f"{p23_path} has sha256 {digest}, P24 expects {p24['p23_config_sha256']}"
        )
    p23, p22, p22_sha = load_configs(p23_path)
    names = [spec["name"] for spec in p23["candidates"]]
    if p24["selected"] not in names or p24["selected"] == "p7c":
        raise SystemExit(f"P24 selected {p24['selected']!r} is not a P23 candidate")
    return p24, p23, p22, p22_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=["screen", "gate1", "p24-gate1"])
    parser.add_argument("--config", help="P23 config (screen, gate1) or P24 config (p24-gate1)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--p22-gate0-dir", help="P22 Gate 0 report root (screen)")
    parser.add_argument("--p22-gate1-dir", required=True, help="P22 Gate 1 report root")
    parser.add_argument("--screen-dir", help="P23 screen report root (gate1, p24-gate1)")
    args = parser.parse_args()

    if args.gate == "p24-gate1":
        p24, p23, p22, p22_sha = load_p24_config(
            Path(args.config or "configs/p24_whu_roma_gate1.yaml")
        )
    else:
        p24 = None
        p23, p22, p22_sha = load_configs(
            Path(args.config or "configs/p23_whu_strong_localizer.yaml")
        )
    if args.gate != "screen" and not args.screen_dir:
        raise SystemExit("--screen-dir is required for Gate 1")
    if args.gate == "screen" and not args.p22_gate0_dir:
        raise SystemExit("--p22-gate0-dir is required for the screen")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(int(p23["seed"]))
    device = select_device(args.device)
    if args.gate == "screen":
        result = run_screen(
            p23, p22, p22_sha, device, output_dir,
            p22_gate0_dir=Path(args.p22_gate0_dir),
            p22_gate1_dir=Path(args.p22_gate1_dir),
        )
    elif args.gate == "gate1":
        selected, authorisation = selected_from_screen(Path(args.screen_dir))
        result = run_gate1(
            p23, p22, p22_sha, device, output_dir,
            selected=selected,
            authorisation=authorisation,
            anchor_tolerance=float(p23["gate1"]["anchor_tolerance"]),
            p22_gate1_dir=Path(args.p22_gate1_dir),
        )
    else:
        selected, authorisation = selected_for_p24(p24, Path(args.screen_dir))
        result = run_gate1(
            p23, p22, p22_sha, device, output_dir,
            selected=selected,
            authorisation=authorisation,
            anchor_tolerance=float(p24["gate1"]["anchor_tolerance"]),
            p22_gate1_dir=Path(args.p22_gate1_dir),
            gate_name="p24_gate1",
            row_groups=[
                (str(label), int(first), int(last))
                for label, first, last in p24["gate1"]["row_groups"]
            ],
        )
    print(f"{args.gate} verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
