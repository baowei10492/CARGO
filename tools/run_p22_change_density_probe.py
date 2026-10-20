"""A1: train-region change-density dose-response for the frozen localizer.

docs/P22_CHANGE_DENSITY_DIAGNOSTICS.md. Train-region 3072 px windows are
stratified by their change fraction (train labels, used only to select),
then measured exactly as P22 Gate 0: 6x box-downsampled, frozen LEVIR P7C,
clean plus the physical family. The validation strip and the test raster are
never read.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from rs_cd.change_density import (
    block_change_counts,
    probe_bin_stats,
    probe_decision,
    select_stratified,
    spearman,
    window_change_fractions,
)
from rs_cd.physical_shift import box_downsample, p22_scenarios, scenario_translation
from rs_cd.runtime import save_json, seed_everything, select_device
from tools.run_p22_whu_physical_shift import (
    _offset_row,
    _shift,
    _write_csv,
    estimate_offset,
    load_localizer,
    load_rasters,
    normalize_device,
    unit_region,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p22_change_density_probe.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    probe = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if probe.get("status") != "preregistered":
        raise SystemExit("change-density probe config is not marked preregistered")
    base = yaml.safe_load(Path(probe["base_config"]).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(int(base["seed"]))
    device = select_device(args.device)

    geometry = base["geometry"]
    factor = int(geometry["downsample_factor"])
    window_size = int(geometry["window_size"])
    grid = probe["train_grid"]
    block = int(grid["block"])
    bins = [tuple(pair) for pair in probe["bins"]]
    if window_size != block * int(probe["blocks_per_window"]):
        raise SystemExit("window size must equal blocks_per_window x block")

    rasters = load_rasters(Path(base["raw_root"]), "train", with_label=True)
    counts = block_change_counts(
        rasters["label"], block=block, columns=int(grid["columns"]),
        rows=int(grid["rows"]), threshold=int(probe["mask_threshold"]),
    )
    candidates = window_change_fractions(counts, blocks_per_window=int(probe["blocks_per_window"]), block=block)
    selected = select_stratified(
        candidates, per_bin=int(probe["selection"]["per_bin"]),
        min_spacing=int(probe["selection"]["min_spacing"]), bins=bins,
    )
    availability = [
        sum(1 for c in candidates if bins[i][0] <= c["change_fraction"] < bins[i][1] or (i == len(bins) - 1 and c["change_fraction"] == bins[i][1]))
        for i in range(len(bins))
    ]
    print(f"candidate windows {len(candidates)}; per-bin available {availability}; selected {len(selected)}")

    localizer, localizer_sha, _ = load_localizer(
        Path(base["checkpoints"]["localizer"]), base["checkpoints"].get("localizer_sha256"), device
    )
    scenarios = [s for s in p22_scenarios(factor) if s.family in {"clean", "physical"}]
    amp = bool(base.get("amp", True))
    seed = int(base["seed"])
    records: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for index, window in enumerate(selected, start=1):
        left, top = int(window["left"]), int(window["top"])
        sample_id = f"whu_p22dx_y{top:05d}_x{left:05d}"
        t1 = unit_region(rasters["t1"], left, top, window_size, device)
        t2 = unit_region(rasters["t2"], left, top, window_size, device)
        coarse_t1 = normalize_device(box_downsample(t1.unsqueeze(0), factor))
        by_key: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            dx, dy = scenario_translation(sample_id, scenario, seed)
            conditional, applied, gate = estimate_offset(
                localizer, coarse_t1, _shift(t2, dx, dy), factor=factor, amp=amp
            )
            row = _offset_row(sample_id, scenario, dx, dy, conditional, applied, gate, factor)
            row.update(change_fraction=window["change_fraction"], bin=window["bin"])
            records.append(row)
            by_key[scenario.key] = row
        s2 = by_key["physical_shift:s2:t2"]
        s3 = by_key["physical_shift:s3:t2"]
        outcomes.append(
            {
                "id": sample_id, "left": left, "top": top,
                "change_fraction": window["change_fraction"], "bin": window["bin"],
                "s2_hit": s2["coarse_error"] <= 1.0 + 1e-6, "s3_hit": s3["coarse_error"] <= 1.0 + 1e-6,
                "s2_error": s2["native_error"], "s3_error": s3["native_error"],
                "clean_q": bool(by_key["clean:s0:common"]["q_gate"]),
            }
        )
        print(f"[probe {index}/{len(selected)}] {sample_id} change={window['change_fraction']:.3f} bin={window['bin']}")

    stats = probe_bin_stats(outcomes, bins=bins)
    decision = probe_decision(stats, probe["a1_thresholds"])
    change = [o["change_fraction"] for o in outcomes]
    result = {
        "diagnostic": "p22_a1_train_change_density",
        "verdict": decision["verdict"],
        "decision": decision,
        "bins": stats,
        "candidates_per_bin": availability,
        "spearman": {
            "change_vs_s2_error": spearman(change, [o["s2_error"] for o in outcomes]),
            "change_vs_s3_error": spearman(change, [o["s3_error"] for o in outcomes]),
            "change_vs_clean_q": spearman(change, [1.0 if o["clean_q"] else 0.0 for o in outcomes]),
        },
        "localizer_sha256": localizer_sha,
        "labels_read": "train region only, for stratification",
        "validation_strip_touched": False,
        "test_raster_touched": False,
    }
    _write_csv(output_dir / "probe_offsets.csv", records)
    _write_csv(output_dir / "probe_windows.csv", outcomes)
    save_json(output_dir / "probe_decision.json", result)
    print(f"A1 verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
