"""Matching baselines in the registration module's transformation family: the numbers the paper needs (2026-10-08).

Reads the localizer ablation run (reports/p31/localizer_ablation/LATEST, one JSON per variant) and the per-crop
registration errors (reports/p31/overnight/run/errors/errors_<variant>.csv) and reports, next to the original 4-DoF
baselines:

    ablation   per dataset: level-3 success (%), level-0 F1 change and level-3 F1 gain (percentage points) when the
               registration result is adopted for every pair (Table 7)
    errors     pooled level 3: success, median, 90th percentile and share above 25 px, with the paper's Eq. (10) error
               and with the synthetic transformation alone; level 0: share above 25 px per dataset and pooled, median
               displacement of the estimates up to 25 px (Section 4.1.3); level-3 success by changed-area bin (Fig. 4a)

Writes baseline_fit_check.md / .json and success_bins_extra.csv (Fig. 4a rows of the variants not yet in the paper
data) to --output-dir. Reporting only; nothing is trained or tuned here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools import run_p30
from tools.p31_overnight_tables import OVERNIGHT, bin_table, read_csv, write_csv

DATASETS = ("LEVIR-CD", "DSIFN-CD", "WHU-CD")
VARIANTS = ("v2_full", "v2_no_reliability", "roma_similarity", "roma_ts_ransac", "roma_ts_huber", "sift_similarity",
            "sift_ts_ransac")
PAPER_BIN_VARIANTS = ("v2_full", "v2_no_reliability", "v2_coarse_only", "v2_one_fine_step", "roma_similarity",
                      "sift_similarity")


def ablation_numbers(root: Path, variant: str) -> dict[str, Any] | None:
    path = root / f"{variant}.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name in DATASETS:
        entry = result["datasets"][name]
        out[name] = {"success_l3": 100 * float(entry["localization_success"]["3"]),
                     "l0_change": -100 * float(entry["clean_drop"]["estimated"]),
                     "l3_gain": 100 * float(entry["gain"]["3"]["estimated"])}
    return out


def error_numbers(variant: str) -> dict[str, Any] | None:
    path = OVERNIGHT / "errors" / f"errors_{variant}.csv"
    if not path.is_file():
        return None
    rows = read_csv(path)

    def column(level: int, dataset: str | None = None):
        picked = [r for r in rows if int(r["level"]) == level and (dataset is None or r["dataset"] == dataset)]
        direct = np.array([float(r["ate_raw"]) for r in picked])
        composed = np.array([float(r["ate_diff"]) if r["ate_diff"] not in ("", "None") else np.nan for r in picked])
        return direct, np.where(np.isnan(composed), direct, np.minimum(direct, composed))

    direct, eq10 = column(3)
    out: dict[str, Any] = {
        "level3_pooled": {"crops": int(len(eq10)), "success_eq10": 100 * float((eq10 <= 2).mean()),
                          "success_tstar": 100 * float((direct <= 2).mean()), "median_eq10": float(np.median(eq10)),
                          "p90_eq10": float(np.percentile(eq10, 90)), "over25_eq10": 100 * float((eq10 > 25).mean()),
                          "over25_tstar": 100 * float((direct > 25).mean())},
        "level3_success_eq10": {}, "level0_over25": {}, "level0_median_displacement_up_to_25px": {},
    }
    for name in DATASETS:
        out["level3_success_eq10"][name] = 100 * float((column(3, name)[1] <= 2).mean())
        displacement = column(0, name)[0]
        out["level0_over25"][name] = 100 * float((displacement > 25).mean())
        kept = displacement[displacement <= 25]
        out["level0_median_displacement_up_to_25px"][name] = float(np.median(kept)) if kept.size else None
    out["level0_over25"]["pooled"] = 100 * float((column(0)[0] > 25).mean())
    return out


def render(report: dict[str, Any]) -> str:
    lines = ["# Matching baselines in the registration module's family", ""]
    lines += ["## Table 7 (always registering, DINOv3): success L3 % / L0 change / L3 gain", "",
              "| variant | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for variant, entry in report.items():
        if entry["ablation"]:
            cells = [f"{a['success_l3']:.1f} / {a['l0_change']:+.1f} / {a['l3_gain']:+.1f}"
                     for a in (entry["ablation"][n] for n in DATASETS)]
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines += ["", "## Registration errors (pooled over the three datasets)", "",
              "| variant | L3 success Eq.10 | L3 success T* | L3 median | L3 p90 | L3 >25 Eq.10 | L3 >25 T* | "
              "L0 >25 pooled | L0 >25 per dataset | L0 median disp. <=25 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for variant, entry in report.items():
        e = entry["errors"]
        if not e:
            continue
        p = e["level3_pooled"]
        per = " / ".join(f"{e['level0_over25'][n]:.1f}" for n in DATASETS)
        med = " / ".join("-" if e["level0_median_displacement_up_to_25px"][n] is None
                         else f"{e['level0_median_displacement_up_to_25px'][n]:.2f}" for n in DATASETS)
        lines.append(f"| {variant} | {p['success_eq10']:.1f} | {p['success_tstar']:.1f} | {p['median_eq10']:.2f} | "
                     f"{p['p90_eq10']:.1f} | {p['over25_eq10']:.1f} | {p['over25_tstar']:.1f} | "
                     f"{e['level0_over25']['pooled']:.1f} | {per} | {med} |")
    lines += ["", "Per-dataset order: " + ", ".join(DATASETS) + "."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", default=None, help="default: reports/p31/localizer_ablation/LATEST")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.ablation_dir) if args.ablation_dir else run_p30.pointed("reports/p31/localizer_ablation/LATEST")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {v: {"ablation": ablation_numbers(root, v), "errors": error_numbers(v)} for v in args.variants}
    (output_dir / "baseline_fit_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = render(report)
    (output_dir / "baseline_fit_check.md").write_text(text, encoding="utf-8")
    extra = [v for v in args.variants if v not in PAPER_BIN_VARIANTS and report[v]["errors"]]
    if extra:
        write_csv(output_dir / "success_bins_extra.csv", bin_table(extra))
    print(text)


if __name__ == "__main__":
    main()
