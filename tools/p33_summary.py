"""P33 read-out (docs/P33_REVIEWER_EXPERIMENTS.md): every result next to its reading rule. CPU only, reporting only.

    1, 2  registrar + the paper's selector recipe (reports/p33/selector/<localizer>): per combination the level-0 F1
          change and the level-3 gain of the selector over never registering; over the combinations the level-0 mean
          and worst, the degraded count (level-0 drop of more than 0.5 points, Table 6) and the mean level-3 gain,
          next to the paper's selector on the same combinations (frozen test runs)
    2, 5  always-registering rows (Table 7) and pooled registration errors (Figure 4) of the retrained modules and the
          classical baselines, next to the registration module (tools.p31_baseline_fit_check)
    3     S2Looking: level-3 gain with tile-bootstrap interval, level-0 change, level-0 adoption rate
    4     interpolation control: per combination the adoption rate and the selector's level-0 change at level 0 and
          under the control; DiffRegCD's DSIFN-CD gap between levels 0 and 3 recovered by the control
    6     RoMa on identical pairs; the repeated registration-module numbers must equal reports/p33/identity
Pieces that have not finished are listed as missing. Output: reports/p33/summary/summary.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from tools.p31_baseline_fit_check import ablation_numbers, error_numbers
from tools.p31_selector_ablation import DATASETS
from tools.p31_statistics import Cell, check_cell, draws, paired
from tools.p33_selector_runs import frozen_anchor
from tools import run_p30

ROOT = Path("reports/p33")
TABLE6_ROWS = ("dinov3", "bit", "cdlamba", "cdxlstm")
RETRAINED = ("v2r_conf_only", "v2r_no_change_label")
INTERP_ROWS = ("dinov3", "bit", "cdlamba", "cdxlstm", "snunet", "bifa")
ALWAYS_VARIANTS = ("v2_full", "v2_no_reliability", *RETRAINED, "roma_ts_ransac", "phasecorr_fm", "ecc_ts")
UNSAFE = -0.005          # Table 6: a combination is degraded when its level-0 F1 drops by more than 0.5 points


def read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def changes(f1: dict[str, dict[str, float]]) -> dict[str, float]:
    """Selector and always registering minus never registering, levels 0 and 3."""
    return {"l0": f1["0"]["selector"] - f1["0"]["raw"], "l3": f1["3"]["selector"] - f1["3"]["raw"],
            "always_l0": f1["0"]["estimated"] - f1["0"]["raw"], "always_l3": f1["3"]["estimated"] - f1["3"]["raw"]}


def aggregate(cells: dict[str, dict[str, float]]) -> dict[str, Any]:
    l0 = [cell["l0"] for cell in cells.values()]
    return {"combinations": len(cells), "l0_mean": float(np.mean(l0)), "l0_worst": float(min(l0)),
            "degraded": int(sum(value < UNSAFE for value in l0)),
            "l3_gain": float(np.mean([cell["l3"] for cell in cells.values()]))}


# ------------------------------------------------------------------------------------------------- rules 1, 2
def selector_block(localizer: str, rows: Sequence[str]) -> dict[str, Any]:
    cells, reference, missing = {}, {}, []
    for row in rows:
        for name in DATASETS:
            key = f"{row}/{name}"
            entry = read(ROOT / "selector" / localizer / row / f"test_{name}.json")
            if entry is None:
                missing.append(key)
                continue
            reference[key] = changes(frozen_anchor(row, name))
            cells[key] = {**changes(entry["f1"]), "tau": entry["tau"],
                          "adopted_l0": entry["selection_rate"]["0"]["selector"],
                          "adopted_l3": entry["selection_rate"]["3"]["selector"]}
    out: dict[str, Any] = {"localizer": localizer, "cells": cells, "missing": missing}
    if cells:
        out["mean"] = aggregate(cells)
        out["paper_same_cells"] = aggregate(reference)
    return out


def rule_1(block: dict[str, Any], paper: dict[str, Any]) -> str:
    if block["missing"]:
        return f"incomplete ({len(block['missing'])} combinations missing)"
    mean = block["mean"]
    if mean["l3_gain"] >= paper["l3_gain"] - 0.005 and mean["degraded"] <= paper["degraded"]:
        return "reaches the paper's selector: the selector carries the accuracy, the module's advantage is cost"
    return "below the paper's selector: report the registration module's advantage with the numbers"


def rule_2(block: dict[str, Any]) -> str:
    if block["missing"]:
        return f"incomplete ({len(block['missing'])} combinations missing)"
    mean, paper = block["mean"], block["paper_same_cells"]
    l3, l0 = mean["l3_gain"] - paper["l3_gain"], mean["l0_mean"] - paper["l0_mean"]      # retrained minus full
    not_worse = l0 >= 0 and mean["degraded"] <= paper["degraded"]
    numbers = f"L3 {pp(l3)}, L0 mean {pp(l0)} points, degraded {mean['degraded']} vs {paper['degraded']}"
    if l3 > 0.01 and not_worse:
        return f"clearly better than the full module ({numbers}): the authors decide with these numbers"
    if l3 >= -0.01 and not_worse:
        return (f"within the rule-2 margin of the full module ({numbers}): not reported in the paper, wording "
                "unchanged (amended rule 2, 2026-10-09); numbers kept for the response letter")
    return f"outside the rule-2 margin ({numbers}): the full module's advantage is reported"


# ------------------------------------------------------------------------------------------------- rules 2, 5
def always_block() -> dict[str, Any]:
    root = run_p30.pointed("reports/p31/localizer_ablation/LATEST")
    out: dict[str, Any] = {"ablation_run": str(root), "variants": {}, "missing": []}
    for variant in ALWAYS_VARIANTS:
        ablation, errors = ablation_numbers(root, variant), error_numbers(variant)
        if ablation is None or errors is None:
            out["missing"].append(variant)
        seconds = read(root / f"{variant}.json")
        out["variants"][variant] = {
            "ablation": ablation, "errors": errors,
            "median_seconds": None if seconds is None else
            {name: seconds["datasets"][name].get("median_seconds") for name in DATASETS}}
    return out


def rule_5(block: dict[str, Any], variant: str) -> str:
    module, other = block["variants"]["v2_full"]["errors"], block["variants"][variant]["errors"]
    if not module or not other:
        return "incomplete"
    a, b = module["level3_pooled"], other["level3_pooled"]
    beats = (b["success_eq10"] > a["success_eq10"] and b["over25_eq10"] < a["over25_eq10"]
             and other["level0_over25"]["pooled"] < module["level0_over25"]["pooled"])
    return "beats the module on success and gross errors: revisit the claim" if beats else "no claim changes"


# ---------------------------------------------------------------------------------------------------- rule 3
def s2looking_block(repeats: int, seed: int) -> dict[str, Any]:
    out_dir = ROOT / "selector" / "v2" / "dinov3_s2looking"
    entry = read(out_dir / "test_S2Looking.json")
    if entry is None:
        return {"missing": True}
    bundle = torch.load(out_dir / "selector_S2Looking.pth", map_location="cpu", weights_only=False)
    cell = Cell("dinov3_s2looking", "dinov3_s2looking", "S2Looking", out_dir / "records_S2Looking.csv", bundle)
    check_cell(cell, entry["f1"])
    intervals = paired(cell, cell.selected(), cell.counts["raw"], draws(cell.tiles, repeats, seed))
    return {"missing": False, "crops": entry["crops"], "tiles": cell.tiles, "tau": entry["tau"],
            "f1": entry["f1"], "selector_minus_never": intervals, **changes(entry["f1"]),
            "adopted_l0": entry["selection_rate"]["0"]["selector"],
            "adopted_l3": entry["selection_rate"]["3"]["selector"]}


# ---------------------------------------------------------------------------------------------------- rule 4
def interp_block() -> dict[str, Any]:
    cells, missing, flagged = {}, [], []
    for row in INTERP_ROWS:
        for name in DATASETS:
            key = f"{row}/{name}"
            entry = read(ROOT / "interp" / row / f"interp_{name}.json")
            if entry is None:
                missing.append(key)
                continue
            level0, control = entry["level0"], entry["interp"]
            cell = {"adopted_level0": level0["adoption_rate"], "adopted_control": control["adoption_rate"],
                    "l0_change_level0": level0["f1"]["selector"] - level0["f1"]["raw"],
                    "l0_change_control": control["f1"]["selector"] - control["f1"]["raw"],
                    "anchor_passed": entry.get("anchor", {}).get("passed")}
            cell["adoption_difference"] = cell["adopted_control"] - cell["adopted_level0"]
            cell["l0_change_difference"] = cell["l0_change_control"] - cell["l0_change_level0"]
            if abs(cell["adoption_difference"]) > 0.05 or abs(cell["l0_change_difference"]) > 0.003:
                flagged.append(key)
            cells[key] = cell
    out: dict[str, Any] = {"cells": cells, "missing": missing, "flagged": flagged}
    summary = read(ROOT / "diffregcd_interp" / "summary.json")
    if summary is not None and "DSIFN-CD" in summary:
        d = summary["DSIFN-CD"]
        gap, recovered = d["3"] - d["0"], d["0_interp"] - d["0"]
        out["diffregcd"] = {**summary, "dsifn_gap": gap, "dsifn_recovered": recovered,
                            "explains_anomaly": bool(gap > 0 and recovered >= gap / 2)}
    else:
        out["diffregcd"] = None
    return out


# ---------------------------------------------------------------------------------------------------- rule 6
def identity_block() -> dict[str, Any]:
    first, second = read(ROOT / "identity" / "identity.json"), read(ROOT / "identity_roma" / "identity" / "identity.json")
    if first is None or second is None:
        return {"missing": True}
    largest = 0.0
    for name, entry in first["datasets"].items():
        for key in ("identical_t1_px", "identical_t2_px", "known_level1_error_px"):
            for stat, value in entry[key].items():
                largest = max(largest, abs(value - second["datasets"][name][key][stat]))
    return {"missing": False, "module_repeat_max_difference": largest,
            "module_repeat_passed": largest <= 1e-4,
            "roma_identical_t1_px": {name: entry.get("roma_identical_t1_px") for name, entry in second["datasets"].items()},
            "module_identical_t1_px": {name: entry["identical_t1_px"] for name, entry in first["datasets"].items()}}


# ------------------------------------------------------------------------------------------------- rendering
def pp(value: float) -> str:
    return f"{value * 100:+.1f}"


def render_selector(title: str, block: dict[str, Any], verdict: str) -> list[str]:
    lines = [f"## {title}", ""]
    if block["cells"]:
        lines += ["| combination | L0 change | L3 gain | always L0 | always L3 | adopted L0 | adopted L3 | tau |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, c in block["cells"].items():
            lines.append(f"| {key} | {pp(c['l0'])} | {pp(c['l3'])} | {pp(c['always_l0'])} | {pp(c['always_l3'])} | "
                         f"{c['adopted_l0']:.3f} | {c['adopted_l3']:.3f} | {c['tau']} |")
        for label, m in (("this registrar", block["mean"]), ("paper, same combinations", block["paper_same_cells"])):
            lines.append(f"\n{label}: L0 mean {pp(m['l0_mean'])}, worst {pp(m['l0_worst'])}, degraded "
                         f"{m['degraded']}/{m['combinations']}, L3 gain {pp(m['l3_gain'])}")
    if block["missing"]:
        lines.append(f"\nmissing: {', '.join(block['missing'])}")
    lines += ["", f"**Reading:** {verdict}", ""]
    return lines


def render(report: dict[str, Any]) -> str:
    lines = ["# P33 read-out", "", f"Paper's selector over Table 6's twelve combinations (check against Table 6: "
             f"0.0 / -0.6 / 2 of 12 / +11.1): L0 mean {pp(report['paper_table6']['l0_mean'])}, worst "
             f"{pp(report['paper_table6']['l0_worst'])}, degraded {report['paper_table6']['degraded']}/12, "
             f"L3 gain {pp(report['paper_table6']['l3_gain'])}", ""]
    for localizer, title in (("roma_ts_ransac", "1. RoMa (3 DoF, RANSAC 3 px) + selector"),
                             ("phasecorr_fm", "1. Phase correlation (Fourier-Mellin) + selector")):
        lines += render_selector(title, report["selector"][localizer], report["rules"][localizer])
    for variant in RETRAINED:
        lines += render_selector(f"2. Retrained module {variant} + selector (DINOv3, BIT)",
                                 report["selector"][variant], report["rules"][variant])
    always = report["always"]
    lines += ["## 2, 5. Always registering (Table 7: success L3 % / L0 change / L3 gain) and pooled errors", "",
              "| variant | " + " | ".join(DATASETS) + " | L3 median px | L3 p90 px | L3 >25 px % | L0 >25 px % | "
              "median s (LEVIR) |", "|---|" + "---|" * (len(DATASETS) + 5)]
    for variant, entry in always["variants"].items():
        a, e = entry["ablation"], entry["errors"]
        cells = [f"{a[n]['success_l3']:.1f} / {a[n]['l0_change']:+.1f} / {a[n]['l3_gain']:+.1f}" if a else "-"
                 for n in DATASETS]
        if e:
            p = e["level3_pooled"]
            cells += [f"{p['median_eq10']:.2f}", f"{p['p90_eq10']:.1f}", f"{p['over25_eq10']:.1f}",
                      f"{e['level0_over25']['pooled']:.1f}"]
        else:
            cells += ["-"] * 4
        seconds = (entry["median_seconds"] or {}).get("LEVIR-CD")
        cells.append("-" if seconds is None else f"{seconds:.3f}")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines += ["", f"**Reading (5):** phase correlation: {report['rules']['phasecorr_fm_always']}; "
              f"ECC: {report['rules']['ecc_ts_always']}", ""]
    if always["missing"]:
        lines += [f"missing: {', '.join(always['missing'])}", ""]
    s = report["s2looking"]
    lines += ["## 3. S2Looking (never seen by the registration module)", ""]
    if s["missing"]:
        lines += ["missing", ""]
    else:
        ci = s["selector_minus_never"]["3"]
        lines += [f"{s['crops']} crops, {s['tiles']} tiles, tau {s['tau']}: L3 gain {pp(ci['value'])} "
                  f"[{pp(ci['low'])}, {pp(ci['high'])}]; L0 change {pp(s['l0'])}; adopted L0 {s['adopted_l0']:.3f}, "
                  f"L3 {s['adopted_l3']:.3f}; always registering L0 {pp(s['always_l0'])}, L3 {pp(s['always_l3'])}", ""]
    i = report["interp"]
    lines += ["## 4. Interpolation control", "", "| combination | adopted L0 -> control | L0 change -> control | anchor |",
              "|---|---|---|---|"]
    for key, c in i["cells"].items():
        lines.append(f"| {key} | {c['adopted_level0']:.3f} -> {c['adopted_control']:.3f} | "
                     f"{pp(c['l0_change_level0'])} -> {pp(c['l0_change_control'])} | {c['anchor_passed']} |")
    lines += ["", f"over the line (adoption > 5 points or L0 change > 0.3 points): {i['flagged'] or 'none'}"]
    if i["missing"]:
        lines.append(f"missing: {', '.join(i['missing'])}")
    d = i["diffregcd"]
    lines += ["", "DiffRegCD: missing" if d is None else
              f"DiffRegCD DSIFN-CD F1: L0 {d['DSIFN-CD']['0'] * 100:.1f}, control {d['DSIFN-CD']['0_interp'] * 100:.1f}, "
              f"L3 {d['DSIFN-CD']['3'] * 100:.1f}; gap {pp(d['dsifn_gap'])}, recovered {pp(d['dsifn_recovered'])}; "
              f"explains the anomaly: {d['explains_anomaly']}", ""]
    t = report["identity"]
    lines += ["## 6. Identical pairs", ""]
    if t["missing"]:
        lines += ["missing", ""]
    else:
        for name in DATASETS:
            r, m = t["roma_identical_t1_px"][name], t["module_identical_t1_px"][name]
            lines.append(f"- {name}: module median {m['median']:.3f} px (p90 {m['p90']:.3f}); RoMa median "
                         + ("-" if not r else f"{r['median']:.3f} px (p90 {r['p90']:.3f}, {r['crops']} crops)"))
        lines += [f"- module repeat equal to reports/p33/identity: {t['module_repeat_passed']} "
                  f"(max difference {t['module_repeat_max_difference']:.2e})", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(ROOT / "summary"))
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    paper = aggregate({f"{row}/{name}": changes(frozen_anchor(row, name)) for row in TABLE6_ROWS for name in DATASETS})
    selector = {localizer: selector_block(localizer, TABLE6_ROWS) for localizer in ("roma_ts_ransac", "phasecorr_fm")}
    selector.update({variant: selector_block(variant, ("dinov3", "bit")) for variant in RETRAINED})
    always = always_block()
    rules = {localizer: rule_1(selector[localizer], paper) for localizer in ("roma_ts_ransac", "phasecorr_fm")}
    rules.update({variant: rule_2(selector[variant]) for variant in RETRAINED})
    rules.update({f"{variant}_always": rule_5(always, variant) for variant in ("phasecorr_fm", "ecc_ts")})
    report = {"paper_table6": paper, "selector": selector, "always": always,
              "s2looking": s2looking_block(args.repeats, args.seed), "interp": interp_block(),
              "identity": identity_block(), "rules": rules}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    text = render(report)
    (output / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
