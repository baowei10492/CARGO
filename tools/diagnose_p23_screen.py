"""Read-only diagnosis of a finished P23 screen (main record 188).

Descriptive only: nothing here can change the preregistered screen verdict.
The screen scored every estimate against the injected shift. WHU's two
epochs already carry a multi-pixel intrinsic misregistration at native
resolution (main records 73, 121), which a precise localizer measures and a
6x-quantised one does not. This tool separates the two readings of a miss:

  offset_explained   the estimate moved by the injected amount relative to
                     the same localizer's clean estimate (a stable
                     tile-level offset, not a tracking failure)
  tracking_failure   it did not
  nonconverged       no consensus

and reports where the misses sit (change density, strip rows) and how the
clean offsets of RoMa and P7C compare. Input: run/screen_offsets.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Sequence

from rs_cd.change_density import DEFAULT_BINS, assign_bin

CLEAN = "clean:s0:common"
PHYSICAL = tuple(f"physical_shift:s{severity}:t2" for severity in (1, 2, 3))
ROW_GROUPS = (("rows 00-10", 0, 10), ("rows 11-21", 11, 21), ("rows 22-28", 22, 28))


def _f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _b(row: dict[str, Any], key: str) -> bool:
    value = row[key]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return bool(value)


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(1 for flag in flags if flag) / len(flags) if flags else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def classify(shifted: dict[str, Any], clean: dict[str, Any] | None, tolerance: float) -> dict[str, Any]:
    """Raw and differential error of one shifted estimate."""

    gt = (_f(shifted, "gt_dx"), _f(shifted, "gt_dy"))
    est = (_f(shifted, "est_dx"), _f(shifted, "est_dy"))
    raw = math.hypot(est[0] - gt[0], est[1] - gt[1])
    result: dict[str, Any] = {"raw_error": raw, "diff_error": None}
    if not _b(shifted, "converged"):
        result["kind"] = "nonconverged"
        return result
    if raw <= tolerance + 1e-6:
        result["kind"] = "tracked"
    if clean is not None and _b(clean, "converged"):
        base = (_f(clean, "est_dx"), _f(clean, "est_dy"))
        diff = math.hypot(est[0] - base[0] - gt[0], est[1] - base[1] - gt[1])
        result["diff_error"] = diff
        if "kind" not in result:
            result["kind"] = "offset_explained" if diff <= tolerance + 1e-6 else "tracking_failure"
    # Missed, and no converged clean estimate to tell an offset from a failure.
    result.setdefault("kind", "missed_without_clean_reference")
    return result


def residual_spread(rows: Sequence[dict[str, Any]]) -> float | None:
    """Largest pairwise distance between (estimate - injected) over converged rows."""

    residuals = [
        (_f(row, "est_dx") - _f(row, "gt_dx"), _f(row, "est_dy") - _f(row, "gt_dy"))
        for row in rows
        if _b(row, "converged")
    ]
    if len(residuals) < 2:
        return None
    return max(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for index, a in enumerate(residuals)
        for b in residuals[index + 1 :]
    )


def val_row(sample_id: str) -> int | None:
    match = re.fullmatch(r"whu_val_r(\d+)c\d+", sample_id)
    return int(match.group(1)) if match else None


def diagnose(rows: Sequence[dict[str, Any]], candidate: str, *, tolerance: float = 2.0) -> dict[str, Any]:
    own = [row for row in rows if row["candidate"] == candidate]
    clean_by_key = {(row["set"], row["id"]): row for row in own if row["scenario"] == CLEAN}
    report: dict[str, Any] = {"candidate": candidate, "tolerance_px": tolerance, "sets": {}}
    for set_name in sorted({row["set"] for row in own}):
        members = [row for row in own if row["set"] == set_name]
        clean_rows = [row for row in members if row["scenario"] == CLEAN]
        clean_magnitudes = [
            math.hypot(_f(row, "est_dx"), _f(row, "est_dy"))
            for row in clean_rows
            if _b(row, "converged")
        ]
        entry: dict[str, Any] = {
            "clean": {
                "n": len(clean_rows),
                "converged": sum(1 for row in clean_rows if _b(row, "converged")),
                "median_offset": _median(clean_magnitudes),
                "offset_over_2px": _rate([value > 2.0 for value in clean_magnitudes]),
            },
            "scenarios": {},
        }
        classified: list[dict[str, Any]] = []
        for scenario in sorted({row["scenario"] for row in members} - {CLEAN}):
            results = []
            for row in (r for r in members if r["scenario"] == scenario):
                outcome = classify(row, clean_by_key.get((set_name, row["id"])), tolerance)
                outcome.update(
                    id=row["id"],
                    scenario=scenario,
                    change_fraction=row.get("change_fraction"),
                )
                results.append(outcome)
            classified.extend(results)
            kinds = [item["kind"] for item in results]
            diffs = [item["diff_error"] for item in results if item["diff_error"] is not None]
            entry["scenarios"][scenario] = {
                "n": len(results),
                "raw_pck": _rate([item["raw_error"] <= tolerance + 1e-6 for item in results]),
                "diff_pck": _rate([value <= tolerance + 1e-6 for value in diffs]),
                "diff_n": len(diffs),
                "median_diff_error": _median(diffs),
                "kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
            }

        physical = [item for item in classified if item["scenario"] in PHYSICAL]
        by_tile: dict[str, list[dict[str, Any]]] = {}
        for row in members:
            if row["scenario"] in PHYSICAL:
                by_tile.setdefault(row["id"], []).append(row)
        spreads = {tile: residual_spread(tile_rows) for tile, tile_rows in by_tile.items()}
        entry["physical_tiles"] = {
            "tiles": len(by_tile),
            "stable_residual_tiles": sum(
                1 for tile, value in spreads.items()
                if value is not None and value <= tolerance and len(by_tile[tile]) == len(PHYSICAL)
                and all(_b(row, "converged") for row in by_tile[tile])
            ),
            "median_residual_spread": _median([v for v in spreads.values() if v is not None]),
        }

        if set_name == "val_tiles":
            entry["change_bins"] = []
            for index, (low, high) in enumerate(DEFAULT_BINS):
                items = [
                    item for item in physical
                    if item["change_fraction"] not in (None, "")
                    and assign_bin(float(item["change_fraction"]), DEFAULT_BINS) == index
                ]
                kinds = [item["kind"] for item in items]
                entry["change_bins"].append(
                    {
                        "range": [low, high],
                        "estimates": len(items),
                        "kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
                    }
                )
            entry["row_groups"] = []
            for label, first, last in ROW_GROUPS:
                items = [
                    item for item in physical
                    if (val_row(item["id"]) is not None and first <= val_row(item["id"]) <= last)
                ]
                kinds = [item["kind"] for item in items]
                entry["row_groups"].append(
                    {"rows": label, "estimates": len(items),
                     "kinds": {kind: kinds.count(kind) for kind in sorted(set(kinds))}}
                )
        report["sets"][set_name] = entry
    return report


def clean_offsets_versus_reference(
    rows: Sequence[dict[str, Any]], candidate: str, reference: str, *, tolerance: float = 3.0
) -> dict[str, Any]:
    """Validation clean pairs: does the candidate's offset match P7C's where the q gate fired?"""

    clean = {
        (row["candidate"], row["id"]): row
        for row in rows
        if row["set"] == "val_tiles" and row["scenario"] == CLEAN
    }
    fired = [
        row for (name, _), row in clean.items()
        if name == reference and _b(row, "trigger")
    ]
    agree = []
    for row in fired:
        other = clean.get((candidate, row["id"]))
        if other is None or not _b(other, "converged"):
            continue
        if (
            abs(_f(other, "est_dx") - _f(row, "est_dx")) <= tolerance
            and abs(_f(other, "est_dy") - _f(row, "est_dy")) <= tolerance
        ):
            agree.append(row["id"])
    return {
        "reference_triggered": len(fired),
        "candidate_agrees": len(agree),
        "tolerance_px": tolerance,
        "ids": agree,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen_dir", help="reports/p23_whu/screen/<tag>")
    parser.add_argument("--candidate", default="roma_outdoor")
    parser.add_argument("--reference", default="p7c")
    args = parser.parse_args()

    screen_dir = Path(args.screen_dir)
    with (screen_dir / "run" / "screen_offsets.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        "note": "descriptive, post hoc; does not change the preregistered screen verdict",
        "candidate": diagnose(rows, args.candidate),
        "reference": diagnose(rows, args.reference),
        "clean_offsets": clean_offsets_versus_reference(rows, args.candidate, args.reference),
    }
    # Post hoc: kept outside the preregistered screen directory.
    output = screen_dir.parent.parent / "diagnosis" / f"{screen_dir.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for key in ("candidate", "reference"):
        report = result[key]
        print(f"\n===== {report['candidate']} =====")
        for set_name, entry in report["sets"].items():
            clean = entry["clean"]
            print(f"[{set_name}] clean converged {clean['converged']}/{clean['n']}, "
                  f"median |offset| {clean['median_offset']}, share > 2 px {clean['offset_over_2px']}")
            for scenario, values in entry["scenarios"].items():
                print(f"  {scenario:22s} raw_pck2 {values['raw_pck']:.2f}  diff_pck2 "
                      f"{values['diff_pck'] if values['diff_pck'] is None else round(values['diff_pck'], 2)} "
                      f"(n={values['diff_n']})  kinds {values['kinds']}")
            tiles = entry["physical_tiles"]
            print(f"  physical tiles {tiles['tiles']}: stable residual {tiles['stable_residual_tiles']}, "
                  f"median spread {tiles['median_residual_spread']}")
            for item in entry.get("change_bins", []):
                print(f"  change {item['range'][0]:.2f}-{item['range'][1]:.2f}: {item['kinds']}")
            for item in entry.get("row_groups", []):
                print(f"  {item['rows']}: {item['kinds']}")
    clean = result["clean_offsets"]
    print(f"\nclean: P7C q fired on {clean['reference_triggered']} val tiles; "
          f"RoMa within {clean['tolerance_px']:g} px of P7C on {clean['candidate_agrees']}")
    print(f"written {output}")


if __name__ == "__main__":
    main()
