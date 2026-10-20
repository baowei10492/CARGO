from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.build_final_paper_evidence import (
    _copy_report_tree,
    _resolve_latest,
    _sha256,
    paper_main_rows,
    render_main_figure_svg,
    render_main_table_markdown,
)
from tools.summarize_p13_external_comparison import comparison_summary


REQUIRED_REPORTS = (
    "p12_final_test",
    "p13_diffregcd_validation",
    "p13_diffregcd_external_freeze",
    "p13_diffregcd_final_test",
)

PROTOCOL_FILES = (
    "docs/P12_FINAL_TEST_PROTOCOL.md",
    "docs/P13_DIFFREGCD_EXTERNAL_BASELINE_PROTOCOL.md",
    "docs/P13_DIFFREGCD_EXECUTION.md",
    "docs/P13_DIFFREGCD_FINAL_TEST_PROTOCOL.md",
    "configs/p12_levir_final_test.yaml",
    "configs/p13_diffregcd_validation.yaml",
    "configs/p13_diffregcd_final_test.yaml",
)

SCENARIO_LABELS = {
    "clean:s0:common": "Clean",
    "shift:s1:t2": "Shift-2",
    "shift:s2:t2": "Shift-4",
    "shift:s3:t2": "Shift-8",
}

EXPECTED_EXTERNAL_COMMIT = "6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38"

# Numbers already recorded and frozen in the main experiment log
# (sections 102 / 105 / 107). This packaging tool is only allowed to
# reproduce them; if it cannot, it refuses to emit a package instead of
# publishing a second, silently different set of numbers.
RECORDED_PROJECT_F1 = {
    "clean:s0:common": 0.900673,
    "shift:s1:t2": 0.897089,
    "shift:s2:t2": 0.868934,
    "shift:s3:t2": 0.819015,
}
RECORDED_EXTERNAL_F1 = {
    "clean:s0:common": 0.870729,
    "shift:s1:t2": 0.795491,
    "shift:s2:t2": 0.613877,
    "shift:s3:t2": 0.621907,
}
RECORDED_MEAN_SHIFT_ADVANTAGE = 0.184588
RECORDED_VALID_PIXELS = 125960192
RECORDED_VALIDATION_CHANGE_F1 = 0.872024
RECORDED_TOLERANCE = 1e-6


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(passed: bool, **fields: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **fields}


def _as_int(value: Any, *, default: int = -1) -> int:
    """Coerce a marker field to int so a malformed file fails the gate.

    A missing or non-numeric field must be reported as a failed gate,
    never as a traceback that hides which check went wrong.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def reproduction_gate(
    summary: dict[str, Any],
    *,
    tolerance: float = RECORDED_TOLERANCE,
) -> dict[str, Any]:
    """Check the comparison against the frozen recorded numbers.

    The recorded values are rounded to six decimals, so the tolerance is
    an agreement check on the published table, not a metric threshold.
    """
    checks: dict[str, Any] = {}
    for row in summary["scenario_comparison"]:
        scenario = str(row["scenario"])
        expected_project = RECORDED_PROJECT_F1[scenario]
        expected_external = RECORDED_EXTERNAL_F1[scenario]
        project_f1 = float(row["project_f1"])
        external_f1 = float(row["external_f1"])
        checks[scenario] = {
            "project_f1": project_f1,
            "recorded_project_f1": expected_project,
            "project_absolute_delta": abs(project_f1 - expected_project),
            "external_f1": external_f1,
            "recorded_external_f1": expected_external,
            "external_absolute_delta": abs(
                external_f1 - expected_external
            ),
            "project_valid_pixels": int(row["project_valid_pixels"]),
            "external_valid_pixels": int(row["external_valid_pixels"]),
            "recorded_valid_pixels": RECORDED_VALID_PIXELS,
            "passed": (
                abs(project_f1 - expected_project) <= tolerance
                and abs(external_f1 - expected_external) <= tolerance
                and int(row["project_valid_pixels"])
                == RECORDED_VALID_PIXELS
                and int(row["external_valid_pixels"])
                == RECORDED_VALID_PIXELS
                and bool(row["same_valid_pixel_count"])
            ),
        }
    mean_advantage = float(
        summary["mean_project_minus_external_shift_f1"]
    )
    mean_check = _gate(
        abs(mean_advantage - RECORDED_MEAN_SHIFT_ADVANTAGE) <= tolerance,
        mean_project_minus_external_shift_f1=mean_advantage,
        recorded_mean_project_minus_external_shift_f1=(
            RECORDED_MEAN_SHIFT_ADVANTAGE
        ),
        absolute_delta=abs(
            mean_advantage - RECORDED_MEAN_SHIFT_ADVANTAGE
        ),
    )
    status_check = _gate(
        summary.get("status") == "completed"
        and bool(summary.get("same_valid_pixel_count_all_scenarios"))
        and summary.get("test_result_used_for_model_selection") is False,
        status=summary.get("status"),
        same_valid_pixel_count_all_scenarios=summary.get(
            "same_valid_pixel_count_all_scenarios"
        ),
        test_result_used_for_model_selection=summary.get(
            "test_result_used_for_model_selection"
        ),
    )
    passed = (
        all(record["passed"] for record in checks.values())
        and mean_check["passed"]
        and status_check["passed"]
    )
    return {
        "passed": passed,
        "tolerance": tolerance,
        "scenarios": checks,
        "mean_shift_advantage": mean_check,
        "comparison_status": status_check,
    }


def marker_gate(
    p12_marker: dict[str, Any],
    p13_marker: dict[str, Any],
) -> dict[str, Any]:
    p12 = _gate(
        p12_marker.get("status") == "completed"
        and _as_int(p12_marker.get("allowed_reruns")) == 0,
        status=p12_marker.get("status"),
        allowed_reruns=p12_marker.get("allowed_reruns"),
    )
    p13 = _gate(
        p13_marker.get("status") == "completed"
        and _as_int(p13_marker.get("allowed_reruns")) == 0
        and p13_marker.get("reporting_only_no_selection") is True
        and p13_marker.get("inference_path") == "official_unmodified",
        status=p13_marker.get("status"),
        allowed_reruns=p13_marker.get("allowed_reruns"),
        reporting_only_no_selection=p13_marker.get(
            "reporting_only_no_selection"
        ),
        inference_path=p13_marker.get("inference_path"),
    )
    return {
        "passed": p12["passed"] and p13["passed"],
        "p12_levir_test_consumed": p12,
        "p13_diffregcd_levir_test_consumed": p13,
    }


def freeze_gate(freeze_manifest: dict[str, Any]) -> dict[str, Any]:
    return _gate(
        freeze_manifest.get("status") == "frozen_external_test"
        and freeze_manifest.get("selected_external_path")
        == "official_unmodified"
        and freeze_manifest.get("external_commit")
        == EXPECTED_EXTERNAL_COMMIT
        and bool(freeze_manifest.get("test_authorized", False)),
        status=freeze_manifest.get("status"),
        selected_external_path=freeze_manifest.get(
            "selected_external_path"
        ),
        external_commit=freeze_manifest.get("external_commit"),
        expected_external_commit=EXPECTED_EXTERNAL_COMMIT,
        selection_basis=freeze_manifest.get("selection_basis"),
        test_authorized=freeze_manifest.get("test_authorized"),
    )


def validation_gate(
    validation_results: dict[str, Any],
    *,
    tolerance: float = RECORDED_TOLERANCE,
) -> dict[str, Any]:
    official = (
        validation_results.get("clean_reproduction_audit", {})
        .get("paths", {})
        .get("official_unmodified", {})
    )
    record = official.get("change_f1")
    observed = (
        float(record["observed"])
        if isinstance(record, dict) and "observed" in record
        else None
    )
    author_delta = (
        float(record["delta"])
        if isinstance(record, dict) and "delta" in record
        else None
    )
    reproduced = (
        observed is not None
        and abs(observed - RECORDED_VALIDATION_CHANGE_F1) <= tolerance
    )
    return _gate(
        reproduced
        and validation_results.get("split") == "val"
        and bool(validation_results.get("test_consumed", True)) is False
        and validation_results.get("external_commit")
        == EXPECTED_EXTERNAL_COMMIT,
        split=validation_results.get("split"),
        test_consumed=validation_results.get("test_consumed"),
        external_commit=validation_results.get("external_commit"),
        official_clean_change_f1=observed,
        recorded_clean_change_f1=RECORDED_VALIDATION_CHANGE_F1,
        author_reported_delta=author_delta,
    )


def external_comparison_rows(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary["scenario_comparison"]:
        scenario = str(row["scenario"])
        rows.append(
            {
                "scenario": SCENARIO_LABELS[scenario],
                "scenario_key": scenario,
                "project_f1": float(row["project_f1"]),
                "external_f1": float(row["external_f1"]),
                "project_minus_external": float(
                    row["project_minus_external_f1"]
                ),
                "valid_pixels": int(row["project_valid_pixels"]),
            }
        )
    return rows


def render_external_table_markdown(
    rows: list[dict[str, Any]],
    *,
    project_method: str,
    external_method: str,
) -> str:
    lines = [
        (
            "| Scenario | Ours (`{project}`) | DiffRegCD (`{external}`) "
            "| Ours − DiffRegCD (pp) |"
        ).format(project=project_method, external=external_method),
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {project:.6f} | {external:.6f} "
            "| {delta:+.2f} |".format(
                scenario=row["scenario"],
                project=row["project_f1"],
                external=row["external_f1"],
                delta=100.0 * row["project_minus_external"],
            )
        )
    shifts = rows[1:]
    mean_advantage = sum(
        float(row["project_minus_external"]) for row in shifts
    ) / len(shifts)
    supports = {int(row["valid_pixels"]) for row in rows}
    lines.extend(
        [
            "",
            f"Mean shift advantage: **{100.0 * mean_advantage:+.2f} pp**.",
            (
                "Valid pixels per scenario (identical for both methods): "
                f"{sorted(supports)[0]:,}."
            ),
            (
                "DiffRegCD runs the authors' public `official_unmodified` "
                "path at the frozen upstream commit; the test was "
                "reporting-only and selected nothing."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def render_external_figure_svg(
    rows: list[dict[str, Any]],
    *,
    project_label: str = "Ours (safe-veto task)",
    external_label: str = "DiffRegCD (official)",
) -> str:
    width, height = 980, 560
    left, right, top, bottom = 92, 34, 54, 102
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.55, 0.95

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = {"project": "#1877B9", "external": "#C2703D"}
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#172033}"
        ".axis{stroke:#5F6B7A;stroke-width:1}"
        ".grid{stroke:#DDE2E8;stroke-width:1}"
        ".label{font-size:16px}.small{font-size:14px}"
        ".title{font-size:22px;font-weight:700}"
        "</style>",
        (
            '<text class="title" x="490" y="30" text-anchor="middle">'
            "One-shot LEVIR Test: Ours vs DiffRegCD (WACV 2026)</text>"
        ),
    ]
    for tick in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        yy = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" '
            f'x2="{width-right}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text class="small" x="{left-12}" y="{yy+5:.1f}" '
            f'text-anchor="end">{tick:.2f}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top}" '
        f'x2="{left}" y2="{height-bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{height-bottom}" '
        f'x2="{width-right}" y2="{height-bottom}"/>'
    )
    group_w = plot_w / len(rows)
    bar_w = 58
    baseline_y = y(y_min)
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        for offset, key, color in (
            (-bar_w / 2, "project_f1", colors["project"]),
            (bar_w / 2, "external_f1", colors["external"]),
        ):
            value = float(row[key])
            xx = center + offset - bar_w / 2
            yy = y(value)
            parts.append(
                f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w}" '
                f'height="{baseline_y-yy:.1f}" rx="3" fill="{color}"/>'
            )
            parts.append(
                f'<text class="small" x="{xx+bar_w/2:.1f}" '
                f'y="{yy-7:.1f}" text-anchor="middle">{value:.3f}</text>'
            )
        parts.append(
            f'<text class="label" x="{center:.1f}" y="{height-bottom+29}" '
            f'text-anchor="middle">'
            f'{html.escape(str(row["scenario"]))}</text>'
        )
        parts.append(
            f'<text class="small" x="{center:.1f}" '
            f'y="{height-bottom+50}" text-anchor="middle" '
            f'fill="#0B6E4F">'
            f'+{100*float(row["project_minus_external"]):.2f} pp</text>'
        )
    legend_y = height - 28
    parts.extend(
        [
            (
                f'<rect x="228" y="{legend_y-13}" width="18" height="18" '
                f'fill="{colors["project"]}"/>'
            ),
            (
                f'<text class="small" x="254" y="{legend_y+1}">'
                f"{html.escape(project_label)}</text>"
            ),
            (
                f'<rect x="558" y="{legend_y-13}" width="18" height="18" '
                f'fill="{colors["external"]}"/>'
            ),
            (
                f'<text class="small" x="584" y="{legend_y+1}">'
                f"{html.escape(external_label)}</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _manifest(output: Path, sources: dict[str, str]) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "files": files,
    }


def _render_readme(
    verification: dict[str, Any],
    external_table: str,
    main_table: str,
) -> str:
    lines = [
        "# P13 external-baseline evidence package",
        "",
        "Assembled after the frozen one-shot LEVIR tests of P12 (this "
        "method) and P13 (DiffRegCD, WACV 2026).",
        "This package is reporting-only: it reruns no model, reads no "
        "test image, and must not be used to retune anything.",
        "",
        "## Same-support external comparison",
        "",
        external_table.strip(),
        "",
        "## Internal Raw vs Primary result (P12)",
        "",
        main_table.strip(),
        "",
        "## Verification gates",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for name in (
        "markers",
        "external_freeze",
        "external_validation",
        "recorded_number_reproduction",
        "recomputed_comparison_matches_recorded",
    ):
        lines.append(
            f"| `{name}` | {bool(verification[name]['passed'])} |"
        )
    lines.extend(
        [
            "",
            "`recorded_number_reproduction` re-derives the published "
            "table from the raw per-scenario TP/FP/FN/TN in "
            "`results.json` and refuses to package anything that "
            "disagrees with the frozen log.",
            "",
            "## Dataset policy",
            "",
            "- LEVIR test was consumed exactly once by P12 and once by "
            "P13; both markers carry `allowed_reruns = 0`.",
            "- S2Looking labeled test remains sealed after its "
            "unlabeled applicability gate failed.",
            "- Neither test result was used for model, path, or "
            "threshold selection.",
            "",
            "## Known limits that must travel with these numbers",
            "",
            "- The advantage is demonstrated for controlled global "
            "translation; it does not extrapolate to rotation, "
            "perspective, or local deformation.",
            "- P9-B: the preregistered Task-veto failed the no-harm gate "
            "at 1.0 px subpixel and 0.5 deg rotation.",
            "- The backbones differ, so the margin over DiffRegCD cannot "
            "be attributed to the router alone.",
            "",
        ]
    )
    return "\n".join(lines)


def build_package(
    project_root: Path,
    output: Path,
    *,
    include_checkpoints: bool = False,
    tolerance: float = RECORDED_TOLERANCE,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    resolved: dict[str, Path] = {}
    for name in REQUIRED_REPORTS:
        path = _resolve_latest(project_root, name)
        if path is None:
            raise FileNotFoundError(
                f"Required report pointer is missing: reports/{name}/LATEST"
            )
        resolved[name] = path

    p12_root = resolved["p12_final_test"]
    p13_root = resolved["p13_diffregcd_final_test"]
    freeze_root = resolved["p13_diffregcd_external_freeze"]
    validation_root = resolved["p13_diffregcd_validation"]

    p12_summary = _read_json(p12_root / "levir" / "final_test_summary.json")
    p12_results = _read_json(p12_root / "levir" / "results.json")
    external_results = _read_json(p13_root / "levir" / "results.json")
    external_summary = _read_json(
        p13_root / "levir" / "final_test_summary.json"
    )
    recorded_comparison = _read_json(p13_root / "p12_vs_diffregcd.json")
    freeze_manifest = _read_json(freeze_root / "freeze_manifest.json")
    validation_results = _read_json(
        validation_root / "validation" / "results.json"
    )
    p12_marker = _read_json(
        project_root / "reports" / "p12_final_test"
        / "LEVIR_TEST_CONSUMED.json"
    )
    p13_marker = _read_json(
        project_root / "reports" / "p13_diffregcd_final_test"
        / "DIFFREGCD_LEVIR_TEST_CONSUMED.json"
    )

    recomputed = comparison_summary(
        p12_results["scenario_results"],
        external_results["scenario_results"],
        project_method=str(recorded_comparison["project_method"]),
        external_method=str(recorded_comparison["external_method"]),
    )
    fresh_rows = recomputed["scenario_comparison"]
    stored_rows = recorded_comparison["scenario_comparison"]
    same_length = len(fresh_rows) == len(stored_rows)
    recompute_diffs = [
        {
            "scenario": str(fresh["scenario"]),
            "project_absolute_delta": abs(
                float(fresh["project_f1"]) - float(stored["project_f1"])
            ),
            "external_absolute_delta": abs(
                float(fresh["external_f1"]) - float(stored["external_f1"])
            ),
            "same_scenario": str(fresh["scenario"])
            == str(stored["scenario"]),
        }
        for fresh, stored in zip(fresh_rows, stored_rows)
    ]
    recompute_gate = _gate(
        same_length
        and all(
            item["project_absolute_delta"] <= 1e-12
            and item["external_absolute_delta"] <= 1e-12
            and item["same_scenario"]
            for item in recompute_diffs
        )
        and abs(
            float(recomputed["mean_project_minus_external_shift_f1"])
            - float(
                recorded_comparison[
                    "mean_project_minus_external_shift_f1"
                ]
            )
        )
        <= 1e-12,
        scenario_count_matches=same_length,
        scenarios=recompute_diffs,
    )

    verification = {
        "status": "unverified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_only_no_selection": True,
        "markers": marker_gate(p12_marker, p13_marker),
        "external_freeze": freeze_gate(freeze_manifest),
        "external_validation": validation_gate(
            validation_results, tolerance=tolerance
        ),
        "recorded_number_reproduction": reproduction_gate(
            recomputed, tolerance=tolerance
        ),
        "recomputed_comparison_matches_recorded": recompute_gate,
        "external_summary_status": external_summary.get("status"),
        "s2looking_labeled_test_sealed_note_present": (
            p12_root / "S2LOOKING_TEST_NOT_CONSUMED.txt"
        ).is_file(),
    }
    failed = [
        name
        for name in (
            "markers",
            "external_freeze",
            "external_validation",
            "recorded_number_reproduction",
            "recomputed_comparison_matches_recorded",
        )
        if not verification[name]["passed"]
    ]
    if failed:
        raise ValueError(
            "P13 evidence verification failed: "
            + ", ".join(failed)
            + "; refusing to write a package that disagrees with the "
            "frozen record"
        )
    verification["status"] = "verified"

    output.mkdir(parents=True)
    assets = output / "paper_assets"
    assets.mkdir()

    main_rows = paper_main_rows(p12_summary)
    main_table = render_main_table_markdown(
        main_rows, primary_method=str(p12_summary["primary_method"])
    )
    _write_csv(assets / "table_main_final_test.csv", main_rows)
    (assets / "table_main_final_test.md").write_text(
        main_table, encoding="utf-8"
    )
    (assets / "figure_main_final_test.svg").write_text(
        render_main_figure_svg(main_rows, primary_label="Safe-veto task"),
        encoding="utf-8",
    )

    external_rows = external_comparison_rows(recomputed)
    external_table = render_external_table_markdown(
        external_rows,
        project_method=str(recomputed["project_method"]),
        external_method=str(recomputed["external_method"]),
    )
    _write_csv(assets / "table_external_comparison.csv", external_rows)
    (assets / "table_external_comparison.md").write_text(
        external_table, encoding="utf-8"
    )
    (assets / "figure_external_comparison.svg").write_text(
        render_external_figure_svg(external_rows), encoding="utf-8"
    )

    evidence = output / "evidence"
    evidence.mkdir()
    for name, source in resolved.items():
        _copy_report_tree(
            source,
            evidence / name,
            include_checkpoints=include_checkpoints,
        )
    for marker_name, marker_path in (
        (
            "LEVIR_TEST_CONSUMED.json",
            project_root / "reports" / "p12_final_test"
            / "LEVIR_TEST_CONSUMED.json",
        ),
        (
            "DIFFREGCD_LEVIR_TEST_CONSUMED.json",
            project_root / "reports" / "p13_diffregcd_final_test"
            / "DIFFREGCD_LEVIR_TEST_CONSUMED.json",
        ),
    ):
        shutil.copy2(marker_path, evidence / marker_name)

    protocol = output / "protocol"
    protocol.mkdir()
    missing_protocol: list[str] = []
    for relative in PROTOCOL_FILES:
        source = project_root / relative
        if not source.is_file():
            missing_protocol.append(relative)
            continue
        shutil.copy2(source, protocol / Path(relative).name)
    verification["missing_protocol_files"] = missing_protocol

    (output / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _render_readme(verification, external_table, main_table),
        encoding="utf-8",
    )
    sources = {name: str(path) for name, path in resolved.items()}
    (output / "evidence_manifest.json").write_text(
        json.dumps(
            _manifest(output, sources), indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Package the frozen P12/P13 evidence and regenerate the "
            "external-comparison table and figure without rerunning any "
            "model or reading any test image."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--archive-path",
        default=None,
        help=(
            "Write a tar.gz at this path (default: OUTPUT_DIR.tar.gz). "
            "Use --no-archive to skip."
        ),
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Only build the directory.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Copy checkpoint files found inside report directories.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()

    verification = build_package(
        project_root,
        output,
        include_checkpoints=bool(args.include_checkpoints),
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))
    print(f"Evidence package: {output}")

    if args.no_archive:
        return
    archive = (
        Path(args.archive_path)
        if args.archive_path
        else output.with_suffix(".tar.gz")
    )
    if not archive.is_absolute():
        archive = project_root / archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output, arcname=output.name)
    digest = _sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    print(f"Archive: {archive}")
    print(f"Archive SHA256: {digest}")


if __name__ == "__main__":
    main()
