from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_REPORTS = (
    "p12_final_test",
    "p11_structure_freeze",
    "p11_second_detector_seed",
    "p10_efficiency",
    "p9b_perturbation_robustness",
)

OPTIONAL_REPORTS = (
    "p9_classical_baselines_levir",
    "p7_safe_alignment_classifier_levir",
    "p7_dual_path_oracle_levir",
    "p7_stage_c_s2looking_transfer",
    "p8_whu_p3b",
    "p8_whu_localizer",
    "p8_scale_probe",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_latest(project_root: Path, name: str) -> Path | None:
    pointer = project_root / "reports" / name / "LATEST"
    if not pointer.is_file():
        return None
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Empty LATEST pointer: {pointer}")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"LATEST for {name} does not resolve to a directory: {candidate}"
        )
    return candidate


def paper_main_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if summary.get("status") != "completed":
        raise ValueError("P12 final summary is not completed")
    if summary.get("test_result_used_for_model_selection") is not False:
        raise ValueError("P12 result must not have been used for selection")
    clean = summary["clean"]
    rows: list[dict[str, Any]] = [
        {
            "scenario": "Clean",
            "shift_pixels": 0,
            "raw_f1": float(clean["raw_f1"]),
            "primary_f1": float(clean["primary_f1"]),
            "gain": float(clean["primary_minus_raw"]),
        }
    ]
    expected = (
        ("shift:s1:t2", 2),
        ("shift:s2:t2", 4),
        ("shift:s3:t2", 8),
    )
    for key, pixels in expected:
        item = summary["shift"][key]
        rows.append(
            {
                "scenario": f"Shift-{pixels}",
                "shift_pixels": pixels,
                "raw_f1": float(item["raw_f1"]),
                "primary_f1": float(item["primary_f1"]),
                "gain": float(item["primary_minus_raw"]),
            }
        )
    calculated = sum(row["gain"] for row in rows[1:]) / 3.0
    reported = float(summary["mean_shift_gain"])
    if abs(calculated - reported) > 1e-10:
        raise ValueError(
            f"Mean shift gain mismatch: calculated={calculated}, "
            f"reported={reported}"
        )
    return rows


def render_main_table_markdown(
    rows: list[dict[str, Any]],
    *,
    primary_method: str,
) -> str:
    lines = [
        "| Scenario | Raw F1 | Primary F1 | Gain (pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {raw:.4f} | {primary:.4f} | {gain:+.2f} |".format(
                scenario=row["scenario"],
                raw=row["raw_f1"],
                primary=row["primary_f1"],
                gain=100.0 * row["gain"],
            )
        )
    mean_gain = sum(float(row["gain"]) for row in rows[1:]) / 3.0
    lines.extend(
        [
            "",
            f"Primary method: `{primary_method}`.",
            f"Mean shift gain: **{100.0 * mean_gain:+.2f} pp**.",
            "The final test was reporting-only and was not used for model "
            "selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_main_figure_svg(
    rows: list[dict[str, Any]],
    *,
    primary_label: str,
) -> str:
    width, height = 980, 560
    left, right, top, bottom = 92, 34, 54, 102
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.60, 0.92

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = {"raw": "#8892A0", "primary": "#1877B9"}
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
            "One-shot LEVIR Test: Misregistration Robustness</text>"
        ),
    ]
    for tick in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
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
            (-bar_w / 2, "raw_f1", colors["raw"]),
            (bar_w / 2, "primary_f1", colors["primary"]),
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
            f'text-anchor="middle">{html.escape(str(row["scenario"]))}</text>'
        )
        if index > 0:
            parts.append(
                f'<text class="small" x="{center:.1f}" '
                f'y="{height-bottom+50}" text-anchor="middle" '
                f'fill="#0B6E4F">+{100*float(row["gain"]):.2f} pp</text>'
            )
    legend_y = height - 28
    parts.extend(
        [
            (
                f'<rect x="300" y="{legend_y-13}" width="18" height="18" '
                f'fill="{colors["raw"]}"/>'
            ),
            (
                f'<text class="small" x="326" y="{legend_y+1}">'
                "Raw detector</text>"
            ),
            (
                f'<rect x="500" y="{legend_y-13}" width="18" height="18" '
                f'fill="{colors["primary"]}"/>'
            ),
            (
                f'<text class="small" x="526" y="{legend_y+1}">'
                f"{html.escape(primary_label)}</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _write_main_assets(
    output: Path,
    p12_summary: dict[str, Any],
) -> None:
    assets = output / "paper_assets"
    assets.mkdir(parents=True)
    rows = paper_main_rows(p12_summary)
    primary = str(p12_summary["primary_method"])
    with (assets / "table_main_final_test.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (assets / "table_main_final_test.md").write_text(
        render_main_table_markdown(rows, primary_method=primary),
        encoding="utf-8",
    )
    (assets / "figure_main_final_test.svg").write_text(
        render_main_figure_svg(rows, primary_label="Safe-veto task"),
        encoding="utf-8",
    )


def _copy_report_tree(
    source: Path,
    destination: Path,
    *,
    include_checkpoints: bool,
) -> None:
    ignored_suffixes = {".pth", ".pt", ".ckpt"}

    def ignore(_directory: str, names: list[str]) -> set[str]:
        if include_checkpoints:
            return set()
        return {
            name
            for name in names
            if Path(name).suffix.lower() in ignored_suffixes
        }

    shutil.copytree(source, destination, ignore=ignore)


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


def build_package(
    project_root: Path,
    output: Path,
    *,
    include_checkpoints: bool,
) -> Path:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    resolved: dict[str, Path] = {}
    for name in (*REQUIRED_REPORTS, *OPTIONAL_REPORTS):
        path = _resolve_latest(project_root, name)
        if path is None:
            if name in REQUIRED_REPORTS:
                raise FileNotFoundError(
                    f"Required report pointer is missing: reports/{name}/LATEST"
                )
            continue
        resolved[name] = path

    p12_summary_path = (
        resolved["p12_final_test"] / "levir" / "final_test_summary.json"
    )
    if not p12_summary_path.is_file():
        raise FileNotFoundError(f"Missing P12 summary: {p12_summary_path}")
    p12_summary = _read_json(p12_summary_path)
    _write_main_assets(output, p12_summary)

    reports_out = output / "evidence"
    reports_out.mkdir()
    for name, source in resolved.items():
        _copy_report_tree(
            source,
            reports_out / name,
            include_checkpoints=include_checkpoints,
        )

    sources = {name: str(path) for name, path in resolved.items()}
    readme = [
        "# Final paper evidence package",
        "",
        "This package was assembled after the frozen, one-shot P12 test.",
        "It is reporting-only and must not be used to retune the method.",
        "",
        "## Primary result",
        "",
        (output / "paper_assets" / "table_main_final_test.md")
        .read_text(encoding="utf-8")
        .strip(),
        "",
        "## Dataset policy",
        "",
        "- LEVIR test was consumed exactly once.",
        "- S2Looking labeled test remains sealed after the applicability gate.",
        "- No checkpoint or threshold may be selected from P12 results.",
        "",
        f"Checkpoint files included: `{include_checkpoints}`.",
        "Frozen model paths and hashes remain available in the copied "
        "P11 structure-freeze evidence.",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    manifest = _manifest(output, sources)
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble frozen experiment evidence and generate the main "
            "paper table/figure without rerunning any model."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Copy checkpoint files found inside report directories.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also create OUTPUT_DIR.tar.gz.",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = project_root / output
    output = output.resolve()
    build_package(
        project_root,
        output,
        include_checkpoints=bool(args.include_checkpoints),
    )
    if args.archive:
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output, arcname=output.name)
        print(f"Archive: {archive}")
        print(f"Archive SHA256: {_sha256(archive)}")
    print(f"Evidence package: {output}")


if __name__ == "__main__":
    main()
