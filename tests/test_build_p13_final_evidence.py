import json
from pathlib import Path

import pytest

from tools.build_p13_final_evidence import (
    RECORDED_EXTERNAL_F1,
    RECORDED_PROJECT_F1,
    RECORDED_VALID_PIXELS,
    build_package,
    external_comparison_rows,
    freeze_gate,
    marker_gate,
    render_external_figure_svg,
    render_external_table_markdown,
    reproduction_gate,
    validation_gate,
)
from tools.summarize_p13_external_comparison import comparison_summary


SCENARIOS = (
    "clean:s0:common",
    "shift:s1:t2",
    "shift:s2:t2",
    "shift:s3:t2",
)

RAW_F1 = {
    "clean:s0:common": 0.900673,
    "shift:s1:t2": 0.863712,
    "shift:s2:t2": 0.796434,
    "shift:s3:t2": 0.663834,
}


def _support_row(**fields: object) -> dict[str, object]:
    return {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": RECORDED_VALID_PIXELS - 3,
        **fields,
    }


def _project_rows(f1: dict[str, float]) -> list[dict[str, object]]:
    return [
        _support_row(
            scenario=scenario,
            method="safe_veto_task",
            f1=f1[scenario],
        )
        for scenario in SCENARIOS
    ]


def _external_rows(f1: dict[str, float]) -> list[dict[str, object]]:
    return [
        _support_row(
            scenario=scenario,
            method="official_unmodified",
            change_f1=f1[scenario],
        )
        for scenario in SCENARIOS
    ]


def _summary(
    project_f1: dict[str, float] | None = None,
    external_f1: dict[str, float] | None = None,
) -> dict[str, object]:
    return comparison_summary(
        _project_rows(project_f1 or RECORDED_PROJECT_F1),
        _external_rows(external_f1 or RECORDED_EXTERNAL_F1),
        project_method="safe_veto_task",
        external_method="official_unmodified",
    )


def _p12_summary() -> dict[str, object]:
    gains = {
        scenario: RECORDED_PROJECT_F1[scenario] - RAW_F1[scenario]
        for scenario in SCENARIOS
    }
    return {
        "status": "completed",
        "primary_method": "safe_veto_task",
        "clean": {
            "raw_f1": RAW_F1["clean:s0:common"],
            "primary_f1": RECORDED_PROJECT_F1["clean:s0:common"],
            "primary_minus_raw": gains["clean:s0:common"],
        },
        "shift": {
            scenario: {
                "raw_f1": RAW_F1[scenario],
                "primary_f1": RECORDED_PROJECT_F1[scenario],
                "primary_minus_raw": gains[scenario],
            }
            for scenario in SCENARIOS[1:]
        },
        "mean_shift_gain": sum(
            gains[scenario] for scenario in SCENARIOS[1:]
        )
        / 3.0,
        "test_result_used_for_model_selection": False,
    }


def _freeze_manifest() -> dict[str, object]:
    return {
        "status": "frozen_external_test",
        "selected_external_path": "official_unmodified",
        "external_commit": "6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38",
        "selection_basis": (
            "public_author_test_wiring_and_clean_reproduction_only"
        ),
        "test_authorized": True,
    }


def _validation_results() -> dict[str, object]:
    return {
        "split": "val",
        "test_consumed": False,
        "external_commit": "6b9acb6c4476693916b00bf4a2e2eb37a5d0ed38",
        "clean_reproduction_audit": {
            "paths": {
                "official_unmodified": {
                    "change_f1": {
                        "observed": 0.872024,
                        "author_reference": 0.872,
                        "delta": 0.000024,
                    }
                }
            }
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _project_tree(
    root: Path,
    *,
    project_f1: dict[str, float] | None = None,
) -> Path:
    project_f1 = project_f1 or RECORDED_PROJECT_F1
    p12 = root / "reports" / "p12_final_test"
    p12_run = p12 / "20260727T055925Z"
    _write_json(
        p12_run / "levir" / "final_test_summary.json", _p12_summary()
    )
    _write_json(
        p12_run / "levir" / "results.json",
        {"scenario_results": _project_rows(project_f1)},
    )
    (p12_run / "S2LOOKING_TEST_NOT_CONSUMED.txt").write_text(
        "sealed\n", encoding="utf-8"
    )
    (p12 / "LATEST").write_text(
        "reports/p12_final_test/20260727T055925Z\n", encoding="utf-8"
    )
    _write_json(
        p12 / "LEVIR_TEST_CONSUMED.json",
        {"status": "completed", "allowed_reruns": 0},
    )

    p13 = root / "reports" / "p13_diffregcd_final_test"
    p13_run = p13 / "20260727T083624Z"
    _write_json(
        p13_run / "levir" / "results.json",
        {"scenario_results": _external_rows(RECORDED_EXTERNAL_F1)},
    )
    _write_json(
        p13_run / "levir" / "final_test_summary.json",
        {
            "status": "completed",
            "external_method": "DiffRegCD",
            "inference_path": "official_unmodified",
        },
    )
    _write_json(
        p13_run / "p12_vs_diffregcd.json",
        comparison_summary(
            _project_rows(project_f1),
            _external_rows(RECORDED_EXTERNAL_F1),
            project_method="safe_veto_task",
            external_method="official_unmodified",
        ),
    )
    (p13 / "LATEST").write_text(
        "reports/p13_diffregcd_final_test/20260727T083624Z\n",
        encoding="utf-8",
    )
    _write_json(
        p13 / "DIFFREGCD_LEVIR_TEST_CONSUMED.json",
        {
            "status": "completed",
            "allowed_reruns": 0,
            "reporting_only_no_selection": True,
            "inference_path": "official_unmodified",
        },
    )

    freeze = root / "reports" / "p13_diffregcd_external_freeze"
    freeze_run = freeze / "20260727T083622Z"
    _write_json(freeze_run / "freeze_manifest.json", _freeze_manifest())
    (freeze / "LATEST").write_text(
        "reports/p13_diffregcd_external_freeze/20260727T083622Z\n",
        encoding="utf-8",
    )

    validation = root / "reports" / "p13_diffregcd_validation"
    validation_run = validation / "20260727T070000Z"
    _write_json(
        validation_run / "validation" / "results.json",
        _validation_results(),
    )
    (validation / "LATEST").write_text(
        "reports/p13_diffregcd_validation/20260727T070000Z\n",
        encoding="utf-8",
    )
    return root


def test_reproduction_gate_accepts_recorded_numbers() -> None:
    gate = reproduction_gate(_summary())
    assert gate["passed"]
    assert gate["mean_shift_advantage"]["passed"]
    assert all(
        record["passed"] for record in gate["scenarios"].values()
    )


def test_reproduction_gate_rejects_drifted_numbers() -> None:
    drifted = dict(RECORDED_PROJECT_F1)
    drifted["shift:s2:t2"] += 0.001
    gate = reproduction_gate(_summary(project_f1=drifted))
    assert not gate["passed"]
    assert not gate["scenarios"]["shift:s2:t2"]["passed"]


def test_reproduction_gate_rejects_mismatched_support() -> None:
    summary = _summary()
    summary["scenario_comparison"][0]["project_valid_pixels"] = 1
    summary["scenario_comparison"][0]["same_valid_pixel_count"] = False
    gate = reproduction_gate(summary)
    assert not gate["passed"]


def test_marker_gate_requires_completed_and_sealed() -> None:
    good_p12 = {"status": "completed", "allowed_reruns": 0}
    good_p13 = {
        "status": "completed",
        "allowed_reruns": 0,
        "reporting_only_no_selection": True,
        "inference_path": "official_unmodified",
    }
    assert marker_gate(good_p12, good_p13)["passed"]
    assert not marker_gate(
        {"status": "started_irreversible", "allowed_reruns": 0},
        good_p13,
    )["passed"]
    assert not marker_gate(
        good_p12, {**good_p13, "allowed_reruns": 1}
    )["passed"]
    assert not marker_gate(
        good_p12, {**good_p13, "inference_path": "intended_warped"}
    )["passed"]


def test_freeze_and_validation_gates() -> None:
    assert freeze_gate(_freeze_manifest())["passed"]
    assert not freeze_gate(
        {**_freeze_manifest(), "external_commit": "deadbeef"}
    )["passed"]
    assert not freeze_gate(
        {**_freeze_manifest(), "selected_external_path": "intended_warped"}
    )["passed"]

    assert validation_gate(_validation_results())["passed"]
    consumed = _validation_results()
    consumed["test_consumed"] = True
    assert not validation_gate(consumed)["passed"]
    drifted = _validation_results()
    drifted["clean_reproduction_audit"]["paths"][
        "official_unmodified"
    ]["change_f1"]["observed"] = 0.86
    assert not validation_gate(drifted)["passed"]


def test_external_assets_render() -> None:
    rows = external_comparison_rows(_summary())
    assert [row["scenario"] for row in rows] == [
        "Clean",
        "Shift-2",
        "Shift-4",
        "Shift-8",
    ]
    markdown = render_external_table_markdown(
        rows,
        project_method="safe_veto_task",
        external_method="official_unmodified",
    )
    assert "Mean shift advantage: **+18.46 pp**" in markdown
    assert "125,960,192" in markdown
    svg = render_external_figure_svg(rows)
    assert svg.startswith("<svg")
    assert "+25.51 pp" in svg
    assert "DiffRegCD" in svg


def test_build_package_writes_verified_bundle(tmp_path: Path) -> None:
    root = _project_tree(tmp_path / "project")
    output = root / "reports" / "p13_final_evidence" / "20260727T090000Z"
    verification = build_package(root, output)

    assert verification["status"] == "verified"
    assert verification["markers"]["passed"]
    assert verification["recorded_number_reproduction"]["passed"]
    assert verification["recomputed_comparison_matches_recorded"][
        "passed"
    ]
    assert verification["s2looking_labeled_test_sealed_note_present"]

    for relative in (
        "README.md",
        "verification.json",
        "evidence_manifest.json",
        "paper_assets/table_external_comparison.csv",
        "paper_assets/table_external_comparison.md",
        "paper_assets/figure_external_comparison.svg",
        "paper_assets/table_main_final_test.md",
        "paper_assets/figure_main_final_test.svg",
        "evidence/p12_final_test/levir/results.json",
        "evidence/p13_diffregcd_final_test/p12_vs_diffregcd.json",
        "evidence/p13_diffregcd_external_freeze/freeze_manifest.json",
        "evidence/p13_diffregcd_validation/validation/results.json",
        "evidence/LEVIR_TEST_CONSUMED.json",
        "evidence/DIFFREGCD_LEVIR_TEST_CONSUMED.json",
    ):
        assert (output / relative).is_file(), relative

    manifest = json.loads(
        (output / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_build_package_refuses_when_numbers_disagree(
    tmp_path: Path,
) -> None:
    drifted = dict(RECORDED_PROJECT_F1)
    drifted["shift:s3:t2"] -= 0.01
    root = _project_tree(tmp_path / "project", project_f1=drifted)
    output = root / "reports" / "p13_final_evidence" / "20260727T090000Z"
    with pytest.raises(ValueError, match="verification failed"):
        build_package(root, output)
    assert not output.exists()


def test_build_package_refuses_unfinished_marker(tmp_path: Path) -> None:
    root = _project_tree(tmp_path / "project")
    _write_json(
        root / "reports" / "p13_diffregcd_final_test"
        / "DIFFREGCD_LEVIR_TEST_CONSUMED.json",
        {
            "status": "started_irreversible",
            "allowed_reruns": 0,
            "reporting_only_no_selection": True,
            "inference_path": "official_unmodified",
        },
    )
    output = root / "reports" / "p13_final_evidence" / "20260727T090000Z"
    with pytest.raises(ValueError, match="markers"):
        build_package(root, output)
