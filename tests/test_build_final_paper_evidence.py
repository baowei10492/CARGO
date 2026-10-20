import pytest

from tools.build_final_paper_evidence import (
    paper_main_rows,
    render_main_figure_svg,
    render_main_table_markdown,
)


def _summary():
    return {
        "status": "completed",
        "primary_method": "safe_veto_task",
        "clean": {
            "raw_f1": 0.90,
            "primary_f1": 0.90,
            "primary_minus_raw": 0.0,
        },
        "shift": {
            "shift:s1:t2": {
                "raw_f1": 0.86,
                "primary_f1": 0.89,
                "primary_minus_raw": 0.03,
            },
            "shift:s2:t2": {
                "raw_f1": 0.79,
                "primary_f1": 0.87,
                "primary_minus_raw": 0.08,
            },
            "shift:s3:t2": {
                "raw_f1": 0.66,
                "primary_f1": 0.82,
                "primary_minus_raw": 0.16,
            },
        },
        "mean_shift_gain": 0.09,
        "test_result_used_for_model_selection": False,
    }


def test_paper_rows_and_outputs() -> None:
    rows = paper_main_rows(_summary())
    assert [row["shift_pixels"] for row in rows] == [0, 2, 4, 8]
    assert rows[-1]["gain"] == pytest.approx(0.16)
    markdown = render_main_table_markdown(
        rows, primary_method="safe_veto_task"
    )
    assert "Mean shift gain: **+9.00 pp**" in markdown
    svg = render_main_figure_svg(rows, primary_label="Safe-veto task")
    assert svg.startswith("<svg")
    assert "Shift-8" in svg
    assert "+16.00 pp" in svg


def test_paper_rows_reject_inconsistent_mean() -> None:
    summary = _summary()
    summary["mean_shift_gain"] = 0.10
    with pytest.raises(ValueError, match="Mean shift gain mismatch"):
        paper_main_rows(summary)
