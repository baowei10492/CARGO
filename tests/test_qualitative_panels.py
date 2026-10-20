import numpy as np

import inspect

from rs_cd.datasets.levir import PairedChangeDataset
from tools.render_qualitative_panels import (
    FORBIDDEN_SPLITS,
    REQUIRED_BATCH_KEYS,
    overlay_error_map,
    select_cases,
)


def _record(
    sample_id: str,
    severity: int,
    raw_f1: float,
    aligned_f1: float,
    selected: bool,
) -> dict:
    return {
        "sample_id": sample_id,
        "scenario": "clean:s0:common" if severity == 0 else "shift",
        "scenario_severity": severity,
        "raw_f1": raw_f1,
        "aligned_f1": aligned_f1,
        "selected_aligned": selected,
        "router_probability": 0.5,
        "q_gate": True,
        "oracle_prefers_aligned": aligned_f1 > raw_f1,
    }


def test_test_split_is_refused() -> None:
    assert "test" in FORBIDDEN_SPLITS


def test_required_batch_keys_match_the_dataset_contract() -> None:
    # The tool reads batch["id"]; a rename in the dataset must fail here
    # rather than after the model has already been loaded onto the GPU.
    source = inspect.getsource(PairedChangeDataset.__getitem__)
    for key in REQUIRED_BATCH_KEYS:
        assert f'"{key}"' in source, key


def test_error_overlay_marks_three_outcomes() -> None:
    prediction = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    target = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    panel = overlay_error_map(prediction, target)
    assert tuple(panel[0, 0]) == (255, 255, 255)
    assert tuple(panel[0, 1]) == (214, 96, 77)
    assert tuple(panel[1, 0]) == (69, 117, 180)
    assert tuple(panel[1, 1]) == (0, 0, 0)


def test_case_selection_sorts_by_decision_quality() -> None:
    records = [
        _record("clean_small", 0, 0.90, 0.88, False),
        _record("clean_big", 0, 0.90, 0.50, False),
        _record("shift_best", 3, 0.60, 0.85, True),
        _record("shift_small", 3, 0.80, 0.82, True),
        _record("shift_missed", 3, 0.60, 0.90, False),
        _record("shift_harmful", 3, 0.85, 0.60, True),
    ]
    cases = select_cases(records, per_category=2)

    abstentions = cases["correct_abstention_clean"]
    assert [row["sample_id"] for row in abstentions] == [
        "clean_big",
        "clean_small",
    ]
    adoptions = cases["correct_adoption_shift"]
    assert adoptions[0]["sample_id"] == "shift_best"
    assert cases["missed_opportunity"][0]["sample_id"] == "shift_missed"
    assert cases["harmful_adoption"][0]["sample_id"] == "shift_harmful"


def test_crop_window_covers_change_with_margin() -> None:
    from tools.render_qualitative_panels import change_bounding_box

    gt = np.zeros((1024, 1024), dtype=np.uint8)
    gt[500:540, 600:630] = 255
    top, left, height, width = change_bounding_box([gt, gt, gt])
    assert height == width, "window must be square"
    assert height >= 192, "window must not be smaller than the minimum"
    assert top <= 500 and top + height >= 540
    assert left <= 600 and left + width >= 630
    assert height < 1024, "a small change must not return the full tile"


def test_crop_window_covers_predictions_outside_the_label() -> None:
    from tools.render_qualitative_panels import change_bounding_box

    gt = np.zeros((512, 512), dtype=np.uint8)
    gt[100:120, 100:120] = 255
    false_positive = np.zeros((512, 512), dtype=np.uint8)
    false_positive[400:420, 400:420] = 255
    top, left, height, width = change_bounding_box([gt, false_positive])
    assert top <= 100 and top + height >= 420
    assert left <= 100 and left + width >= 420


def test_crop_window_falls_back_when_nothing_changes() -> None:
    from tools.render_qualitative_panels import change_bounding_box

    empty = np.zeros((600, 800), dtype=np.uint8)
    top, left, height, width = change_bounding_box([empty])
    assert (height, width) == (192, 192)
    assert 0 <= top <= 600 - height
    assert 0 <= left <= 800 - width


def test_case_selection_respects_limit() -> None:
    records = [
        _record(f"shift_{index}", 3, 0.5, 0.9, True)
        for index in range(10)
    ]
    cases = select_cases(records, per_category=3)
    assert len(cases["correct_adoption_shift"]) == 3
    assert cases["correct_abstention_clean"] == []
