import pytest

from tools.summarize_matched_image_size_control import (
    matched_image_size_decision,
)


def _summary(pck1, pck2, image_size, *, s8_pck1=None, boundary=0.05):
    return {
        "per_shift": {
            "shift:s2:t2": {
                "raw_pck1": pck1,
                "raw_pck2": pck2,
                "boundary_fraction": boundary,
            },
            "shift:s3:t2": {
                "raw_pck1": pck1 if s8_pck1 is None else s8_pck1,
                "raw_pck2": pck2,
                "boundary_fraction": 0.45,
            },
        },
        "evaluation_image_size": image_size,
    }


# The real WHU stage-A numbers (main record 73).
WHU = _summary(0.13218390804597702, 0.4367816091954023, 512, s8_pck1=0.36)


def _decide(matched_pck1, native_pck1=0.90):
    return matched_image_size_decision(
        _summary(native_pck1, 0.95, "native"),
        _summary(matched_pck1, 0.80, 512),
        WHU,
        exonerate_min=0.60,
        convict_max=0.30,
        retention_min=0.80,
    )


def test_reference_survives_the_crop_blames_the_dataset() -> None:
    decision = _decide(0.78)
    assert decision["verdict"] == "image_size_not_the_cause"
    assert (
        decision["attribution"]
        == "dataset_property_second_negative_case_supported"
    )
    assert decision["retention_ratio"] == pytest.approx(0.78 / 0.90)


def test_reference_collapses_with_the_crop_blames_image_size() -> None:
    decision = _decide(0.22)
    assert decision["verdict"] == "image_size_is_the_dominant_confound"
    assert (
        decision["attribution"]
        == "whu_verdict_limited_to_512px_configuration"
    )


def test_middle_band_is_ambiguous() -> None:
    decision = _decide(0.45)
    assert decision["verdict"] == "ambiguous_partial_confound"
    assert (
        decision["attribution"]
        == "neither_explanation_may_be_claimed_alone"
    )


def test_high_absolute_but_large_drop_is_not_exoneration() -> None:
    # 0.65 clears the absolute bar but retains only 0.65/0.95 = 0.68 of
    # the native value, so image size demonstrably hurt: not exonerated.
    decision = matched_image_size_decision(
        _summary(0.95, 0.98, "native"),
        _summary(0.65, 0.80, 512),
        WHU,
        exonerate_min=0.60,
        convict_max=0.30,
        retention_min=0.80,
    )
    assert decision["verdict"] == "ambiguous_partial_confound"


def test_control_never_reopens_the_target_gates() -> None:
    for matched in (0.78, 0.45, 0.22):
        decision = _decide(matched)
        assert decision["gates_reopened"] is False
        # The target's own numbers are passed through untouched.
        assert decision["target"]["pck1"] == pytest.approx(0.1321839, abs=1e-6)


def test_primary_scenario_is_the_least_contaminated_one() -> None:
    decision = _decide(0.78)
    assert decision["primary_scenario"] == "shift:s2:t2"
    # Shift-8 is reported but explicitly non-decisive.
    reported = decision["shift8_reported_not_decisive"]
    assert reported["target"]["boundary_fraction"] == pytest.approx(0.45)


def test_unusable_reference_raises() -> None:
    with pytest.raises(ValueError):
        matched_image_size_decision(
            _summary(0.0, 0.0, "native"),
            _summary(0.5, 0.6, 512),
            WHU,
            exonerate_min=0.60,
            convict_max=0.30,
            retention_min=0.80,
        )


def test_overlapping_bands_raise() -> None:
    with pytest.raises(ValueError):
        matched_image_size_decision(
            _summary(0.90, 0.95, "native"),
            _summary(0.85, 0.90, 512),
            WHU,
            exonerate_min=0.60,
            convict_max=0.90,
            retention_min=0.80,
        )


def test_control_requires_image_size_to_change() -> None:
    with pytest.raises(ValueError, match="image size was not varied"):
        matched_image_size_decision(
            _summary(0.90, 0.95, 512),
            _summary(0.78, 0.80, 512),
            WHU,
            exonerate_min=0.60,
            convict_max=0.30,
            retention_min=0.80,
        )
