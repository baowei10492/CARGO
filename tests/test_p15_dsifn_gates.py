from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.audit_augmentation_leakage import (
    canonical_difference_hash,
    canonical_pair_digest,
    hamming_distance,
    leakage_report,
)
from tools.build_source_crop_split import (
    assign_splits,
    build_clusters,
    rewrite_manifest,
)
from tools.summarize_p15_gate0 import (
    clean_q_trigger_rate,
    gate0_decision,
    mean_injected_dispersion,
)


CONFIG_PATH = Path("configs/p15_dsifn_screening.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _image(seed: int, size: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_canonical_digest_is_dihedral_invariant() -> None:
    t1 = _image(1)
    t2 = _image(2)
    reference = canonical_pair_digest(t1, t2)
    for rotation in range(4):
        rotated = (
            np.ascontiguousarray(np.rot90(t1, rotation)),
            np.ascontiguousarray(np.rot90(t2, rotation)),
        )
        assert canonical_pair_digest(*rotated) == reference
    mirrored = (
        np.ascontiguousarray(np.fliplr(t1)),
        np.ascontiguousarray(np.fliplr(t2)),
    )
    assert canonical_pair_digest(*mirrored) == reference


def test_canonical_digest_separates_distinct_pairs() -> None:
    assert canonical_pair_digest(_image(1), _image(2)) != (
        canonical_pair_digest(_image(3), _image(4))
    )


def test_canonical_digest_is_joint_over_t1_and_t2() -> None:
    t1, t2 = _image(5), _image(6)
    independent = canonical_pair_digest(
        np.ascontiguousarray(np.rot90(t1, 1)), t2
    )
    assert independent != canonical_pair_digest(t1, t2)


def test_difference_hash_is_dihedral_invariant() -> None:
    image = _image(7, size=64)
    reference = canonical_difference_hash(image)
    assert canonical_difference_hash(np.rot90(image, 2)) == reference
    assert canonical_difference_hash(np.fliplr(image)) == reference
    assert hamming_distance(reference, reference) == 0


def test_leakage_report_flags_cross_split_copies() -> None:
    records = [
        {
            "sample_id": "train_1",
            "split": "train",
            "canonical_digest": "a" * 64,
            "difference_hash": 0,
        },
        {
            "sample_id": "val_1",
            "split": "val",
            "canonical_digest": "a" * 64,
            "difference_hash": 0,
        },
        {
            "sample_id": "test_1",
            "split": "test",
            "canonical_digest": "b" * 64,
            "difference_hash": 2**63,
        },
    ]
    report = leakage_report(records)
    assert report["cross_split_exact_duplicate_groups"] == 1
    assert report["cross_split_exact_duplicate_samples"] == 2
    assert not report["decision"]["official_split_usable"]
    assert "re-split" in report["decision"]["required_action"]


def test_leakage_report_accepts_clean_splits() -> None:
    records = [
        {
            "sample_id": "train_1",
            "split": "train",
            "canonical_digest": "a" * 64,
            "difference_hash": 0b0000,
        },
        {
            "sample_id": "val_1",
            "split": "val",
            "canonical_digest": "c" * 64,
            "difference_hash": 0xFFFFFFFFFFFFFFFF,
        },
    ]
    report = leakage_report(records)
    assert report["cross_split_exact_duplicate_groups"] == 0
    assert report["cross_split_near_duplicate_pairs"] == 0
    assert report["decision"]["official_split_usable"]


def test_leakage_report_flags_near_duplicates() -> None:
    records = [
        {
            "sample_id": "train_1",
            "split": "train",
            "canonical_digest": "a" * 64,
            "difference_hash": 0b1111,
        },
        {
            "sample_id": "val_1",
            "split": "val",
            "canonical_digest": "d" * 64,
            "difference_hash": 0b1110,
        },
    ]
    report = leakage_report(records, near_duplicate_hamming_max=5)
    assert report["cross_split_exact_duplicate_groups"] == 0
    assert report["cross_split_near_duplicate_pairs"] == 1
    assert not report["decision"]["official_split_usable"]


def _hygiene(
    *,
    dispersion: float,
    clean_q: float,
    tile_size: int = 256,
    center_crop: int | None = None,
    tiles: int = 4,
) -> dict:
    injected = {
        key: {
            "median_max_pairwise_tile_distance": dispersion,
            "tiles_per_image": [tiles],
            "forced_choice_regime": False,
        }
        for key in ("shift:s1:t2", "shift:s2:t2", "shift:s3:t2")
    }
    return {
        "label": "unit-test",
        "tile_size": tile_size,
        "center_crop": center_crop,
        "scenarios": {
            "clean:s0:common": {
                "median_max_pairwise_tile_distance": 9.0,
                "tiles_per_image": [tiles],
                "forced_choice_regime": True,
                "whole_image_q_trigger_rate": clean_q,
            },
            **injected,
        },
    }


def test_dispersion_ignores_clean_and_averages_injected() -> None:
    summary = _hygiene(dispersion=2.0, clean_q=0.1)
    assert mean_injected_dispersion(summary) == pytest.approx(2.0)
    assert clean_q_trigger_rate(summary) == pytest.approx(0.1)


def test_forced_choice_scenario_is_rejected_as_evidence() -> None:
    summary = _hygiene(dispersion=2.0, clean_q=0.1)
    summary["scenarios"]["shift:s1:t2"]["forced_choice_regime"] = True
    with pytest.raises(ValueError, match="forced-choice"):
        mean_injected_dispersion(summary)


def test_gate0_green_when_close_to_matched_control() -> None:
    decision = gate0_decision(
        _hygiene(dispersion=7.9, clean_q=0.08),
        _hygiene(dispersion=7.6, clean_q=0.0625, center_crop=512),
        config=_config(),
    )
    assert decision["branch"] == "green"
    assert decision["action"] == "proceed_to_gate1"
    assert decision["geometry_match"]["passed"]


def test_gate0_red_needs_both_signals() -> None:
    config = _config()
    control = _hygiene(dispersion=7.6, clean_q=0.0625, center_crop=512)
    both_red = gate0_decision(
        _hygiene(dispersion=12.0, clean_q=0.55), control, config=config
    )
    assert both_red["branch"] == "red"
    assert both_red["action"].startswith("stop_loss")

    one_red = gate0_decision(
        _hygiene(dispersion=12.0, clean_q=0.20), control, config=config
    )
    assert one_red["branch"] == "yellow"
    assert one_red["action"] == "proceed_to_gate1"


def test_gate0_reports_geometry_mismatch() -> None:
    decision = gate0_decision(
        _hygiene(dispersion=1.0, clean_q=0.05, tile_size=256),
        _hygiene(
            dispersion=0.5,
            clean_q=0.0625,
            tile_size=512,
            center_crop=512,
            tiles=4,
        ),
        config=_config(),
    )
    assert not decision["geometry_match"]["passed"]


def test_gate0_records_no_label_or_test_access() -> None:
    decision = gate0_decision(
        _hygiene(dispersion=7.7, clean_q=0.07),
        _hygiene(dispersion=7.6, clean_q=0.0625, center_crop=512),
        config=_config(),
    )
    assert decision["labels_read"] is False
    assert decision["test_split_touched"] is False


def _record(
    sample_id: str,
    split: str,
    digest: str,
    dhash: int,
) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "canonical_digest": digest,
        "difference_hash": dhash,
    }


def test_clusters_merge_exact_and_near_duplicates() -> None:
    records = [
        _record("a", "train", "d1", 0b0000),
        _record("b", "val", "d1", 0b1111_1111),
        _record("c", "train", "d2", 0b0001),
        _record("d", "test", "d3", 0xFFFF_FFFF_FFFF_FFFF),
    ]
    clusters = build_clusters(records, cluster_hamming_max=2)
    assert clusters["a"] == clusters["b"], "exact digest must merge"
    assert clusters["a"] == clusters["c"], "hamming 1 must merge"
    assert clusters["d"] != clusters["a"]


def test_clusters_respect_hamming_threshold() -> None:
    records = [
        _record("a", "train", "d1", 0b0000),
        _record("b", "val", "d2", 0b0111),
    ]
    assert (
        build_clusters(records, cluster_hamming_max=2)["a"]
        != build_clusters(records, cluster_hamming_max=2)["b"]
    )
    merged = build_clusters(records, cluster_hamming_max=5)
    assert merged["a"] == merged["b"]


def test_assign_splits_keeps_clusters_whole_and_is_deterministic() -> None:
    records = [
        _record(f"s{index}", "train", f"d{index // 3}", index * 97)
        for index in range(30)
    ]
    clusters = build_clusters(records, cluster_hamming_max=-1)
    first = assign_splits(
        records,
        clusters,
        protected_split="test",
        holdout_fraction=0.2,
        salt="unit",
    )
    second = assign_splits(
        records,
        clusters,
        protected_split="test",
        holdout_fraction=0.2,
        salt="unit",
    )
    assert first["assignments"] == second["assignments"]
    by_cluster = defaultdict(set)
    for sample_id, split in first["assignments"].items():
        by_cluster[clusters[sample_id]].add(split)
    assert all(len(splits) == 1 for splits in by_cluster.values())
    assert set(first["summary"]["split_counts"]) <= {"train", "val"}


def test_protected_split_is_never_reassigned_and_leaks_are_dropped() -> None:
    records = [
        _record("train_clean", "train", "d1", 0),
        _record("train_leak", "train", "d2", 1 << 40),
        _record("test_1", "test", "d2", 1 << 40),
    ]
    clusters = build_clusters(records, cluster_hamming_max=0)
    result = assign_splits(
        records,
        clusters,
        protected_split="test",
        holdout_fraction=0.2,
        salt="unit",
    )
    assert result["assignments"]["test_1"] == "test"
    assert "train_leak" not in result["assignments"]
    assert result["dropped_sample_ids"] == ["train_leak"]
    assert result["summary"][
        "clusters_contaminating_protected_split"
    ] == 1


def test_rewrite_manifest_sets_parent_id_and_drops_removed() -> None:
    manifest = [
        {
            "sample_id": "a",
            "split": "train",
            "t1_path": "train/t1/a.jpg",
            "t2_path": "train/t2/a.jpg",
            "mask_path": "train/mask/a.png",
            "parent_id": "a",
        },
        {
            "sample_id": "gone",
            "split": "train",
            "t1_path": "train/t1/gone.jpg",
            "t2_path": "train/t2/gone.jpg",
            "mask_path": "train/mask/gone.png",
            "parent_id": "gone",
        },
    ]
    rows = rewrite_manifest(
        manifest,
        assignments={"a": "val"},
        clusters={"a": "cluster_00007", "gone": "cluster_00008"},
    )
    assert len(rows) == 1
    assert rows[0]["split"] == "val"
    assert rows[0]["parent_id"] == "cluster_00007"
    assert rows[0]["t1_path"] == "train/t1/a.jpg"


def test_chaining_warning_when_one_cluster_dominates() -> None:
    records = [
        _record(f"s{index}", "train", "same", 0) for index in range(10)
    ]
    clusters = build_clusters(records, cluster_hamming_max=0)
    result = assign_splits(
        records,
        clusters,
        protected_split="test",
        holdout_fraction=0.2,
        salt="unit",
    )
    assert result["summary"]["chaining_warning"] is True
    assert result["summary"]["largest_cluster_fraction"] == 1.0
