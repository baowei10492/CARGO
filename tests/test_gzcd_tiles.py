import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.build_gzcd_tiles import (
    EXPECTED_GEOMETRIC_TILES,
    SPLIT_ASSIGNMENT,
    build,
    index_by_raster,
    is_no_data,
    raster_token,
    tile_origins,
)


def test_raster_token_ignores_differing_dates() -> None:
    # T1, T2 and the label carry different acquisition dates, so pairing
    # must key on the testN token alone.
    assert raster_token(Path("P_GZ_test10_2010_1103_Level_18.tif")) == "test10"
    assert raster_token(Path("P_GZ_test10_2016_1128_Level_18.tif")) == "test10"
    assert raster_token(Path("P_GZ_test10_2010_2016.png")) == "test10"
    assert raster_token(Path("P_GZ_test4_2019_0615_Level_18.tif")) == "test4"


def test_raster_token_rejects_unparseable_names() -> None:
    with pytest.raises(ValueError, match="Cannot parse"):
        raster_token(Path("random_file.tif"))


def test_split_assignment_covers_every_raster_once() -> None:
    assert len(SPLIT_ASSIGNMENT) == 20
    assert set(SPLIT_ASSIGNMENT.values()) == {"train", "val", "test"}
    counts = {
        split: sum(1 for value in SPLIT_ASSIGNMENT.values() if value == split)
        for split in ("train", "val", "test")
    }
    assert counts == {"train": 9, "val": 5, "test": 6}


def test_tile_origins_drops_partial_edges() -> None:
    origins = tile_origins(width=1100, height=600, tile=512)
    assert origins == [(0, 0), (0, 512)]


def test_no_data_filter_uses_pure_black_only() -> None:
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    assert is_no_data(black, max_black_fraction=0.5)
    dark = np.full((10, 10, 3), 1, dtype=np.uint8)
    assert not is_no_data(dark, max_black_fraction=0.5)
    half = np.zeros((10, 10, 3), dtype=np.uint8)
    half[:, 5:] = 200
    assert is_no_data(half, max_black_fraction=0.5)
    assert not is_no_data(half, max_black_fraction=0.51)


def _write_raster(
    root: Path,
    token: str,
    *,
    width: int,
    height: int,
    black_columns: int = 0,
) -> None:
    rng = np.random.default_rng(abs(hash(token)) % 2**32)
    image = rng.integers(1, 256, size=(height, width, 3), dtype=np.uint8)
    if black_columns:
        image[:, :black_columns] = 0
    label = np.zeros((height, width), dtype=np.uint8)
    label[: height // 4, : width // 4] = 255
    for folder, name, array in (
        ("T1", f"P_GZ_{token}_2010_0101_Level_18.tif", image),
        ("T2", f"P_GZ_{token}_2018_0101_Level_18.tif", image[::-1].copy()),
        ("labels_change", f"P_GZ_{token}_2010_2018.png", label),
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(root / folder / name)


def test_index_by_raster_rejects_duplicate_tokens(tmp_path: Path) -> None:
    root = tmp_path / "src"
    (root / "T1").mkdir(parents=True)
    for name in (
        "P_GZ_test1_2010_0101_Level_18.tif",
        "P_GZ_test1_2011_0101_Level_18.tif",
    ):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
            root / "T1" / name
        )
    with pytest.raises(ValueError, match="Duplicate raster token"):
        index_by_raster(root / "T1", (".tif",))


def test_build_keeps_each_raster_in_one_split(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _write_raster(source, "test1", width=1024, height=512)
    _write_raster(source, "test13", width=1024, height=512)
    _write_raster(source, "test10", width=512, height=512)
    output = tmp_path / "out"
    manifest = tmp_path / "manifest.csv"
    report = build(
        source,
        output,
        tile=512,
        max_black_fraction=0.5,
        manifest_path=manifest,
        dry_run=False,
    )
    assert report["status"] == "completed"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    by_parent: dict[str, set[str]] = {}
    for row in rows:
        by_parent.setdefault(row["parent_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_parent.values())
    # test1 -> train, test13 -> val, test10 -> test
    assert by_parent["test1"] == {"train"}
    assert by_parent["test13"] == {"val"}
    assert by_parent["test10"] == {"test"}
    assert report["tiles_per_split"] == {"train": 2, "val": 2, "test": 1}
    for row in rows:
        assert (output / row["t1_path"]).is_file()
        assert (output / row["mask_path"]).is_file()


def test_build_drops_no_data_tiles(tmp_path: Path) -> None:
    source = tmp_path / "src"
    # left half of the raster is pure black, so the first tile goes
    _write_raster(source, "test1", width=1024, height=512, black_columns=512)
    report = build(
        source,
        tmp_path / "out",
        tile=512,
        max_black_fraction=0.5,
        manifest_path=tmp_path / "manifest.csv",
        dry_run=True,
    )
    assert report["dropped_no_data_tiles"]["train"] == 1
    assert report["tiles_per_split"]["train"] == 1


def test_build_rejects_unknown_raster(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _write_raster(source, "test99", width=512, height=512)
    with pytest.raises(ValueError, match="absent from the frozen split"):
        build(
            source,
            tmp_path / "out",
            tile=512,
            max_black_fraction=0.5,
            manifest_path=tmp_path / "manifest.csv",
            dry_run=True,
        )


def test_build_rejects_shape_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _write_raster(source, "test1", width=512, height=512)
    label = np.zeros((256, 256), dtype=np.uint8)
    Image.fromarray(label).save(
        source / "labels_change" / "P_GZ_test1_2010_2018.png"
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        build(
            source,
            tmp_path / "out",
            tile=512,
            max_black_fraction=0.5,
            manifest_path=tmp_path / "manifest.csv",
            dry_run=True,
        )


def test_expected_geometric_counts_match_the_protocol() -> None:
    assert EXPECTED_GEOMETRIC_TILES == {"train": 511, "val": 109, "test": 109}
