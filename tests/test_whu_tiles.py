import numpy as np
import pytest
from PIL import Image

from tools.build_whu_tiles import (
    build_split,
    column_partition,
    full_tile_grid,
)


def test_full_tile_grid_drops_incomplete_edges() -> None:
    # The real WHU train raster.
    assert full_tile_grid(21243, 15354, 512) == (41, 29)
    # The real WHU test raster: the 1-pixel remainder column is dropped.
    assert full_tile_grid(11265, 15354, 512) == (22, 29)
    assert full_tile_grid(1024, 512, 512) == (2, 1)
    with pytest.raises(ValueError):
        full_tile_grid(300, 300, 512)
    with pytest.raises(ValueError):
        full_tile_grid(1024, 1024, 0)


def test_column_partition_is_disjoint_with_buffer() -> None:
    partition = column_partition(41, 6, 1)
    assert partition["train"] == list(range(34))
    assert partition["buffer_dropped"] == [34]
    assert partition["val"] == [35, 36, 37, 38, 39, 40]
    assert not set(partition["train"]) & set(partition["val"])
    # No train column is adjacent to a validation column.
    assert max(partition["train"]) + 1 < min(partition["val"])


def test_column_partition_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError):
        column_partition(5, 5, 1)
    with pytest.raises(ValueError):
        column_partition(41, 0, 1)
    with pytest.raises(ValueError):
        column_partition(41, 6, -1)


def _write_raster(path, array, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        # Passing a bool array with mode="1" makes PIL reinterpret the
        # bytes as packed bits; go through uint8 then convert.
        image = Image.fromarray(
            (array.astype(np.uint8) * 255)
        ).convert("1")
    else:
        image = Image.fromarray(array)
    image.save(path)


def _make_fake_dataset(root, width, height):
    two = root / "1. The two-period image data"
    rng = np.random.default_rng(0)
    for raster_split, (w, h) in (
        ("train", (width, height)),
        ("test", (width, height)),
    ):
        for year in ("2012", "2016"):
            _write_raster(
                two
                / year
                / "whole_image"
                / raster_split
                / "image"
                / f"{year}_{raster_split}.tif",
                rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            )
        mask = np.zeros((h, w), dtype=bool)
        mask[:, : w // 2] = True
        _write_raster(
            two / "change_label" / raster_split / "change_label.tif",
            mask,
            binary=True,
        )
    return root


def test_build_split_emits_disjoint_columns(tmp_path) -> None:
    # 2348x600 -> 4 complete columns, 1 complete row.
    raw = _make_fake_dataset(tmp_path / "raw", width=2348, height=600)
    out = tmp_path / "out"
    records = build_split(
        raw,
        out,
        "train",
        tile_size=512,
        validation_columns=1,
        buffer_columns=1,
        dry_run=False,
    )
    # columns 0,1 -> train; column 2 -> dropped buffer; column 3 -> val
    assert [record["sample_id"] for record in records] == [
        "whu_train_r00c00",
        "whu_train_r00c01",
        "whu_val_r00c03",
    ]
    assert [record["split"] for record in records] == [
        "train",
        "train",
        "val",
    ]
    assert 2 not in {record["column"] for record in records}
    for split, name in (
        ("train", "whu_train_r00c00"),
        ("train", "whu_train_r00c01"),
        ("val", "whu_val_r00c03"),
    ):
        for folder in ("A", "B", "label"):
            assert (out / split / folder / f"{name}.png").is_file()
    # Left half (0..1173) is change: columns 0 and 1 are fully change,
    # column 3 (1536..2047) has none.
    assert records[0]["change_fraction"] == pytest.approx(1.0)
    assert records[1]["change_fraction"] == pytest.approx(1.0)
    assert records[2]["change_fraction"] == pytest.approx(0.0)
    # Tiles must be readable and carry the requested size.
    tile = Image.open(out / "val" / "A" / "whu_val_r00c03.png")
    assert tile.size == (512, 512)


def test_build_split_test_raster_uses_all_columns(tmp_path) -> None:
    raw = _make_fake_dataset(tmp_path / "raw", width=1536, height=512)
    out = tmp_path / "out"
    records = build_split(
        raw,
        out,
        "test",
        tile_size=512,
        validation_columns=1,
        buffer_columns=1,
        dry_run=True,
    )
    assert {record["split"] for record in records} == {"test"}
    assert len(records) == 3
    assert [record["column"] for record in records] == [0, 1, 2]
    # dry_run must not write anything.
    assert not (out / "test").exists()


def test_build_split_rejects_mismatched_label(tmp_path) -> None:
    raw = _make_fake_dataset(tmp_path / "raw", width=1024, height=512)
    two = raw / "1. The two-period image data"
    bad = np.zeros((512, 512), dtype=bool)
    _write_raster(
        two / "change_label" / "test" / "change_label.tif",
        bad,
        binary=True,
    )
    with pytest.raises(ValueError):
        build_split(
            raw,
            tmp_path / "out",
            "test",
            tile_size=512,
            validation_columns=1,
            buffer_columns=1,
            dry_run=True,
        )
