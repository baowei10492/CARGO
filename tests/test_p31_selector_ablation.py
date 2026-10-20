"""P31 selector ablation: border-free rows in, the P27 ablation run on them, the P27 table reader out."""
from __future__ import annotations

import csv
import json

import pytest
import torch

from test_p30_evaluation import _setup


def test_border_free_row(tmp_path, monkeypatch) -> None:
    from tools.p27_ablation_tables import selector_cells
    from tools.p31_selector_ablation import check_bundles, run_row
    from tools.run_p27_selector import harness, open_dataset, write_csv

    tool, p30, evaluation, selector, groups, checkpoints = _setup(tmp_path, monkeypatch)
    names = [spec["name"] for spec in evaluation["datasets"]]
    device = torch.device("cpu")
    root = tmp_path / "row"
    (root / "features").mkdir(parents=True)
    (root / "bundles").mkdir()
    summary = tool.run_features(p30, evaluation, selector, device, root / "features")
    (root / "features" / "features_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    tool.run_train(p30, evaluation, selector, root / "features", root / "bundles")

    # Test records as the P30 / P31 evaluate step writes them: one file per dataset.
    h = harness(evaluation)
    records = []
    for spec in evaluation["datasets"]:
        dataset = open_dataset(spec, spec["split"])
        crops = tool.margin_crops(dataset, spec, h["size"], 64)
        rows = tool.records_for(dataset, spec, [(crop, "val") for crop in crops], tool.open_localizer(p30, "p30", device)[0],
                                groups["original"][spec["name"]], h, device, "test")
        bundle = torch.load(root / "bundles" / f"selector_{spec['name']}.pth", weights_only=False)
        tool.system_table(rows, bundle)                 # adds selector_probability, as in the real runs
        path = tmp_path / f"records_{spec['name']}.csv"
        write_csv(path, rows)
        records.append(path)

    with pytest.raises(SystemExit, match="not a border-free selector"):
        check_bundles(root, names, "another recipe")
    out = tmp_path / "ablation" / "toy" / "20260919T000000Z"
    result = run_row("toy", selector, "recipe", out, train_root=root, records=records, names=names)
    assert result["protocol"] == "p30 border-free"
    for name in names:
        entry = result["datasets"][name]
        assert entry["full_reproduces_main"]["tau_equal"]
        assert entry["full_reproduces_main"]["max_probability_difference"] < 1e-6
        assert set(entry["variants"]) == {"full", "registration_only", "task_only", "without_forward_backward"}
    with (out / "test_records.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == sum(1 for path in records for _ in open(path, encoding="utf-8")) - len(records)
    cells = selector_cells(tmp_path / "ablation", rows=(("toy", "Toy", tuple(names)),))
    assert len(cells) == len(names) and set(cells[0]["values"]) >= {"always", "gate", "full"}
