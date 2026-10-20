"""DiffRegCD's input convention (T1 perturbed, T2 reference) for the P30/P31 pipeline (record 192.72)."""
from __future__ import annotations

import torch

from test_p27_f1 import _tiny_localizer
from test_p30_evaluation import _setup


def test_detector_still_receives_earlier_then_later() -> None:
    from tools.p31_swapped_convention import EarlierFirst

    seen = []
    EarlierFirst(lambda a, b: seen.append((a, b)) or {})(torch.zeros(1), torch.ones(1))
    assert seen[0][0].item() == 1.0 and seen[0][1].item() == 0.0   # (moved T1, reference T2)


def test_swapped_pipeline_matches_its_raw_path_and_the_clean_run(tmp_path, monkeypatch) -> None:
    from rs_cd.p27_selector import pooled_f1
    from tools.p31_swapped_convention import swapped_raw, swapped_records
    from tools.run_p27_selector import harness, open_dataset

    tool, _, evaluation, _, groups, _ = _setup(tmp_path, monkeypatch)
    h = harness(evaluation)
    device = torch.device("cpu")
    for spec in evaluation["datasets"]:
        dataset = open_dataset(spec, spec["split"])
        crops = tool.margin_crops(dataset, spec, h["size"], 64)
        detector = groups["original"][spec["name"]]
        records = swapped_records(dataset, spec, crops, _tiny_localizer(), detector, h, device, "toy")
        assert len(records) == len(crops) * (1 + len(h["levels"]))
        raw = swapped_raw(dataset, spec, crops, detector, h, device)
        usual = tool.raw_scores(dataset, spec, crops, detector, h, device)
        for level in ("0", *(str(k) for k in h["levels"])):
            members = [r for r in records if str(r["level"]) == level]
            assert abs(pooled_f1((r, "raw") for r in members) - raw[level]) < 1e-9
        assert abs(raw["0"] - usual["0"]) < 1e-9                       # aligned pair: nothing to exchange
        assert abs(raw["3"] - usual["3"]) > 1e-6                       # misaligned: the moved date differs
