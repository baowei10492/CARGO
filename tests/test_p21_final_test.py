"""Amendment D: the one-shot LEVIR-CD test must be the validation cell, run once.

The test split is read a single time per detector, so nothing on this path
can be debugged afterwards. These tests pin, without weights or a GPU, that
the test evaluator reproduces the validation evaluator, that every setting is
the validation setting, that only the frozen inputs are accepted, and that a
second run cannot start.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from rs_cd.dual_path import (
    ROUTER_FEATURE_NAMES,
    SAFE_REGISTRATION_FEATURE_NAMES,
    SAFE_TASK_FEATURE_NAMES,
    SafeAlignmentClassifier,
)
from rs_cd.models.dual_path_adapter import DualPathAdapter
from tools.evaluate_p21_final_test import (
    VALIDATION_MODEL_OVERRIDES,
    check_protocol,
    claim_test_marker,
    complete_test_marker,
    load_cell_model,
    summarize_test_rows,
    verified_test_inputs,
)
from tools.evaluate_safe_alignment_transfer import evaluate_transfer_validation
from tools.freeze_p21_validation import (
    CLASSIFIER_FILES,
    freeze_matrix,
    write_frozen_manifest,
)
from tools.summarize_p21_matrix import break_even_fraction
from tools.train_evaluate_safe_alignment_classifier import _evaluate_validation

ROOT = Path(__file__).resolve().parents[1]
TEST_PROTOCOL = ROOT / "configs" / "p21_levir_final_test.yaml"
VALIDATION_PROTOCOL = ROOT / "configs" / "p7_safe_alignment_classifier.yaml"
VALIDATION_TOOL = ROOT / "tools" / "train_evaluate_safe_alignment_classifier.py"
RUN_SCRIPT = ROOT / "scripts" / "run_p21_final_test.sh"
REPORTED = ["raw", "always_aligned", "safe_veto_task"]


def test_test_settings_equal_the_validation_settings() -> None:
    test = yaml.safe_load(TEST_PROTOCOL.read_text(encoding="utf-8"))
    validation = yaml.safe_load(VALIDATION_PROTOCOL.read_text(encoding="utf-8"))
    for key in (
        "seed",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "amp",
        "threshold",
        "evaluation_border_pixels",
        "maximum_offset_pixels",
        "shift_severities",
        "primary_method",
    ):
        assert test[key] == validation[key], key
    assert test["split"] == "test"
    assert test["dataset"] == "LEVIR-CD"
    assert test["reporting_only_no_selection"] is True
    assert test["reported_methods"] == REPORTED
    for detector in ("BIT", "CD-Lamba", "CDxLSTM"):
        check_protocol(test, detector=detector)


def test_protocol_departures_are_refused() -> None:
    base = yaml.safe_load(TEST_PROTOCOL.read_text(encoding="utf-8"))
    for key, value in (
        ("split", "val"),
        ("dataset", "DSIFN-CD"),
        ("reporting_only_no_selection", False),
        ("reported_methods", ["raw", "safe_veto_task"]),
    ):
        with pytest.raises(ValueError):
            check_protocol({**base, key: value}, detector="BIT")
    with pytest.raises(ValueError):
        check_protocol(base, detector="DINOv3")


def test_model_overrides_match_the_validation_tool_source() -> None:
    text = VALIDATION_TOOL.read_text(encoding="utf-8")
    for key, value in VALIDATION_MODEL_OVERRIDES.items():
        literal = repr(value) if isinstance(value, bool) else f'"{value}"'
        assert f'config["model"]["{key}"] = {literal}' in text, key


class _Localizer(nn.Module):
    """Input-dependent registration carrying every field the evaluators read."""

    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))

    def estimate_registration(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        difference = (image_t1 - image_t2).flatten(1)
        dx = torch.round(torch.tanh(difference.mean(dim=1) * 20.0) * 8.0)
        dy = torch.round(torch.tanh(difference.std(dim=1) - 1.0) * 8.0)
        offset = torch.stack((dx, dy), dim=1)
        confidence = torch.sigmoid(difference.abs().mean(dim=1) - 1.0)
        return {
            "conditional_hard_offset_pixels": offset,
            "alignment_sample_gate": (offset.abs().amax(dim=1) > 0).float(),
            "registration_shift_probability": confidence,
            "registration_zero_probability": 1.0 - confidence,
            "registration_confidence": confidence,
            "registration_margin": 0.5 * confidence,
            "registration_local_margin": 0.25 * confidence,
            "registration_reliability": confidence.sqrt(),
        }


class _Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(3407)
        self.convolution = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        with torch.no_grad():
            self.convolution.weight.copy_(
                0.3 * torch.randn(self.convolution.weight.shape, generator=generator)
            )
            self.convolution.bias.zero_()

    def forward(
        self, image_t1: torch.Tensor, image_t2: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pair = torch.cat((image_t1, image_t2), dim=1)
        return {"change_logits": self.convolution(pair)}


def _samples(count: int = 6, size: int = 32) -> list[dict[str, object]]:
    generator = torch.Generator().manual_seed(3407)
    samples = []
    for index in range(count):
        image_t1 = torch.randn(3, size, size, generator=generator)
        image_t2 = image_t1 + 0.5 * torch.randn(3, size, size, generator=generator)
        mask = (torch.rand(1, size, size, generator=generator) > 0.6).float()
        samples.append(
            {"id": f"pair_{index}", "image_t1": image_t1, "image_t2": image_t2, "mask": mask}
        )
    return samples


def _classifiers() -> dict[str, dict[str, object]]:
    bundles = {}
    for offset, (name, features) in enumerate(
        (
            ("safe_alignment_registration", SAFE_REGISTRATION_FEATURE_NAMES),
            ("safe_alignment_task", SAFE_TASK_FEATURE_NAMES),
        )
    ):
        with torch.random.fork_rng():
            torch.manual_seed(3407 + offset)
            model = SafeAlignmentClassifier(
                input_dim=len(features), hidden_dims=(8, 4), dropout=0.0
            ).eval()
        bundles[name] = {
            "feature_indices": torch.tensor(
                [ROUTER_FEATURE_NAMES.index(feature) for feature in features]
            ),
            "feature_mean": torch.zeros(len(features)),
            "feature_std": torch.ones(len(features)),
            "model": model,
            "probability_threshold": 0.5,
        }
    return bundles


PROTOCOL = {
    "seed": 3407,
    "batch_size": 2,
    "num_workers": 0,
    "prefetch_factor": 2,
    "amp": False,
    "threshold": 0.5,
    "evaluation_border_pixels": 2,
    "maximum_offset_pixels": 8.0,
    "shift_severities": [1, 2, 3],
}
CONFIG = {
    "data": {"normalization": "imagenet"},
    "loss": {"bce_weight": 1.0, "dice_weight": 1.0, "pos_weight": 2.0},
}


def _normalized(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def value(item: object) -> object:
        if isinstance(item, torch.Tensor):
            item = item.item()
        if isinstance(item, float) and math.isnan(item):
            return "nan"
        return item

    return [{key: value(item) for key, item in row.items()} for row in rows]


def test_test_evaluator_reproduces_the_validation_evaluator(tmp_path: Path) -> None:
    """The cells were judged by one function; the test must run the same rows."""

    model = DualPathAdapter(_Localizer(), _Detector()).eval()
    classifiers = _classifiers()
    samples = _samples()
    device = torch.device("cpu")
    validation_dir = tmp_path / "validation"
    test_dir = tmp_path / "test"
    validation_dir.mkdir()
    test_dir.mkdir()

    # The validation tool calls this inside its own no-grad main; gradients
    # do not change any number, only whether PyTorch warns.
    with torch.no_grad():
        validation_rows, validation_decisions = _evaluate_validation(
            model,
            samples,
            classifiers,
            None,
            protocol=dict(PROTOCOL),
            config=CONFIG,
            device=device,
            output_dir=validation_dir,
            tile_config=None,
        )
    test_rows, test_decisions = evaluate_transfer_validation(
        model,
        samples,
        classifiers,
        protocol=dict(PROTOCOL),
        config=CONFIG,
        device=device,
        output_dir=test_dir,
    )

    assert _normalized(test_rows) == _normalized(validation_rows)
    assert _normalized(test_decisions) == _normalized(validation_decisions)
    # Not vacuous: displacement separates the fixed paths and the gate fires.
    by_key = {(row["scenario"], row["method"]): row for row in test_rows}
    assert by_key[("shift:s3:t2", "raw")]["f1"] != by_key[("shift:s3:t2", "always_aligned")]["f1"]
    assert any(
        (row["aligned_selection_rate"] or 0.0) > 0.0
        for row in test_rows
        if row["method"] == "q_gate"
    )


def _builder(_config: dict[str, object]) -> DualPathAdapter:
    return DualPathAdapter(_Localizer(), _Detector())


def test_cell_model_loads_strictly_with_the_validation_overrides() -> None:
    source = DualPathAdapter(_Localizer(), _Detector())
    with torch.no_grad():
        source.detector.convolution.bias.fill_(0.25)
    payload = {
        "config": {
            "data": {"dataset_root": "/train/LEVIR-CD", "manifest": "train.csv"},
            "loss": {"bce_weight": 1.0},
            "model": {"name": "dual_path_adapter", "localizer": {}, "detector": {}},
        },
        "model": source.state_dict(),
    }
    model, config = load_cell_model(
        payload,
        dataset_root="/data/LEVIR-CD",
        manifest="manifests/datasets/levir_cd.csv",
        builder=_builder,
    )
    assert config["data"]["dataset_root"] == "/data/LEVIR-CD"
    assert config["data"]["manifest"] == "manifests/datasets/levir_cd.csv"
    for key, value in VALIDATION_MODEL_OVERRIDES.items():
        assert config["model"][key] == value
    assert payload["config"]["data"]["dataset_root"] == "/train/LEVIR-CD"
    assert not model.training
    assert torch.all(model.detector.convolution.bias == 0.25)

    extra_key = {**payload, "model": {**source.state_dict(), "extra.weight": torch.zeros(1)}}
    with pytest.raises(RuntimeError):
        load_cell_model(extra_key, dataset_root="/data", manifest="m.csv", builder=_builder)
    single_detector = {
        **payload,
        "config": {**payload["config"], "model": {"name": "rschange_detector"}},
    }
    with pytest.raises(ValueError, match="dual-path"):
        load_cell_model(single_detector, dataset_root="/data", manifest="m.csv", builder=_builder)


def _summary_rows() -> list[dict[str, object]]:
    values = {
        "clean:s0:common": {"raw": 0.90, "always_aligned": 0.84, "safe_veto_task": 0.90, "loss_oracle": 0.91},
        "shift:s1:t2": {"raw": 0.86, "always_aligned": 0.89, "safe_veto_task": 0.89, "loss_oracle": 0.90},
        "shift:s2:t2": {"raw": 0.79, "always_aligned": 0.88, "safe_veto_task": 0.88, "loss_oracle": 0.89},
        "shift:s3:t2": {"raw": 0.67, "always_aligned": 0.89, "safe_veto_task": 0.87, "loss_oracle": 0.89},
    }
    rates = {"clean:s0:common": 0.0, "shift:s1:t2": 0.9, "shift:s2:t2": 0.95, "shift:s3:t2": 1.0}
    return [
        {
            "scenario": scenario,
            "method": method,
            "f1": f1,
            "aligned_selection_rate": rates[scenario] if method == "safe_veto_task" else None,
        }
        for scenario, methods in values.items()
        for method, f1 in methods.items()
    ]


def test_summary_reports_both_fixed_paths_and_the_router() -> None:
    summary = summarize_test_rows(
        _summary_rows(), reported_methods=REPORTED, primary_method="safe_veto_task"
    )
    assert summary["reporting_only_no_gate"] is True
    assert summary["test_result_used_for_model_selection"] is False
    assert list(summary["methods"]) == REPORTED

    routed = summary["methods"]["safe_veto_task"]
    assert routed["clean_minus_raw"] == pytest.approx(0.0)
    assert routed["shift_mean_gain_over_raw"] == pytest.approx((0.03 + 0.09 + 0.20) / 3)
    assert summary["methods"]["always_aligned"]["clean_minus_raw"] == pytest.approx(-0.06)
    assert summary["primary_aligned_selection_rate"] == {
        "clean": 0.0, "2px": 0.9, "4px": 0.95, "8px": 1.0,
    }

    raw_mean = (0.86 + 0.79 + 0.67) / 3
    oracle_mean = (0.90 + 0.89 + 0.89) / 3
    assert summary["capture_descriptive"]["mean_capture_rate"] == pytest.approx(
        (0.88 - raw_mean) / (oracle_mean - raw_mean)
    )
    assert summary["break_even_clean_fraction"] == pytest.approx(
        break_even_fraction(0.90, 0.88, 0.84, (0.89 + 0.88 + 0.89) / 3)
    )


def test_summary_requires_every_reported_row_and_the_oracle() -> None:
    without_oracle = [
        row
        for row in _summary_rows()
        if not (row["method"] == "loss_oracle" and row["scenario"] == "shift:s2:t2")
    ]
    without_always = [row for row in _summary_rows() if row["method"] != "always_aligned"]
    for rows in (without_oracle, without_always):
        with pytest.raises(ValueError):
            summarize_test_rows(rows, reported_methods=REPORTED, primary_method="safe_veto_task")


MATRIX = {
    "detectors": {
        "BIT": {"slug": "bit"},
        "CD-Lamba": {"slug": "cdlamba"},
        "CDxLSTM": {"slug": "cdxlstm"},
    },
    "datasets": {
        "LEVIR-CD": {"slug": "levir"},
        "DSIFN-CD": {"slug": "dsifn"},
        "GZ-CD": {"slug": "gzcd"},
    },
}
TAG = "20260915T000000Z"
COMMIT = "a3bc6ffc99d74c7a6a92a69f9471af2293dbff2a"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_cell(
    root: Path,
    name: str,
    *,
    raw_clean_f1: float = 0.88,
    guard_passed: bool = True,
) -> None:
    report = f"reports/p21/cells/{name}/{TAG}"
    base = root / report
    recorded = {
        "cell_summary.json": json.dumps(
            {
                "verdict": "no_step_needed",
                "preconditions": {"raw_clean_f1": raw_clean_f1},
            }
        ).encode(),
        "safe_alignment_experiment/validation_scenario_metrics.csv": f"scenario,method,f1\n{name}\n".encode(),
    }
    if name.endswith("_levir"):
        recorded["dual_path.pth"] = f"dual path of {name}".encode()
        for relative in CLASSIFIER_FILES.values():
            _write(base / relative, f"{relative} of {name}".encode())
        _write(base / "RSCHANGE_COMMIT", f"{COMMIT}\n".encode())
    for relative, data in recorded.items():
        _write(base / relative, data)
    _write(
        base / "instrument_guard.json",
        json.dumps({"passed": guard_passed, "stage1_best_val_f1": 0.87}).encode(),
    )
    listing = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {report}/{relative}\n"
        for relative, data in recorded.items()
    )
    listing += f"{'0' * 64}  configs/p21/{name}.yaml\n"
    _write(base / "SHA256SUMS.txt", listing.encode())
    _write(root / f"reports/p21/cells/{name}/LATEST", f"{report}\n".encode())


def _make_matrix(root: Path) -> None:
    for detector in MATRIX["detectors"].values():
        for dataset in MATRIX["datasets"].values():
            _make_cell(root, f"{detector['slug']}_{dataset['slug']}")


def test_freeze_records_every_cell_and_the_levir_test_inputs(tmp_path: Path) -> None:
    _make_matrix(tmp_path)
    manifest = freeze_matrix(tmp_path, MATRIX)

    assert manifest["status"] == "frozen"
    assert len(manifest["cells"]) == 9
    assert set(manifest["test_inputs"]) == {"BIT", "CD-Lamba", "CDxLSTM"}
    entry = manifest["test_inputs"]["CDxLSTM"]
    assert entry["rschange_commit"] == COMMIT
    dual_path = tmp_path / entry["dual_path"]["path"]
    assert entry["dual_path"]["sha256"] == hashlib.sha256(dual_path.read_bytes()).hexdigest()
    inputs = verified_test_inputs(manifest, "CDxLSTM", project_root=tmp_path)
    assert set(inputs["paths"]) == {"dual_path", *CLASSIFIER_FILES}


def test_freeze_records_the_recipe_each_cell_ran(tmp_path: Path) -> None:
    """A cell that deviates from the shared config must be visible later."""

    _make_matrix(tmp_path)
    manifest = freeze_matrix(tmp_path, MATRIX)
    cell = manifest["cells"]["cdxlstm_gzcd"]
    assert cell["stage1_config"] == {
        "path": "configs/p21/cdxlstm_gzcd.yaml",
        "sha256": "0" * 64,
    }
    assert cell["raw_clean_f1"] == 0.88
    assert cell["stage1_best_val_f1"] == 0.87


def test_freeze_refuses_a_numerically_invalid_cell(tmp_path: Path) -> None:
    """An all-zero run is summarised as no_step_needed; it is not a verdict."""

    _make_matrix(tmp_path)
    _make_cell(tmp_path, "cdxlstm_gzcd", raw_clean_f1=0.0)
    with pytest.raises(SystemExit, match="numerically invalid"):
        freeze_matrix(tmp_path, MATRIX)


def test_freeze_refuses_a_failed_instrument_guard(tmp_path: Path) -> None:
    _make_matrix(tmp_path)
    _make_cell(tmp_path, "cdlamba_dsifn", guard_passed=False)
    with pytest.raises(SystemExit, match="instrument guard"):
        freeze_matrix(tmp_path, MATRIX)


def test_freeze_refuses_an_unfinished_cell(tmp_path: Path) -> None:
    _make_matrix(tmp_path)
    (tmp_path / "reports/p21/cells/cdlamba_gzcd/LATEST").unlink()
    with pytest.raises(SystemExit, match="cdlamba_gzcd"):
        freeze_matrix(tmp_path, MATRIX)


def test_freeze_refuses_a_checkpoint_that_differs_from_its_cell_record(tmp_path: Path) -> None:
    _make_matrix(tmp_path)
    (tmp_path / f"reports/p21/cells/bit_levir/{TAG}/dual_path.pth").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="dual_path.pth"):
        freeze_matrix(tmp_path, MATRIX)


def test_inputs_changed_after_freezing_are_refused(tmp_path: Path) -> None:
    _make_matrix(tmp_path)
    manifest = freeze_matrix(tmp_path, MATRIX)
    classifier = tmp_path / f"reports/p21/cells/cdlamba_levir/{TAG}" / CLASSIFIER_FILES["safe_alignment_task"]
    classifier.write_bytes(b"retrained after freezing")
    with pytest.raises(SystemExit, match="safe_alignment_task"):
        verified_test_inputs(manifest, "CD-Lamba", project_root=tmp_path)


def test_frozen_manifest_is_written_once(tmp_path: Path) -> None:
    path = tmp_path / "reports/p21/VALIDATION_FROZEN.json"
    write_frozen_manifest(path, {"status": "frozen"})
    with pytest.raises(SystemExit, match="frozen once"):
        write_frozen_manifest(path, {"status": "frozen"})


def test_test_marker_is_one_shot(tmp_path: Path) -> None:
    marker = tmp_path / "reports/p21_final_test/bit_LEVIR_TEST_CONSUMED.json"
    claim_test_marker(marker, detector="BIT", output_dir=tmp_path / "first")
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record["status"] == "started_irreversible"
    assert record["allowed_reruns"] == 0

    with pytest.raises(SystemExit, match="already started"):
        claim_test_marker(marker, detector="BIT", output_dir=tmp_path / "second")
    complete_test_marker(marker, tmp_path / "first" / "final_test_summary.json")
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "completed"
    with pytest.raises(SystemExit):
        claim_test_marker(marker, detector="BIT", output_dir=tmp_path / "third")


def test_run_script_checks_every_precondition_before_the_test() -> None:
    script = RUN_SCRIPT.read_text(encoding="utf-8")
    first_test_read = script.index("tools.evaluate_p21_final_test")
    for guard in (
        'grep -q "修订 D"',
        "inference_tile_size",
        "VALIDATION_FROZEN.json",
        "_LEVIR_TEST_CONSUMED.json",
        "python -m pytest",
    ):
        assert -1 < script.index(guard) < first_test_read, guard
    for pair in ('"BIT bit"', '"CD-Lamba cdlamba"', '"CDxLSTM cdxlstm"'):
        assert pair in script
