from pathlib import Path

from tools.audit_diffregcd_external import audit_source


def _fake_external(root: Path) -> None:
    for relative in (
        "model/cd_model_256.py",
        "config/levir_test.json",
        "data/CMUCDFlowDataset.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "model/cd_model_256.py").write_text(
        "\n".join(
            [
                "warped_feats_B = warp_features_all(self.feats_B, flow)",
                "self.pred_cm = self.netCD(self.feats_A, warped_feats_B)",
                "self.pred_cm = self.netCD(self.feats_A, self.feats_B)",
                "self.pred_corase_flow, self.logits = self.netReg(fA, fB)",
                "self.pred_flow_coarse.detach()",
            ]
        ),
        encoding="utf-8",
    )


def test_external_audit_documents_wiring_ambiguity(tmp_path: Path) -> None:
    _fake_external(tmp_path)
    result = audit_source(
        tmp_path,
        expected_commit="abc",
        actual_commit="abc",
    )
    assert result["status"] == "completed"
    assert result["inference_wiring_ambiguity_detected"] is True
    assert result["required_validation_paths"] == [
        "official_unmodified",
        "intended_warped",
    ]


def test_external_audit_rejects_commit_mismatch(tmp_path: Path) -> None:
    _fake_external(tmp_path)
    result = audit_source(
        tmp_path,
        expected_commit="abc",
        actual_commit="def",
    )
    assert result["status"] == "failed"
    assert result["commit_matches"] is False
