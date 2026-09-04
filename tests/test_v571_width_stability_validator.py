import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_v571_body_width.py"
    spec = importlib.util.spec_from_file_location(
        "v571_width_stability_validator_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_contract_is_known_width_and_rotation_stable():
    validator = load_validator()
    saturn = validator.load_pipeline()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "BODY_WIDTH_ENABLE": True,
            "BODY_WIDTH_ENDPOINT_TRIM_FRACTION": 0.125,
            "BODY_WIDTH_SAMPLE_SPACING_PX": 1.0,
            "BODY_WIDTH_SMOOTH_SIGMA_PX": 1.0,
            "BODY_WIDTH_MIN_SAMPLES": 5,
        }
    )

    frame, summary = validator.evaluate_synthetic_geometry(saturn, cfg)

    assert len(frame) == 15
    assert set(frame["expected_width_px"]) == {5.0, 9.0, 13.0}
    assert set(frame["angle_deg"]) == {0, 20, 45, 70, 90}
    assert summary["all_cases_measured"] is True
    assert summary["maximum_absolute_error_px"] <= 1.0
    assert summary["maximum_rotation_spread_px"] <= 1.0


def test_engineering_verdict_does_not_claim_biological_accuracy():
    validator = load_validator()
    synthetic = {
        "maximum_absolute_error_px": 0.4,
        "maximum_rotation_spread_px": 0.5,
    }
    rasterized = pd.DataFrame({"status": ["measured"] * 20})
    measured = pd.DataFrame(
        {
            "mask_dilate1_width_delta_px": np.full(20, 1.8),
            "mask_erode1_width_delta_px": np.full(20, -1.7),
        }
    )

    criteria, verdict = validator.engineering_verdict(
        synthetic,
        rasterized,
        measured,
    )

    assert verdict == "pass"
    assert all(criteria.values())
    assert "biological" not in " ".join(criteria).lower()


def test_validator_exports_one_user_field_and_explicit_limit(tmp_path):
    validator = load_validator()
    coco = {
        "images": [
            {"id": 1, "file_name": "synthetic.png", "width": 96, "height": 64}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [15, 25, 60, 9],
                "segmentation": [[15, 25, 75, 25, 75, 34, 15, 34]],
            }
        ],
    }
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    summary = validator.evaluate(coco_path, tmp_path / "result")

    assert summary["primary_user_field"] == "representative_body_width_um"
    assert summary["primary_measurement_label"] == (
        "apparent central-body mask width"
    )
    assert summary["absolute_biological_accuracy_status"] == "not_established"
    assert summary["engineering_validation_status"] == "pass"
    report = (tmp_path / "result" / "V5_7_1_BODY_WIDTH_VALIDATION.md").read_text(
        encoding="utf-8"
    )
    assert "Alternate width calculations remain technical QC" in report
    assert "PSF-corrected or molecular diameter" in report
    assert "train_mask_dilate_px: 0" in report
    assert (tmp_path / "result" / "synthetic_width_validation.csv").is_file()
    decision = pd.read_csv(tmp_path / "result" / "width_measurement_decision.csv")
    assert decision.loc[0, "primary_user_field"] == (
        "representative_body_width_um"
    )
    assert decision.loc[0, "absolute_biological_accuracy_status"] == (
        "not_established"
    )
    manifest = json.loads(
        (tmp_path / "result" / "evidence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["claim_id"] == "MEAS-BODY-WIDTH-001"
    assert manifest["absolute_biological_accuracy_status"] == "not_established"
    assert manifest["generator_sha256"]
    assert manifest["generated_at_utc"]
    assert manifest["environment"]["python_version"]
    assert {item["path"] for item in manifest["artifacts"]} == {
        "coco_mask_width_validation.csv",
        "coco_mask_width_validation.json",
        "synthetic_width_validation.csv",
        "width_measurement_decision.csv",
        "V5_7_1_BODY_WIDTH_VALIDATION.md",
        "coco_mask_width_validation.png",
    }
    completion = json.loads(
        (tmp_path / "result" / "COMPLETED.json").read_text(encoding="utf-8")
    )
    manifest_bytes = (tmp_path / "result" / "evidence_manifest.json").read_bytes()
    assert completion["status"] == "complete"
    assert completion["qc_only"] is True
    assert completion["manifest_sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    assert not list((tmp_path / "result").glob(".*.tmp"))


def test_validator_fails_closed_for_existing_output_directory(tmp_path):
    validator = load_validator()
    existing = tmp_path / "existing"
    existing.mkdir()

    try:
        validator.prepare_output_dir(existing)
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Existing output directory was accepted")

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.txt").write_text("old evidence", encoding="utf-8")
    try:
        validator.prepare_output_dir(occupied)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Nonempty output directory was accepted")
    assert (occupied / "old.txt").read_text(encoding="utf-8") == "old evidence"
