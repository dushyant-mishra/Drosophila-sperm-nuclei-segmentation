import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_v571_width_threshold_sensitivity.py"
    spec = importlib.util.spec_from_file_location(
        "v571_width_threshold_sensitivity_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_thresholds_and_core_preserve_bounded_scope():
    validator = load_validator()

    assert validator.FOREGROUND_THRESHOLDS == (0.55, 0.60, 0.65)
    assert validator.BASELINE_FOREGROUND_THRESHOLD == pytest.approx(0.60)
    assert validator.CORE_THRESHOLD == pytest.approx(0.50)
    assert validator.SPECIMENS == {
        "KJ-01": "kj_sv_40xx0.75-1",
        "WT-01": "w1118_sv_feb_40xx0.75-1",
    }
    assert validator.EXPECTED_BASELINE_INFERENCE_CALLS == 6
    assert validator.EXPECTED_REPEAT_INFERENCE_CALLS == 1
    assert validator.EXPECTED_TOTAL_INFERENCE_CALLS == 7


def test_max_iou_match_is_deterministic_and_prefers_smallest_label_on_tie():
    validator = load_validator()
    baseline = np.zeros((4, 6), dtype=bool)
    baseline[1:3, 1:5] = True
    labels = np.zeros_like(baseline, dtype=np.int32)
    labels[1:3, 1:3] = 9
    labels[1:3, 3:5] = 4

    match = validator.match_label_by_max_iou(labels, baseline)

    assert match["matched_label"] == 4
    assert match["intersection_px"] == 4
    assert match["union_px"] == 8
    assert match["iou"] == pytest.approx(0.5)


def test_max_iou_match_reports_no_overlap_without_inventing_instance():
    validator = load_validator()
    baseline = np.zeros((5, 5), dtype=bool)
    baseline[1:3, 1:3] = True
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[3:5, 3:5] = 2

    match = validator.match_label_by_max_iou(labels, baseline)

    assert match == {
        "matched_label": 0,
        "iou": 0.0,
        "intersection_px": 0,
        "union_px": 4,
    }


def test_sensitivity_row_reports_changes_relative_to_baseline():
    validator = load_validator()
    row = validator.build_sensitivity_row(
        threshold=0.55,
        core_threshold=0.50,
        match={
            "matched_label": 7,
            "iou": 0.8,
            "intersection_px": 80,
            "union_px": 100,
        },
        baseline_area_px=80,
        matched_area_px=100,
        baseline_width_um=2.0,
        matched_width_um=2.2,
    )

    assert row["mask_area_change_px"] == pytest.approx(20)
    assert row["mask_area_change_percent"] == pytest.approx(25)
    assert row["body_width_change_um"] == pytest.approx(0.2)
    assert row["body_width_change_percent"] == pytest.approx(10)
    assert row["match_found"] is True
    assert row["matched_instance_id"] == 7


def test_relative_change_is_nan_for_unavailable_or_zero_baseline():
    validator = load_validator()

    assert np.isnan(validator.relative_change_percent(1.0, 0.0))
    assert np.isnan(validator.relative_change_percent(np.nan, 2.0))
    assert np.isnan(validator.relative_change_percent(2.0, np.nan))
    assert validator.max_abs_finite([-2.0, np.nan, 1.0]) == pytest.approx(2.0)
    assert np.isnan(validator.max_abs_finite([np.nan, None]))


def test_no_match_row_is_explicit_and_keeps_zero_area():
    validator = load_validator()
    row = validator.build_sensitivity_row(
        threshold=0.65,
        core_threshold=0.50,
        match={
            "matched_label": 0,
            "iou": 0.0,
            "intersection_px": 0,
            "union_px": 20,
        },
        baseline_area_px=20,
        matched_area_px=0,
        baseline_width_um=2.0,
        matched_width_um=np.nan,
    )

    assert row["match_found"] is False
    assert row["matched_mask_area_px"] == 0
    assert row["mask_area_change_percent"] == pytest.approx(-100)
    assert np.isnan(row["body_width_change_percent"])


def test_three_plane_selection_requires_consecutive_representative_planes():
    validator = load_validator()
    detections = pd.DataFrame(
        {
            "track_id": [4, 4, 4, 4, 8],
            "z_slice": [36, 34, 35, 40, 35],
            "sperm_id": [2, 2, 2, 3, 1],
        }
    )

    selected = validator.select_three_observations(detections, 4, 35)
    assert selected["z_slice"].tolist() == [34, 35, 36]

    with pytest.raises(ValueError, match="required consecutive planes"):
        validator.select_three_observations(
            detections[detections["z_slice"] != 34], 4, 35
        )


def test_array_hash_includes_shape_and_dtype():
    validator = load_validator()
    base = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert validator.sha256_array(base) == validator.sha256_array(base.copy())
    assert validator.sha256_array(base) != validator.sha256_array(base.reshape(3, 2))
    assert validator.sha256_array(base) != validator.sha256_array(base.astype(np.float64))


def test_specimen_provenance_binds_stack_roi_metadata_and_calibration(
    tmp_path, monkeypatch
):
    validator = load_validator()
    settings = tmp_path / "settings"
    settings.mkdir()
    roi = settings / "roi.npy"
    roi.write_bytes(b"roi")
    for name, content in (
        ("source_image_manifest.json", b"{}"),
        ("microscope_metadata_used.xml", b"<xml />"),
        ("calibration_used.json", b"{}"),
    ):
        (settings / name).write_bytes(content)
    image0 = tmp_path / "z0.tif"
    image1 = tmp_path / "z1.tif"
    image0.write_bytes(b"z0")
    image1.write_bytes(b"z1")
    monkeypatch.setattr(validator, "retained_settings_dir", lambda _stem: settings)

    result = validator.build_specimen_provenance(
        "Specimen-01",
        "stem",
        {0: str(image0), 1: str(image1)},
        roi,
        {"UM_PER_PX_XY": 0.4, "UM_PER_SLICE_Z": 1.2},
    )

    assert result["roi_sha256"] == validator.sha256_file(roi)
    assert result["microscope_metadata_sha256"] == validator.sha256_file(
        settings / "microscope_metadata_used.xml"
    )
    assert result["resolved_xy_um_per_pixel"] == pytest.approx(0.4)
    assert result["resolved_z_um_per_slice"] == pytest.approx(1.2)
    assert result["stack_preprocess_source_count"] == 2
    assert [row["z_index"] for row in result["stack_preprocess_sources"]] == [0, 1]


def test_repeated_dual_head_arrays_must_be_bit_identical():
    validator = load_validator()
    foreground = np.arange(9, dtype=np.float32).reshape(3, 3)
    core = foreground / 10.0

    record = validator.verify_repeated_probabilities(
        foreground, core, foreground.copy(), core.copy()
    )
    assert record["foreground_identical"] is True
    assert record["core_identical"] is True
    assert record["foreground_sha256"] == record["repeated_foreground_sha256"]
    assert record["core_sha256"] == record["repeated_core_sha256"]

    changed = foreground.copy()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="non-identical dual-head arrays"):
        validator.verify_repeated_probabilities(
            foreground, core, changed, core.copy()
        )


def test_deterministic_torch_configuration_is_enforced(monkeypatch):
    validator = load_validator()
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        manual_seed_all=lambda seed: calls.append(("cuda_seed", seed)),
    )
    cudnn = SimpleNamespace(benchmark=True, deterministic=False)
    fake_torch = SimpleNamespace(
        cuda=cuda,
        backends=SimpleNamespace(cudnn=cudnn),
        manual_seed=lambda seed: calls.append(("seed", seed)),
        use_deterministic_algorithms=lambda enabled: calls.append(
            ("algorithms", enabled)
        ),
        set_deterministic_debug_mode=lambda mode: calls.append(("debug", mode)),
    )

    settings = validator.configure_deterministic_torch(fake_torch)

    assert ("seed", 0) in calls
    assert ("cuda_seed", 0) in calls
    assert ("algorithms", True) in calls
    assert ("debug", "error") in calls
    assert cudnn.benchmark is False
    assert cudnn.deterministic is True
    assert settings["deterministic_algorithms"] is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_threshold_validator_fails_closed_for_existing_output(tmp_path):
    validator = load_validator()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validator.prepare_output_dir(existing)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    old = occupied / "old.csv"
    old.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        validator.prepare_output_dir(occupied)
    assert old.read_text(encoding="utf-8") == "do not replace"


def test_atomic_writer_leaves_no_temporary_file(tmp_path):
    validator = load_validator()
    destination = tmp_path / "artifact.json"
    validator.atomic_write_text(destination, '{"status":"complete"}')
    assert destination.read_text(encoding="utf-8") == '{"status":"complete"}'
    assert not list(tmp_path.glob(".*.tmp"))
