import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_audit_blockers",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_length_alone_above_20_um_is_review_not_technical_veto():
    saturn = load_saturn()
    mask = np.zeros((32, 96), dtype=bool)
    mask[13:18, 10:80] = True
    seg = {
        "unet_primary_instance_labels": mask.astype(np.int32),
        "unet_primary_centerline_labels": skeletonize(mask).astype(np.int32),
        "unet_probability": np.where(mask, 0.98, 0.0).astype(np.float32),
        "unet_primary_parent_by_instance": {1: 1},
        "unet_primary_instance_sources": {1: "unet_primary"},
        "unet_primary_centerline_metadata": {1: {}},
        "unet_primary_technical_failures": [],
    }
    cfg = saturn.cfg_with_resolved_pixels(
        {
            **saturn.CONFIG,
            "UM_PER_PX_XY": 0.4,
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_PRIMARY_OVERLONG_SPLIT_ENABLE": False,
            "MAX_GEODESIC_LEN_UM": 20.0,
            "MAX_WIDTH_UM": 20.0,
        }
    )
    result = saturn._measure_unet_primary_instances(seg, cfg)["results"][0]
    assert result["over_20_um_review"] is True
    assert result["suspected_multi_object_merge"] is False

    tracks = pd.DataFrame(
        {
            "track_id": [1],
            "centroid_x": [20.0],
            "centroid_y": [10.0],
            "total_3d_length_um": [25.0],
            "suspected_multi_object_merge": [False],
        }
    )
    audited = saturn.flag_quality_tracks(tracks, cfg)
    assert bool(audited.loc[0, "technical_valid"])
    assert "long" in audited.loc[0, "morphology_warning_reasons"]


def test_over_20_um_branched_component_is_a_technical_merge_failure():
    saturn = load_saturn()
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:51, 10:82] = True
    mask[18:49, 45:50] = True
    centerline = skeletonize(mask)
    seg = {
        "unet_primary_instance_labels": mask.astype(np.int32),
        "unet_primary_centerline_labels": centerline.astype(np.int32),
        "unet_probability": np.where(mask, 0.98, 0.0).astype(np.float32),
        "unet_primary_parent_by_instance": {1: 1},
        "unet_primary_instance_sources": {1: "unet_primary"},
        "unet_primary_centerline_metadata": {1: {"raw_branch_count": 1}},
        "unet_primary_technical_failures": [],
    }
    cfg = saturn.cfg_with_resolved_pixels(
        {
            **saturn.CONFIG,
            "UM_PER_PX_XY": 0.4,
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_PRIMARY_OVERLONG_SPLIT_ENABLE": False,
            "MAX_GEODESIC_LEN_UM": 20.0,
            "MAX_WIDTH_UM": 20.0,
        }
    )
    result = saturn._measure_unet_primary_instances(seg, cfg)["results"][0]
    assert result["over_20_um_review"] is True
    assert result["suspected_multi_object_merge"] is True


def test_centroid_path_tortuosity_uses_one_anisotropic_3d_path():
    saturn = load_saturn()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "sperm_id": [1, 1, 1],
            "z_slice": [0, 1, 2],
            "centroid_x": [0.0, 3.0, 3.0],
            "centroid_y": [0.0, 0.0, 4.0],
        }
    )
    metrics = saturn._track_centroid_path_metrics(
        detections,
        {"UM_PER_PX_XY": 2.0, "UM_PER_SLICE_Z": 3.0},
    ).iloc[0]
    expected_path = np.hypot(6.0, 3.0) + np.sqrt(8.0**2 + 3.0**2)
    expected_end = np.sqrt(6.0**2 + 8.0**2 + 6.0**2)
    assert metrics["centroid_path_length_3d_um"] == pytest.approx(expected_path)
    assert metrics["centroid_end_to_end_3d_um"] == pytest.approx(expected_end)
    assert metrics["centroid_path_tortuosity_3d"] == pytest.approx(
        expected_path / expected_end
    )


def test_gapped_track_records_missing_slice_without_volume_interpolation():
    saturn = load_saturn()
    detections = pd.DataFrame(
        {
            "track_id": [7, 7],
            "sperm_id": [1, 1],
            "z_slice": [0, 2],
            "centroid_x": [2.0, 2.0],
            "centroid_y": [3.0, 3.0],
        }
    )
    metrics = saturn._track_centroid_path_metrics(
        detections,
        {"UM_PER_PX_XY": 1.0, "UM_PER_SLICE_Z": 0.5},
    ).iloc[0]
    assert metrics["observed_slice_count"] == 2
    assert metrics["missing_slice_count"] == 1


def test_track_summary_uses_filled_area_and_explicit_path_geometry():
    saturn = load_saturn()
    detections = pd.DataFrame(
        {
            "track_id": [3, 3],
            "sperm_id": [1, 1],
            "z_slice": [0, 2],
            "centroid_x": [0.0, 3.0],
            "centroid_y": [0.0, 4.0],
            "length_um_geodesic": [8.0, 10.0],
            "tortuosity": [1.0, 1.0],
            "width_um": [2.0, 2.0],
            "length_width_ratio": [4.0, 5.0],
            "area_px": [16.0, 20.0],
            "instance_mask_area_px": [30.0, 40.0],
            "suspected_multi_object_merge": [False, False],
        }
    )
    cfg = {
        **saturn.CONFIG,
        "UM_PER_PX_XY": 2.0,
        "UM_PER_SLICE_Z": 0.5,
    }
    _, summary = saturn._summarize_tracked_detections(detections, {}, cfg)
    track = summary.iloc[0]
    assert track["total_3d_length_um"] == pytest.approx(np.hypot(10.0, 1.0))
    assert track["observed_slice_mask_volume_um3"] == pytest.approx(
        (30.0 + 40.0) * 2.0**2 * 0.5
    )
    assert track["volume_um3"] == track["observed_slice_mask_volume_um3"]
    assert track["missing_slice_count"] == 1
    assert track["tortuosity_3d_method"] == "ordered_calibrated_centroid_path"


def test_settings_bundle_hashes_sources_roi_and_metadata(tmp_path):
    saturn = load_saturn()
    source = tmp_path / "Project001_Series001_z00_ch00.tif"
    roi = tmp_path / "roi.npy"
    metadata = tmp_path / "Project001_Series001.xml"
    source.write_bytes(b"source pixels")
    np.save(roi, np.ones((2, 2), dtype=bool))
    metadata.write_text("<xml/>", encoding="utf-8")
    cfg = {
        **saturn.CONFIG,
        "SEGMENTATION_ENGINE": "classical_saturn",
        "ROI_MASK_PATH": str(roi),
        "CALIBRATION_METADATA_FILE": str(metadata),
        "_SOURCE_IMAGE_FILES": [str(source)],
    }
    bundle = saturn.save_analysis_settings_bundle(tmp_path / "out", cfg)
    manifest = json.loads(Path(bundle["manifest"]).read_text(encoding="utf-8"))
    roles = {record["role"] for record in manifest["files"]}
    assert {"source_image_manifest", "roi_mask_source", "microscope_metadata_xml"} <= roles
    source_manifest = json.loads(
        (Path(bundle["settings_dir"]) / "source_image_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(source_manifest["ordered_source_images"][0]["sha256"]) == 64


def test_same_path_checkpoint_replacement_changes_cache_identity(tmp_path):
    from utils import saturn_unet25d_bridge as bridge

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"first")
    first = bridge._checkpoint_sha256(checkpoint)
    checkpoint.write_bytes(b"second")
    second = bridge._checkpoint_sha256(checkpoint)
    assert first != second


def test_default_production_profile_is_fail_closed_and_versioned():
    saturn = load_saturn()
    cfg = saturn.CONFIG.copy()
    cfg.pop("_ACTIVE_PROFILE_PATH", None)
    saturn.activate_default_production_profile(cfg)
    assert cfg["SEGMENTATION_ENGINE"] == "unet_primary"
    assert cfg["UNET_OUTPUT_MODE"] == "dual_head"
    assert cfg["TRACKING_BACKEND"] == "global_assignment"
    assert cfg["UNET_PRIMARY_OVERLONG_SPLIT_ENABLE"] is True
    assert cfg["UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM"] == pytest.approx(20.0)
    assert cfg["REQUIRE_LEICA_METADATA"] is True
    assert Path(cfg["_ACTIVE_PROFILE_PATH"]).name == (
        "saturn_v5_7_1_model_c_epoch003.json"
    )


def test_leica_named_stack_cannot_fall_back_when_metadata_required(tmp_path):
    saturn = load_saturn()
    source = tmp_path / "Project001_Series001_z00_ch00.tif"
    source.write_bytes(b"placeholder")
    cfg = {**saturn.CONFIG, "REQUIRE_LEICA_METADATA": True}
    with pytest.raises(ValueError, match="Leica XML calibration unavailable"):
        saturn.resolve_stack_microscope_calibration(
            cfg,
            [source],
            input_dir=tmp_path,
        )


def test_v571_study_and_tuner_gui_select_v571_implementations():
    study_source = (ROOT / "scripts" / "run_v571_study.py").read_text(
        encoding="utf-8"
    )
    tuner_gui_source = (
        ROOT / "utils" / "tuner_gui_Saturnv5_7_1.py"
    ).read_text(encoding="utf-8")
    assert 'sperm_segmentation_saturnv5.7.1.py' in study_source
    assert 'saturn_v5_7_1_model_c_epoch003.json' in study_source
    assert 'tune_parameters_Saturnv5_7_1.py' in tuner_gui_source


def test_primary_comparison_fields_are_in_biological_specimen_table():
    source = (ROOT / "sperm_segmentation_saturnv5.7.1.py").read_text(
        encoding="utf-8"
    )
    biological_block = source.split("biological_columns = [", 1)[1].split("]", 1)[0]
    for field in (
        "median_body_width_um",
        "median_body_width_p90_um",
        "median_area_length_width_um",
        "median_length_body_width_ratio",
    ):
        assert field in biological_block
