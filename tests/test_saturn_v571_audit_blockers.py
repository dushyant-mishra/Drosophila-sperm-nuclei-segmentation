import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.morphology import skeletonize
import tifffile


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
            "projection_z_extent_um": [25.0],
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


@pytest.mark.parametrize("shape_kind", ["straight", "wide", "curved", "thin"])
def test_long_single_core_morphologies_remain_one_instance(shape_kind):
    saturn = load_saturn()
    mask = np.zeros((80, 100), dtype=bool)
    if shape_kind == "straight":
        mask[37:43, 10:90] = True
    elif shape_kind == "wide":
        mask[33:47, 10:90] = True
    elif shape_kind == "thin":
        mask[39:42, 10:90] = True
    else:
        for x in range(10, 90):
            y = int(round(40 + 12 * np.sin((x - 10) / 80 * np.pi)))
            mask[max(0, y - 3):min(mask.shape[0], y + 4), x] = True
    probability = np.where(mask, 0.9, 0.0).astype(np.float32)
    core_probability = np.zeros_like(probability)
    core_probability[mask] = 0.8
    labels = saturn.measure.label(mask).astype(np.int32)
    cfg = saturn.cfg_with_resolved_pixels({
        **saturn.CONFIG,
        "UM_PER_PX_XY": 0.5,
        "UNET_PRIMARY_OVERLONG_SPLIT_ENABLE": True,
        "UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM": 20.0,
    })

    refined, _, audit = saturn._refine_overlong_unet_instances(
        probability,
        labels,
        {1: 1},
        cfg,
        core_probability=core_probability,
    )

    assert int(refined.max()) == 1
    assert np.array_equal(refined > 0, mask)
    assert audit
    assert audit[0]["objective_core_marker_count"] == 1


def test_connected_core_peaks_split_overlong_component_without_length_only_rule():
    saturn = load_saturn()
    mask = np.zeros((48, 100), dtype=bool)
    mask[20:27, 10:90] = True
    probability = np.where(mask, 0.9, 0.0).astype(np.float32)
    x = np.arange(mask.shape[1], dtype=np.float32)
    core_profile = (
        0.55
        + 0.40 * np.exp(-0.5 * ((x - 30.0) / 4.0) ** 2)
        + 0.40 * np.exp(-0.5 * ((x - 70.0) / 4.0) ** 2)
    )
    core_probability = np.zeros_like(probability)
    core_probability[mask] = np.broadcast_to(
        core_profile, mask.shape
    )[mask]
    labels = saturn.measure.label(mask).astype(np.int32)
    cfg = saturn.cfg_with_resolved_pixels({
        **saturn.CONFIG,
        "UM_PER_PX_XY": 0.5,
        "UNET_PRIMARY_OVERLONG_SPLIT_ENABLE": True,
        "UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM": 20.0,
        "UNET_PRIMARY_OVERLONG_CORE_PEAK_PROMINENCE": 0.12,
        "UNET_PRIMARY_OVERLONG_CORE_PEAK_MIN_DISTANCE_UM": 4.0,
    })

    refined, _, audit = saturn._refine_overlong_unet_instances(
        probability,
        labels,
        {1: 1},
        cfg,
        core_probability=core_probability,
    )

    assert int(refined.max()) == 2
    assert np.array_equal(refined > 0, mask)
    assert audit[0]["disposition"] == "overlong_watershed_split"
    assert audit[0]["split_evidence"] == "separated_learned_core_peaks"


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
    assert track["projection_z_extent_um"] == pytest.approx(np.hypot(10.0, 1.0))
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
    tifffile.imwrite(source, np.arange(4, dtype=np.uint16).reshape(2, 2))
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
    assert {
        "source_image_manifest",
        "roi_mask_source",
        "microscope_metadata_xml",
        "runtime_environment",
    } <= roles
    source_manifest = json.loads(
        (Path(bundle["settings_dir"]) / "source_image_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_record = source_manifest["ordered_source_images"][0]
    assert len(source_record["sha256"]) == 64
    assert source_record["shape"] == [2, 2]
    runtime_environment = json.loads(
        (Path(bundle["settings_dir"]) / "runtime_environment.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime_environment["requested_unet_device"] == "auto"
    assert runtime_environment["resolved_unet_device"] in {
        "cpu",
        "cuda",
        "mps",
    }
    assert runtime_environment["deterministic_unet_inference"] is True


def test_manifest_locked_calibration_supports_organized_filenames(tmp_path):
    saturn = load_saturn()
    source = tmp_path / "KJ-01_z0000_ch00.tif"
    tifffile.imwrite(source, np.zeros((8, 8), dtype=np.uint16))
    metadata = tmp_path / "Project001_Series013.xml"
    metadata.write_text(
        """<Root><DimensionDescription DimID="1" NumberOfElements="8"
        Length="0.00000304" Unit="m"/><DimensionDescription DimID="2"
        NumberOfElements="8" Length="0.00000304" Unit="m"/>
        <DimensionDescription DimID="3" NumberOfElements="3"
        Length="0.00000104" Unit="m"/></Root>""",
        encoding="utf-8",
    )
    cfg = {
        **saturn.CONFIG,
        "AUTO_LEICA_CALIBRATION": True,
        "REQUIRE_LEICA_METADATA": True,
        "UM_PER_PX_XY": 0.38,
        "UM_PER_SLICE_Z": 0.52,
        "CALIBRATION_METADATA_FILE": str(metadata),
        "_CALIBRATION_METADATA_SHA256": saturn._sha256_file(metadata),
        "_CALIBRATION_LOCKED_FROM_MANIFEST": True,
        "_CALIBRATION_PROVENANCE": {
            "acquisition_class": "Leica 40x test",
        },
    }

    provenance = saturn.resolve_stack_microscope_calibration(
        cfg, [source], input_dir=tmp_path, require_metadata=True
    )

    assert provenance["status"] == "leica_xml_manifest_locked"
    assert cfg["CALIBRATION_SOURCE"] == "leica_metadata_xml"
    assert cfg["UM_PER_PX_XY"] == pytest.approx(0.38)
    assert cfg["UM_PER_SLICE_Z"] == pytest.approx(0.52)


def test_manifest_locked_calibration_rejects_replaced_or_invalid_xml(tmp_path):
    saturn = load_saturn()
    source = tmp_path / "KJ-01_z0000_ch00.tif"
    metadata = tmp_path / "Project001_Series013.xml"
    tifffile.imwrite(source, np.zeros((8, 8), dtype=np.uint16))
    metadata.write_text("<xml/>", encoding="utf-8")
    cfg = {
        **saturn.CONFIG,
        "UM_PER_PX_XY": 0.38,
        "UM_PER_SLICE_Z": 0.52,
        "CALIBRATION_METADATA_FILE": str(metadata),
        "_CALIBRATION_METADATA_SHA256": saturn._sha256_file(metadata),
        "_CALIBRATION_LOCKED_FROM_MANIFEST": True,
    }
    with pytest.raises(ValueError, match="complete X/Y/Z calibration"):
        saturn.resolve_stack_microscope_calibration(cfg, [source], tmp_path, True)

    metadata.write_text("<changed/>", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        saturn.resolve_stack_microscope_calibration(cfg, [source], tmp_path, True)


def test_batch_discovery_fails_closed_on_mixed_stack_or_channel(tmp_path):
    saturn = load_saturn()
    for name in (
        "sampleA_z0000_ch00.tif",
        "sampleB_z0001_ch00.tif",
    ):
        tifffile.imwrite(tmp_path / name, np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="multiple stack identities"):
        saturn.load_batch_files(tmp_path, "*.tif")

    for path in tmp_path.glob("*.tif"):
        path.unlink()
    tifffile.imwrite(
        tmp_path / "sampleA_z0000_ch01.tif", np.zeros((4, 4), dtype=np.uint16)
    )
    with pytest.raises(ValueError, match="Unparseable source-image identity"):
        saturn.load_batch_files(tmp_path, "*.tif")


def test_comparative_tracking_does_not_veto_length_alone():
    saturn = load_saturn()
    comparative = {**saturn.CONFIG, "ANALYSIS_MODE": "comparative"}
    noncomparative = {**saturn.CONFIG, "ANALYSIS_MODE": "descriptive"}

    assert saturn._tracking_max_joined_length_um(comparative) is None
    assert saturn._tracking_max_joined_length_um(noncomparative) == pytest.approx(
        saturn.CONFIG["TRACK_TECHNICAL_MAX_JOINED_LENGTH_UM"]
    )


def test_tracking_orientation_difference_converts_regionprops_radians():
    saturn = load_saturn()
    assert saturn._angle_diff_deg(0.0, np.pi / 2) == pytest.approx(90.0)
    assert saturn._angle_diff_deg(-np.pi / 2, np.pi / 2) == pytest.approx(0.0)
    assert saturn._orientation_difference_degrees(
        0.0, np.pi / 4
    ) == pytest.approx(45.0)


def test_comparative_tracking_rejects_identity_hop_without_morphology_gate():
    saturn = load_saturn()
    rows = []
    for z, sperm_id, x, y, orientation in (
        (0, 1, 0.0, 0.0, 0.0),
        (1, 1, 0.0, 0.5, np.pi / 2),
        (1, 2, 1.0, 0.0, 0.0),
    ):
        rows.append(
            {
                "z_slice": z,
                "sperm_id": sperm_id,
                "centroid_x": x,
                "centroid_y": y,
                "length_um_geodesic": 10.0,
                "length_px_geodesic": 10.0,
                "width_um": 1.0,
                "length_width_ratio": 10.0,
                "orientation": orientation,
                "area_px": 20.0,
                "bbox_min_y": y - 1.0,
                "bbox_min_x": x - 1.0,
                "bbox_max_y": y + 1.0,
                "bbox_max_x": x + 1.0,
            }
        )
    cfg = {
        **saturn.CONFIG,
        "ANALYSIS_MODE": "comparative",
        "UM_PER_PX_XY": 1.0,
        "UM_PER_SLICE_Z": 1.0,
        "TRACK_MAX_DIST_UM": 3.0,
        "TRACK_TECHNICAL_MAX_ADJACENT_DISPLACEMENT_UM": 2.0,
        "TRACK_TECHNICAL_MAX_ORIENTATION_CHANGE_DEG": 35.0,
        "TRACK_TECHNICAL_ORIENTATION_MIN_LENGTH_UM": 2.0,
        "ASSIGNMENT_MAX_COST": 10.0,
    }
    tracked, _tracks = saturn.track_across_slices_global_assignment(
        pd.DataFrame(rows), cfg
    )
    first_track = tracked.loc[tracked["z_slice"] == 0, "track_id"].iloc[0]
    linked = tracked[tracked["track_id"] == first_track]
    assert set(linked["sperm_id"]) == {1, 2}
    assert len(linked) == 2


def test_probability_cache_requires_matching_checkpoint_identity(tmp_path):
    saturn = load_saturn()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = saturn._sha256_file(checkpoint)
    cfg = {
        **saturn.CONFIG,
        "SEGMENTATION_ENGINE": "unet_primary",
        "UNET_MODEL_PATH": str(checkpoint),
        "UNET_CHECKPOINT_SHA256": checkpoint_sha,
        "_UNET_PROBABILITY_CACHE": {0: np.zeros((4, 4), dtype=np.float32)},
        "_UNET_PROBABILITY_CACHE_CHECKPOINT_SHA256": "0" * 64,
    }
    with pytest.raises(ValueError, match="not authenticated"):
        saturn.validate_analysis_runtime_config(cfg)

    cfg["_UNET_PROBABILITY_CACHE_CHECKPOINT_SHA256"] = checkpoint_sha
    status = saturn.validate_analysis_runtime_config(cfg)
    assert status["checkpoint_sha256"] == checkpoint_sha


def test_same_path_checkpoint_replacement_changes_cache_identity(tmp_path):
    from utils import saturn_unet25d_bridge as bridge

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"first")
    first = bridge._checkpoint_sha256(checkpoint)
    checkpoint.write_bytes(b"second")
    second = bridge._checkpoint_sha256(checkpoint)
    assert first != second


def test_production_checkpoint_cpu_inference_is_repeatable():
    from utils import saturn_unet25d_bridge as bridge

    checkpoint = ROOT / "model_checkpoints" / "v571_model_c_dual_head_epoch003.pt"
    context = np.linspace(0, 1, 3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
    roi = np.ones((32, 32), dtype=bool)
    cfg = {
        "UNET_DEVICE": "cpu",
        "UNET_DETERMINISTIC_INFERENCE": True,
        "UNET_TILE_SIZE": 32,
        "UNET_TILE_OVERLAP": 0,
        "UNET_TILE_BATCH_SIZE": 1,
        "UNET_ROI_PADDING_PX": 0,
        "UNET_OUTSIDE_ROI_ZERO": True,
    }

    first = bridge.predict_probability_heads_tiled(
        context, checkpoint, roi_mask=roi, cfg=cfg
    )
    second = bridge.predict_probability_heads_tiled(
        context, checkpoint, roi_mask=roi, cfg=cfg
    )

    assert set(first) == {"foreground", "core"}
    assert np.array_equal(first["foreground"], second["foreground"])
    assert np.array_equal(first["core"], second["core"])
    assert cfg["_UNET_RUNTIME_PROVENANCE"]["resolved_device"] == "cpu"
    assert cfg["_UNET_RUNTIME_PROVENANCE"]["deterministic_inference"] is True


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


def test_primary_specimen_export_excludes_unqualified_3d_aliases():
    source = (ROOT / "sperm_segmentation_saturnv5.7.1.py").read_text(
        encoding="utf-8"
    )
    biological_block = source.split("biological_columns = [", 1)[1].split(
        "]", 1
    )[0]
    for misleading_alias in (
        '"median_3d_length_um"',
        '"median_3d_thickness_um"',
        '"median_3d_volume_um3"',
    ):
        assert misleading_alias not in biological_block
    for explicit_field in (
        '"median_projection_z_extent_um"',
        '"median_observed_slab_effective_thickness_um"',
        '"median_observed_slice_mask_volume_um3"',
    ):
        assert explicit_field in biological_block


def test_evidence_git_blob_hash_is_checkout_line_ending_independent():
    import importlib.util
    import subprocess

    script_path = ROOT / "scripts" / "generate_v571_stage_evidence.py"
    spec = importlib.util.spec_from_file_location("stage_evidence_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    blob = subprocess.check_output(
        ["git", "show", f"HEAD:{profile.relative_to(ROOT).as_posix()}"], cwd=ROOT
    )
    assert module.git_blob_sha256(profile) == hashlib.sha256(blob).hexdigest()


def test_production_profile_uses_runtime_version_identifier():
    profile = json.loads(
        (ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json")
        .read_text(encoding="utf-8")
    )
    assert profile["_TUNING_METADATA"]["pipeline_version"] == "v5.7.1-body-width"


def test_v571_primary_summaries_use_explicit_projection_and_slab_names(tmp_path):
    saturn = load_saturn()
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2],
            "technical_valid": [True, True],
            "projection_z_extent_um": [8.0, 10.0],
            "observed_slab_effective_thickness_um": [1.4, 1.8],
            "observed_slice_mask_volume_um3": [20.0, 24.0],
            "max_length_2d": [7.5, 9.5],
            "tortuosity_3d": [1.0, 1.1],
            "z_span_um": [1.0, 2.0],
        }
    )
    summary = saturn.export_analysis_summary(
        tmp_path, pd.DataFrame(), tracks, cfg={}
    )
    assert summary["median_projection_z_extent_um"] == pytest.approx(9.0)
    assert summary["median_3d_length_um_legacy_alias"] == pytest.approx(9.0)
    payload = json.loads((tmp_path / "analysis_summary.json").read_text())
    assert payload["median_projection_z_extent_um"] == pytest.approx(9.0)
    assert "median_3d_length_um" not in payload
    assert "median_3d_length_um_legacy_alias" not in payload

    paths = saturn.export_biologist_results(tmp_path, tracks, "v571")
    exported = pd.read_csv(paths["summary"])
    assert exported.loc[0, "median_projection_z_extent_um"] == pytest.approx(9.0)
    assert "median_3d_length_um" not in exported
    assert exported.loc[0, "median_3d_length_um_legacy_alias"] == pytest.approx(9.0)


def test_v571_group_comparison_uses_explicit_projection_metric():
    saturn = load_saturn()
    frame = pd.DataFrame(
        {
            "sample_id": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "group": ["A", "A", "A", "B", "B", "B"],
            "status": ["complete"] * 6,
            "median_projection_z_extent_um": [8, 9, 10, 10, 11, 12],
            "median_3d_length_um_legacy_alias": [8, 9, 10, 10, 11, 12],
        }
    )
    comparisons, _qc = saturn._study_specimen_group_comparisons(
        frame, random_seed=7, bootstrap_resamples=50, permutation_resamples=99
    )
    assert "median_projection_z_extent_um" in set(comparisons["metric"])
    assert "median_3d_length_um_legacy_alias" not in set(comparisons["metric"])
