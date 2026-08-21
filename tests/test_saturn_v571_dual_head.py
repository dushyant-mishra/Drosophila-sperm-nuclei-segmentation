import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_dual_head_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tuner_v571():
    spec = importlib.util.spec_from_file_location(
        "tuner_v571_dual_head_test",
        ROOT / "utils" / "tune_parameters_Saturnv5_7_1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dual_head_cfg(module, **updates):
    cfg = module.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_OUTPUT_MODE": "dual_head",
            "UNET_FOREGROUND_THRESHOLD": 0.60,
            "UNET_CORE_THRESHOLD": 0.50,
            "UNET_PRIMARY_MIN_COMPONENT_PX": 3,
            "UNET_PRIMARY_OVERLONG_SPLIT_ENABLE": False,
            "UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE": False,
        }
    )
    cfg.update(updates)
    return cfg


def test_dual_head_core_markers_split_connected_foreground():
    saturn = load_saturn_v571()
    foreground = np.zeros((48, 64), dtype=np.float32)
    foreground[18:30, 10:54] = 0.90
    core = np.zeros_like(foreground)
    core[21:27, 16:22] = 0.95
    core[21:27, 42:48] = 0.95
    valid = np.ones(foreground.shape, dtype=bool)

    result = saturn._build_unet_primary_segmentation(
        foreground,
        valid,
        dual_head_cfg(saturn),
        core_probability=core,
    )

    labels = result["unet_primary_instance_labels"]
    assert int(labels.max()) == 2
    assert np.array_equal(result["mask_clean"], foreground >= 0.60)
    assert result["unet_primary_debug"]["instance_method"] == (
        "dual_head_core_marker_watershed"
    )
    assert np.array_equal(result["unet_core_probability"], core)


def test_dual_head_preserves_supported_component_without_core_marker():
    saturn = load_saturn_v571()
    foreground = np.zeros((32, 32), dtype=np.float32)
    foreground[10:20, 8:24] = 0.80
    core = np.zeros_like(foreground)

    result = saturn._build_unet_primary_segmentation(
        foreground,
        np.ones(foreground.shape, dtype=bool),
        dual_head_cfg(saturn),
        core_probability=core,
    )

    assert int(result["unet_primary_instance_labels"].max()) == 1
    assert np.count_nonzero(result["unet_instance_seed_mask"]) == 1


def test_dual_head_requires_core_probability():
    saturn = load_saturn_v571()
    probability = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="requires core_probability"):
        saturn._build_unet_primary_segmentation(
            probability,
            np.ones(probability.shape, dtype=bool),
            dual_head_cfg(saturn),
        )


def test_checkpoint_hash_mismatch_stops_before_analysis(tmp_path):
    saturn = load_saturn_v571()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not the selected checkpoint")
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_MODEL_PATH": str(checkpoint),
            "UNET_CHECKPOINT_SHA256": "0" * 64,
        }
    )

    with pytest.raises(ValueError, match="does not match the analysis profile"):
        saturn.validate_analysis_runtime_config(cfg)


def test_versioned_production_profile_selects_model_c_epoch_three():
    saturn = load_saturn_v571()
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    cfg, _ = saturn.load_analysis_profile(profile)
    runtime = saturn.validate_analysis_runtime_config(cfg)

    assert cfg["UNET_OUTPUT_MODE"] == "dual_head"
    assert cfg["UNET_FOREGROUND_THRESHOLD"] == pytest.approx(0.60)
    assert cfg["UNET_CORE_THRESHOLD"] == pytest.approx(0.50)
    assert cfg["TRACKING_BACKEND"] == "global_assignment"
    assert cfg["COMPARATIVE_TRACKING_MORPHOLOGY_NEUTRAL"] is True
    assert cfg["ASSIGNMENT_LENGTH_WEIGHT"] == 0.0
    assert cfg["ASSIGNMENT_WIDTH_WEIGHT"] == 0.0
    assert cfg["ASSIGNMENT_AREA_WEIGHT"] == 0.0
    assert cfg["TRACK_TECHNICAL_MAX_JOINED_LENGTH_UM"] == pytest.approx(20.0)
    assert Path(runtime["checkpoint_path"]).name == (
        "v571_model_c_dual_head_epoch003.pt"
    )
    assert runtime["checkpoint_sha256"] == cfg["UNET_CHECKPOINT_SHA256"]


def _tracking_rows(target_x=10.0, target_length=18.0, target_width=4.0):
    rows = []
    for z, x, length, width, area in (
        (0, 10.0, 2.0, 0.5, 4.0),
        (1, target_x, target_length, target_width, 80.0),
    ):
        rows.append(
            {
                "z_slice": z,
                "sperm_id": 1,
                "source_instance_key": f"z{z}:1",
                "centroid_x": x,
                "centroid_y": 10.0,
                "length_um_geodesic": length,
                "width_um": width,
                "area_px": area,
                "orientation": 0.0,
                "bbox_min_y": 8.0,
                "bbox_min_x": 8.0,
                "bbox_max_y": 12.0,
                "bbox_max_x": max(12.0, target_x + 2.0),
                "unet_mean_probability": 0.9,
                "unet_max_probability": 0.99,
            }
        )
    return pd.DataFrame(rows)


def test_comparative_assignment_does_not_veto_morphology_change():
    saturn = load_saturn_v571()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "ANALYSIS_MODE": "comparative",
            "SEGMENTATION_ENGINE": "unet_primary",
            "COMPARATIVE_TRACKING_MORPHOLOGY_NEUTRAL": True,
            "UM_PER_PX_XY": 0.1,
            "UM_PER_SLICE_Z": 0.1,
            "TRACK_MAX_DIST_UM": 2.0,
            "TRACK_MAX_GAP_SLICES": 0,
            "UNET_TRACK_MAX_RECONSTRUCTED_LENGTH_UM": 20.0,
        }
    )
    tracked, _ = saturn.track_across_slices_global_assignment(
        _tracking_rows(), cfg
    )
    assert tracked["track_id"].nunique() == 1


def test_short_terminal_observation_can_bridge_one_missing_plane():
    saturn = load_saturn_v571()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "ANALYSIS_MODE": "comparative",
            "SEGMENTATION_ENGINE": "unet_primary",
            "COMPARATIVE_TRACKING_MORPHOLOGY_NEUTRAL": True,
            "UM_PER_PX_XY": 0.1,
            "UM_PER_SLICE_Z": 0.5,
            "TRACK_MAX_DIST_UM": 2.0,
            "TRACK_MAX_GAP_SLICES": 1,
            "UNET_TRACK_MAX_RECONSTRUCTED_LENGTH_UM": 20.0,
        }
    )
    rows = _tracking_rows(target_x=10.0, target_length=9.0, target_width=1.0)
    rows.loc[1, "z_slice"] = 2

    tracked, summary = saturn.track_across_slices_global_assignment(rows, cfg)

    assert tracked["track_id"].nunique() == 1
    assert summary.loc[0, "observed_slice_count"] == 2
    assert summary.loc[0, "missing_slice_count"] == 1


def test_bbox_overlap_cannot_bypass_absolute_centroid_distance():
    saturn = load_saturn_v571()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "ANALYSIS_MODE": "comparative",
            "SEGMENTATION_ENGINE": "unet_primary",
            "COMPARATIVE_TRACKING_MORPHOLOGY_NEUTRAL": True,
            "UM_PER_PX_XY": 1.0,
            "UM_PER_SLICE_Z": 0.1,
            "TRACK_MAX_DIST_UM": 2.0,
            "TRACK_MAX_GAP_SLICES": 0,
        }
    )
    rows = _tracking_rows(target_x=15.0, target_length=2.0, target_width=0.5)
    rows.loc[:, "bbox_min_x"] = 0.0
    rows.loc[:, "bbox_max_x"] = 30.0
    tracked, _ = saturn.track_across_slices_global_assignment(rows, cfg)
    assert tracked["track_id"].nunique() == 2


def test_tuner_scores_reciprocal_continuity_without_morphology():
    tuner = load_tuner_v571()
    detections = _tracking_rows()
    tracked = detections.copy()
    tracked["track_id"] = 1
    cfg = tuner.CONFIG.copy()
    cfg.update(
        {
            "UM_PER_PX_XY": 0.1,
            "CONSERVATIVE_MAX_CENTROID_JUMP_UM": 5.0,
        }
    )
    metrics = tuner.reciprocal_tracking_integrity(detections, tracked, cfg)
    assert metrics["reciprocal_candidate_count"] == 1
    assert metrics["reciprocal_candidate_recovery_fraction"] == pytest.approx(1.0)
    assert metrics["accepted_nonreciprocal_fraction"] == pytest.approx(0.0)
