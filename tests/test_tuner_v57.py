import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tuner():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_tuner_test",
        ROOT / "utils" / "tune_parameters_Saturnv5_7.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unet_candidate_sampling_starts_with_evidence_thresholds():
    tuner = load_tuner()
    first = tuner.sample_unet_rescue_candidates(
        tuner.UNET_RESCUE_PARAM_SPACE, 6, 12345, tuner.CONFIG
    )
    second = tuner.sample_unet_rescue_candidates(
        tuner.UNET_RESCUE_PARAM_SPACE, 6, 12345, tuner.CONFIG
    )

    assert first == second
    assert len(first) == 6
    assert first[0][0] == "evidence_0.05_0.30"
    assert first[0][1]["UNET_CANDIDATE_THRESHOLD"] == 0.05
    assert first[0][1]["UNET_SEED_THRESHOLD"] == 0.30
    assert first[0][1]["UNET_RESCUE_THRESHOLD"] == 0.30
    assert first[0][1]["UNET_RESCUE_MIN_SKEL_LEN_UM"] == 2.0
    assert first[0][1]["UNET_SHORT_RESCUE_MIN_MEAN_PROB"] == 0.35
    assert first[1][0] == "balanced_recall_review"
    assert first[1][1]["UNET_RESCUE_THRESHOLD"] == 0.20


def test_tracking_sampling_includes_reviewed_base_first():
    tuner = load_tuner()
    cfg = tuner.CONFIG.copy()
    cfg["TRACK_MAX_DIST_UM"] = 6.25

    candidates = tuner.sample_tracking_candidates(
        tuner.TRACKING_PARAM_SPACE, 4, 12345, cfg
    )

    assert len(candidates) == 4
    assert candidates[0][0] == "reviewed_base"
    assert candidates[0][1]["TRACK_MAX_DIST_UM"] == 6.25


def test_unet_evaluator_preserves_reviewed_base_configuration(monkeypatch):
    tuner = load_tuner()
    observed = {}

    def fake_segment(cfg):
        observed.update(cfg)
        return [], []

    monkeypatch.setattr(tuner, "segment_eval_images", fake_segment)
    monkeypatch.setattr(
        tuner,
        "summarize_candidate",
        lambda rows, segs, cfg: {"score": 0.0},
    )

    base_cfg = tuner.CONFIG.copy()
    base_cfg.update(
        {
            "UNET_MODEL_PATH": "new_epoch_003.pt",
            "THRESHOLD_HI": 91.234,
            "_UNET_PROBABILITY_CACHE": {"z": "cached"},
            "_UNET_PROBABILITY_CACHE_DIR": "new_model_cache",
        }
    )
    tuner.evaluate_unet_rescue_candidate(
        {
            "UNET_CANDIDATE_THRESHOLD": 0.05,
            "UNET_SEED_THRESHOLD": 0.30,
        },
        base_cfg=base_cfg,
    )

    assert observed["UNET_MODEL_PATH"] == "new_epoch_003.pt"
    assert observed["THRESHOLD_HI"] == 91.234
    assert observed["_UNET_PROBABILITY_CACHE"] == {"z": "cached"}
    assert observed["_UNET_PROBABILITY_CACHE_DIR"] == "new_model_cache"
    assert observed["SEGMENTATION_ENGINE"] == "hybrid"
    assert observed["UNET_FAIL_HARD"] is True
    assert observed["UNET_THRESHOLD_MODE"] == "soft"
    assert observed["UNET_RESCUE_ENABLE"] is True


def test_unet_summary_counts_all_rescue_subtypes_and_penalizes_extremes():
    tuner = load_tuner()
    cfg = tuner.CONFIG.copy()
    cfg.update(
        {
            "UM_PER_PX_XY": 1.0,
            "TUNING_OBJECTIVE": "unet_rescue",
            "ANALYSIS_MODE": "comparative",
        }
    )
    rows = [
        {
            "length_px_geodesic": 3.0,
            "width_px": 1.0,
            "length_width_ratio": 3.0,
            "detection_source": "unet_rescued_short_high_confidence",
            "unet_mean_probability": 0.9,
        },
        {
            "length_px_geodesic": 9.0,
            "width_px": 2.0,
            "length_width_ratio": 4.5,
            "detection_source": "unet_rescued_low_ratio_high_confidence",
            "unet_mean_probability": 0.8,
        },
        {
            "length_px_geodesic": 10.0,
            "width_px": 2.0,
            "length_width_ratio": 5.0,
            "detection_source": "unet_rescued_split",
            "unet_mean_probability": 0.8,
        },
        {
            "length_px_geodesic": 21.0,
            "width_px": 2.0,
            "length_width_ratio": 10.5,
            "detection_source": "saturn_classical",
            "unet_mean_probability": 0.0,
        },
    ]

    summary = tuner.summarize_candidate(rows, [], cfg)

    assert summary["unet_total_rescued_count"] == 3
    assert summary["unet_rescued_split_count"] == 1
    assert summary["unet_rescue_fraction"] == 0.75
    assert summary["very_short_object_fraction"] == 0.25
    assert summary["very_long_object_fraction"] == 0.25
    assert summary["unet_rescue_score"] > 250.0


def test_segmentation_evaluator_preserves_reviewed_base_configuration(monkeypatch):
    tuner = load_tuner()
    observed = {}

    def fake_segment(cfg):
        observed.update(cfg)
        return [], []

    monkeypatch.setattr(tuner, "segment_eval_images", fake_segment)
    monkeypatch.setattr(
        tuner,
        "summarize_candidate",
        lambda rows, segs, cfg: {"score": 0.0},
    )

    base_cfg = tuner.CONFIG.copy()
    base_cfg["CLAHE_MODE"] = "low_signal"
    tuner.evaluate_segmentation_candidate(
        {"THRESHOLD_HI": 90.0}, base_cfg=base_cfg
    )

    assert observed["CLAHE_MODE"] == "low_signal"
    assert observed["THRESHOLD_HI"] == 90.0
    assert observed["SEGMENTATION_ENGINE"] == "classical_saturn"
    assert observed["UNET_RESCUE_ENABLE"] is False
    assert observed["TUNING_OBJECTIVE"] == "segmentation"


def test_segmentation_sampling_includes_reviewed_base_first():
    tuner = load_tuner()
    cfg = tuner.CONFIG.copy()
    cfg["THRESHOLD_HI"] = 91.0
    cfg["THRESHOLD_LO"] = 83.0

    candidates = tuner.sample_segmentation_candidates(
        tuner.SEGMENTATION_PARAM_SPACE,
        4,
        12345,
        cfg,
    )

    assert len(candidates) == 4
    assert candidates[0][0] == "reviewed_base"
    assert candidates[0][1]["THRESHOLD_HI"] == 91.0
    assert candidates[0][1]["THRESHOLD_LO"] == 83.0


def test_segmentation_thresholds_keep_four_percentile_point_gap():
    tuner = load_tuner()
    params = tuner.params_from_vector(
        [88.0, 87.0, 8, 1.0, 6.0, 4.0, 2.0, 2.5],
        tuner.SEGMENTATION_PARAM_SPACE,
    )

    assert params["THRESHOLD_HI"] - params["THRESHOLD_LO"] >= 4.0


def test_segmentation_search_bounds_cover_approved_recall_range():
    tuner = load_tuner()
    bounds = {
        key: (lo, hi)
        for key, lo, hi, _is_int in tuner.SEGMENTATION_PARAM_SPACE
    }

    assert bounds["THRESHOLD_HI"] == (82.0, 92.0)
    assert bounds["THRESHOLD_LO"] == (70.0, 84.0)
    assert bounds["MIN_OBJ_PX"] == (3, 10)
    assert bounds["MIN_SKEL_LEN_UM"] == (4.0, 7.0)


def test_segmentation_score_rejects_empty_output():
    tuner = load_tuner()
    cfg = tuner.CONFIG.copy()
    cfg["TUNING_OBJECTIVE"] = "segmentation"
    tuner.roi_mask_global = np.ones((4, 4), dtype=bool)
    tuner.exclusion_mask_global = None
    empty = np.zeros((4, 4), dtype=bool)
    segs = [
        (
            {
                "mask_hyst": empty,
                "mask_clean": empty,
                "skel_pruned": empty,
                "bridge_stats": {
                    "skeleton_pixels_before": 0,
                    "skeleton_pixels_after": 0,
                },
            },
            {
                "results": [],
                "skel_label": np.zeros((4, 4), dtype=np.int32),
            },
        )
    ]

    summary = tuner.summarize_candidate([], segs, cfg)

    assert summary["n_2d"] == 0
    assert summary["empty_slice_fraction"] == 1.0
    assert summary["score"] >= 1e6


def test_roi_metrics_use_valid_roi_and_count_integer_label_leakage():
    tuner = load_tuner()
    cfg = tuner.CONFIG.copy()
    cfg.update({"TUNING_OBJECTIVE": "segmentation", "UM_PER_PX_XY": 1.0})
    roi = np.ones((4, 4), dtype=bool)
    roi[0, 0] = False
    exclusion = np.zeros((4, 4), dtype=bool)
    exclusion[0, 1] = True
    tuner.roi_mask_global = roi
    tuner.exclusion_mask_global = exclusion

    mask_hyst = np.zeros((4, 4), dtype=bool)
    mask_hyst[1, 1] = True
    mask_clean = mask_hyst.copy()
    skel_pruned = mask_hyst.copy()
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1, 1] = 1
    labels[0, 0] = 2
    labels[0, 1] = 2
    rows = [
        {
            "length_px_geodesic": 9.0,
            "width_px": 2.0,
            "length_width_ratio": 4.5,
            "detection_source": "saturn_classical",
        }
    ]
    segs = [
        (
            {
                "mask_hyst": mask_hyst,
                "mask_clean": mask_clean,
                "skel_pruned": skel_pruned,
                "bridge_stats": {
                    "skeleton_pixels_before": 10,
                    "skeleton_pixels_after": 12,
                },
            },
            {"results": rows, "skel_label": labels},
        )
    ]

    summary = tuner.summarize_candidate(rows, segs, cfg)

    assert summary["clean_mask_occupancy"] == pytest.approx(1 / 14)
    assert summary["hysteresis_occupancy"] == pytest.approx(1 / 14)
    assert summary["bridge_inflation"] == pytest.approx(0.2)
    assert summary["outside_roi_overlap_by_stage"]["skel_label"] == 1
    assert summary["exclusion_mask_overlap_by_stage"]["skel_label"] == 1
    assert summary["outside_roi_overlap_count"] == 1
    assert summary["exclusion_mask_overlap_count"] == 1


def test_source_discovery_ignores_unrecognized_tiff_artifacts(tmp_path):
    tuner = load_tuner()
    source_0 = tmp_path / "Project001_Series002_z00_ch00.tif"
    source_1 = tmp_path / "Project001_Series002_z01_ch00.tif"
    overlay = tmp_path / "final_overlay.tif"
    for path in (source_0, source_1, overlay):
        path.touch()

    files = tuner.list_images(tmp_path)

    assert files == [str(source_0), str(source_1)]


def test_profile_mode_restores_auto_context_and_writes_complete_preset(
    monkeypatch,
    tmp_path,
):
    tuner = load_tuner()
    ctx = tuner.segmentation.StackPreprocessContext(
        normalization_low=0.0,
        normalization_high=1.0,
        selected_clahe_clip=0.02,
        selected_clahe_profile="auto_original",
        contrast_score=0.3,
        sampled_z_indices=[0],
        roi_percentiles={},
        saturation_fraction=0.0,
        slice_brightness_statistics=[],
        source_dtype="uint8",
        inferred_bit_depth=8,
        resolved_pixel_parameters={},
        configuration_provenance={},
        image_shape=(4, 4),
        roi_pixel_count=16,
        excluded_pixel_count=0,
    )
    tuner.preprocess_context_global = ctx
    tuner.images_to_eval = [np.zeros((4, 4), dtype=np.uint8)]
    tuner.z_values_eval = [0]
    seen_profiles = []
    empty_labels = np.zeros((4, 4), dtype=np.int32)

    def fake_segment(_cfg):
        seen_profiles.append(
            tuner.preprocess_context_global.selected_clahe_profile
        )
        return [], [
            (
                {},
                {"skel_label": empty_labels, "results": []},
            )
        ]

    monkeypatch.setattr(tuner, "segment_eval_images", fake_segment)
    monkeypatch.setattr(
        tuner,
        "summarize_candidate",
        lambda rows, segs, cfg: {"score": float(len(seen_profiles))},
    )
    monkeypatch.setattr(
        tuner.segmentation,
        "make_overlay",
        lambda image, labels: np.zeros((4, 4, 3), dtype=np.uint8),
    )

    tuner.run_profile_mode(tmp_path, tuner.CONFIG.copy())

    assert seen_profiles == [
        "no_clahe",
        "high_contrast",
        "standard",
        "low_signal",
        "auto_original",
    ]
    assert tuner.preprocess_context_global is ctx
    preset_path = next(tmp_path.glob("best_preprocessing_profile_v5_7_*.json"))
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    assert all(key in preset for key in tuner.CONFIG)
    assert preset["CLAHE_MODE"] == "no_clahe"


def test_tracking_evaluator_runs_production_tracker_with_all_base_params(
    monkeypatch,
):
    tuner = load_tuner()
    observed = {}
    detections = pd.DataFrame(
        {
            "z_slice": [0, 1],
            "sperm_id": [1, 1],
            "length_um_geodesic": [9.0, 9.1],
        }
    )
    tracked = pd.DataFrame(
        {
            "track_link_method": ["new", "assignment_cost"],
            "track_link_distance_um": [float("nan"), 0.5],
            "track_link_gap_slices": [0, 1],
        }
    )
    tracks = pd.DataFrame(
        {
            "track_id": [1],
            "n_slices": [2],
            "total_3d_length_um": [9.2],
            "max_length_2d": [9.1],
        }
    )

    def fake_track(df, cfg):
        observed.update(cfg)
        return tracked.copy(), tracks.copy()

    monkeypatch.setattr(tuner.segmentation, "track_across_slices", fake_track)
    monkeypatch.setattr(
        tuner.segmentation,
        "flag_quality_tracks",
        lambda df, cfg: df.assign(technical_valid=True),
    )
    base_cfg = tuner.CONFIG.copy()
    base_cfg.update(
        {
            "TRACKING_BACKEND": "hybrid_repair",
            "TRACK_MAX_GAP_SLICES": 1,
            "ASSIGNMENT_OVERLAP_WEIGHT": 2.345,
            "ASSIGNMENT_UNET_SUPPORT_WEIGHT": 0.456,
            "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": 15.0,
        }
    )
    result = tuner.evaluate_tracking_candidate(
        detections,
        {
            "TRACK_MAX_DIST_UM": 5.5,
            "ASSIGNMENT_MAX_COST": 7.0,
            "ASSIGNMENT_DIST_WEIGHT": 1.2,
            "HYBRID_REPAIR_MAX_COST": 3.4,
            "HYBRID_REPAIR_MAX_LINK_DIST_UM": 4.5,
            "HYBRID_REPAIR_MIN_OVERLAP": 0.05,
        },
        base_cfg=base_cfg,
    )

    assert observed["ASSIGNMENT_OVERLAP_WEIGHT"] == 2.345
    assert observed["ASSIGNMENT_UNET_SUPPORT_WEIGHT"] == 0.456
    assert observed["HYBRID_REPAIR_MAX_FINAL_LENGTH_UM"] == 15.0
    assert result["n_tracks"] == 1
    assert result["multi_slice_tracks"] == 1
    assert result["resolved_ASSIGNMENT_OVERLAP_WEIGHT"] == 2.345
    assert result["resolved_ASSIGNMENT_UNET_SUPPORT_WEIGHT"] == 0.456


def test_saved_preset_is_complete_and_gui_loadable():
    tuner = load_tuner()
    tuner.z_values_eval = [5, 6]
    tuner.preprocess_context_global = SimpleNamespace(
        selected_clahe_profile="standard"
    )
    cfg = tuner.CONFIG.copy()
    cfg["UNET_MODEL_PATH"] = "epoch_003.pt"
    selected = {
        "mode": "unet_rescue",
        "score": 1.25,
        "numerical_rank": 1,
        "selection_status": "first_candidate_for_visual_inspection",
        "UNET_RESCUE_THRESHOLD": 0.30,
        "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
    }

    preset = tuner.loadable_parameter_preset(cfg, selected)

    assert all(key in preset for key in tuner.CONFIG)
    assert preset["UNET_MODEL_PATH"] == "epoch_003.pt"
    assert preset["UNET_RESCUE_THRESHOLD"] == 0.30
    assert preset["UNET_RESCUE_MIN_SKEL_LEN_UM"] == 2.0
    assert preset["_TUNING_METADATA"]["numerical_rank"] == 1


def test_stratum_aggregation_writes_one_shared_unchanged_preset(tmp_path):
    tuner = load_tuner()
    parameter_values = tuner.candidate_from_config(
        tuner.UNET_RESCUE_PARAM_SPACE,
        tuner.CONFIG,
        {
            "UNET_CANDIDATE_THRESHOLD": 0.05,
            "UNET_SEED_THRESHOLD": 0.30,
            "UNET_RESCUE_THRESHOLD": 0.30,
            "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
        },
    )
    paths = []
    for idx, score in enumerate((4.0, 5.0), start=1):
        path = tmp_path / f"stratum_{idx}.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "candidate_role": "evidence_0.05_0.30",
                        "score": score,
                        "n_2d": 100 + idx,
                        "unet_rescue_fraction": 0.12,
                        "very_long_object_fraction": 0.0,
                        **parameter_values,
                    }
                ]
            ),
            encoding="utf-8",
        )
        paths.append(path)

    cfg = tuner.CONFIG.copy()
    cfg["UNET_MODEL_PATH"] = "epoch_003.pt"
    preset_path, summaries = tuner.aggregate_stratum_results(
        paths,
        tmp_path / "shared",
        cfg,
        "evidence_0.05_0.30",
    )
    preset = json.loads(preset_path.read_text(encoding="utf-8"))

    assert len(summaries) == 1
    assert summaries[0]["stratum_count"] == 2
    assert preset["UNET_MODEL_PATH"] == "epoch_003.pt"
    assert preset["UNET_RESCUE_THRESHOLD"] == 0.30
    assert (
        preset["_TUNING_METADATA"]["candidate_role"]
        == "evidence_0.05_0.30"
    )
