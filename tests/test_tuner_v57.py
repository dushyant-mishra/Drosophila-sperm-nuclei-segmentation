import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


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
