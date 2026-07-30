import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v57():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_multisample_test",
        ROOT / "sperm_segmentation_saturnv5.7.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v57_import_adds_project_root_for_unet_bridge():
    saturn = load_saturn_v57()

    assert str(ROOT) in saturn.sys.path


def test_quality_overlay_counts_only_report_genuinely_unmapped_labels():
    saturn = load_saturn_v57()
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:3, 1] = 1
    labels[3:5, 3] = 2
    labels[5:7, 5] = 3
    labels[2:4, 6] = 4
    slice_tracks = pd.DataFrame(
        {
            "sperm_id": [1, 2, 3],
            "track_id": [101, 102, 103],
        }
    )
    quality = {101: "candidate", 102: "warning", 103: "hard_fail"}

    counts = saturn.quality_overlay_status_counts(labels, slice_tracks, quality)

    assert counts == {
        "candidate": 1,
        "warning": 1,
        "hard_fail": 1,
        "unmapped": 1,
    }
    legend_labels = [
        handle.get_label()
        for handle in saturn.quality_overlay_legend_handles({"candidate", "warning"})
    ]
    assert legend_labels == [
        "Included estimated nucleus",
        "Included; morphology warning",
    ]


def test_hybrid_mode_refuses_missing_checkpoint_instead_of_falling_back():
    saturn = load_saturn_v57()
    shape = (16, 16)
    mask = np.zeros(shape, dtype=bool)
    ridge = np.zeros(shape, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    cfg = saturn.CONFIG.copy()
    cfg.update({"SEGMENTATION_ENGINE": "hybrid", "UNET_MODEL_PATH": ""})

    with pytest.raises(RuntimeError, match="requires UNET_MODEL_PATH"):
        saturn._apply_unet_candidate_support(
            mask,
            ridge,
            valid,
            shape,
            (0, shape[0], 0, shape[1]),
            valid,
            cfg,
            np.zeros((3, *shape), dtype=np.float32),
            z_idx=5,
        )


def test_hybrid_mode_refuses_missing_checkpoint_file():
    saturn = load_saturn_v57()
    shape = (16, 16)
    mask = np.zeros(shape, dtype=bool)
    ridge = np.zeros(shape, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UNET_MODEL_PATH": str(ROOT / "does_not_exist.pt"),
        }
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        saturn._apply_unet_candidate_support(
            mask,
            ridge,
            valid,
            shape,
            (0, shape[0], 0, shape[1]),
            valid,
            cfg,
            np.zeros((3, *shape), dtype=np.float32),
            z_idx=5,
        )


def test_component_distance_transform_matches_full_frame():
    saturn = load_saturn_v57()
    mask = np.zeros((96, 112), dtype=bool)
    mask[15:35, 18:45] = True
    mask[20:30, 25:38] = False

    expected = distance_transform_edt(mask)
    actual = saturn._distance_transform_component(mask)

    np.testing.assert_array_equal(actual, expected)


def test_tracking_rejects_join_that_exceeds_physical_length_guard():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "UM_PER_PX_XY": 0.4,
            "UM_PER_SLICE_Z": 1.0,
            "TRACK_TECHNICAL_MAX_JOINED_LENGTH_UM": 15.0,
        }
    )
    previous = {
        "first_z": 0,
        "first_x": 20.0,
        "first_y": 20.0,
        "last_z": 11,
        "last_x": 20.0,
        "last_y": 20.0,
        "last_width": 2.0,
        "last_length": 10.0,
        "last_area": 30.0,
        "last_orientation": 0.0,
        "max_length_2d": 10.0,
    }
    candidate = {
        "z_slice": 12,
        "centroid_x": 20.0,
        "centroid_y": 20.0,
        "width_um": 2.0,
        "length_um_geodesic": 10.0,
        "area_px": 30.0,
        "orientation": 0.0,
    }

    accepted, reason = saturn.check_extension_consistency(
        previous, candidate, cfg, overlap_exists=True
    )

    assert not accepted
    assert reason.startswith("technical_joined_length=")


def test_unet_rescue_keeps_biologically_short_high_confidence_nucleus():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UM_PER_PX_XY": 0.3,
            "UNET_RESCUE_ENABLE": True,
            "UNET_RESCUE_THRESHOLD": 0.7,
            "UNET_RESCUE_MIN_COMPONENT_PX": 3,
            "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 1,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (48, 48)
    probability = np.zeros(shape, dtype=np.float32)
    probability[23, 16:25] = 0.99
    empty = np.zeros(shape, dtype=bool)
    seg = {
        "skel_pruned": empty.copy(),
        "dist_clean": np.zeros(shape, dtype=float),
        "skel_labeled": np.zeros(shape, dtype=np.int32),
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": empty.copy(),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert len(measured["results"]) == 1
    result = measured["results"][0]
    length_um = result["length_px_geodesic"] * cfg["UM_PER_PX_XY"]
    assert 2.0 <= length_um < cfg["MIN_SKEL_LEN_UM"]
    assert result["detection_source"] == "unet_rescued"


def test_unet_rescue_confidence_exception_keeps_nucleus_below_resolution_floor():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UM_PER_PX_XY": 0.4,
            "UNET_RESCUE_ENABLE": True,
            "UNET_RESCUE_THRESHOLD": 0.7,
            "UNET_RESCUE_MIN_COMPONENT_PX": 3,
            "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
            "UNET_SHORT_RESCUE_MIN_MEAN_PROB": 0.85,
            "MIN_LENGTH_WIDTH_RATIO": 1.5,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 1,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (48, 48)
    probability = np.zeros(shape, dtype=np.float32)
    probability[23, 18:23] = 0.99
    empty = np.zeros(shape, dtype=bool)
    seg = {
        "skel_pruned": empty.copy(),
        "dist_clean": np.zeros(shape, dtype=float),
        "skel_labeled": np.zeros(shape, dtype=np.int32),
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": empty.copy(),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert len(measured["results"]) == 1
    result = measured["results"][0]
    assert result["length_px_geodesic"] * cfg["UM_PER_PX_XY"] < 2.0
    assert result["detection_source"] == "unet_rescued_short_high_confidence"


def test_unet_rescue_retains_low_ratio_nucleus_as_morphology_warning():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UM_PER_PX_XY": 0.3,
            "UNET_RESCUE_ENABLE": True,
            "UNET_RESCUE_THRESHOLD": 0.3,
            "UNET_RESCUE_MIN_COMPONENT_PX": 3,
            "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
            "UNET_LOW_RATIO_RESCUE_MIN_MEAN_PROB": 0.75,
            "UNET_LOW_RATIO_RESCUE_MIN_LENGTH_UM": 4.0,
            "MIN_LENGTH_WIDTH_RATIO": 3.0,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 1,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (64, 64)
    probability = np.zeros(shape, dtype=np.float32)
    probability[25:32, 15:37] = 0.95
    empty = np.zeros(shape, dtype=bool)
    seg = {
        "skel_pruned": empty.copy(),
        "dist_clean": np.zeros(shape, dtype=float),
        "skel_labeled": np.zeros(shape, dtype=np.int32),
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": empty.copy(),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert len(measured["results"]) == 1
    result = measured["results"][0]
    assert result["detection_source"] == "unet_rescued_morphology_warning"
    assert result["unet_rescue_morphology_warning"] is True
    assert (
        "low_length_width_ratio"
        in result["unet_rescue_morphology_warning_reasons"]
    )


def test_unet_rescue_hysteresis_keeps_faint_support_connected_to_seed():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UM_PER_PX_XY": 0.3,
            "UNET_RESCUE_ENABLE": True,
            "UNET_CANDIDATE_THRESHOLD": 0.05,
            "UNET_RESCUE_THRESHOLD": 0.30,
            "UNET_RESCUE_MIN_COMPONENT_PX": 3,
            "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 0,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (48, 48)
    probability = np.zeros(shape, dtype=np.float32)
    probability[24, 12:36] = 0.10
    probability[24, 21:27] = 0.90
    empty = np.zeros(shape, dtype=bool)
    seg = {
        "skel_pruned": empty.copy(),
        "dist_clean": np.zeros(shape, dtype=float),
        "skel_labeled": np.zeros(shape, dtype=np.int32),
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": empty.copy(),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert len(measured["results"]) == 1
    assert measured["results"][0]["length_px_geodesic"] >= 20


def test_unet_rescue_hysteresis_rejects_isolated_low_probability_noise():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UNET_RESCUE_ENABLE": True,
            "UNET_CANDIDATE_THRESHOLD": 0.05,
            "UNET_RESCUE_THRESHOLD": 0.30,
            "UNET_RESCUE_MIN_COMPONENT_PX": 2,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 0,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (32, 32)
    probability = np.zeros(shape, dtype=np.float32)
    probability[15, 8:24] = 0.10
    empty = np.zeros(shape, dtype=bool)
    seg = {
        "skel_pruned": empty.copy(),
        "dist_clean": np.zeros(shape, dtype=float),
        "skel_labeled": np.zeros(shape, dtype=np.int32),
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": empty.copy(),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert measured["results"] == []
    assert sum(measured["unet_rescue_rejected_counts"].values()) == 0


def test_unet_rescue_keeps_twenty_micron_limit_as_hard_guard():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UM_PER_PX_XY": 0.5,
            "MAX_GEODESIC_LEN_UM": 20.0,
            "MAX_GEODESIC_LEN_PX": 40.0,
            "UNET_RESCUE_ENABLE": True,
            "UNET_CANDIDATE_THRESHOLD": 0.05,
            "UNET_RESCUE_THRESHOLD": 0.30,
            "UNET_RESCUE_MIN_COMPONENT_PX": 3,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 0,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (80, 80)
    probability = np.zeros(shape, dtype=np.float32)
    probability[40, 10:65] = 0.95
    empty = np.zeros(shape, dtype=bool)
    seg = {
        "skel_pruned": empty.copy(),
        "dist_clean": np.zeros(shape, dtype=float),
        "skel_labeled": np.zeros(shape, dtype=np.int32),
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": empty.copy(),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert measured["results"] == []
    assert measured["unet_rescue_rejected_counts"]["long"] == 1


def test_unet_report_counts_confidence_exception_sources():
    saturn = load_saturn_v57()
    frame = pd.DataFrame(
        {
            "detection_source": [
                "saturn_classical",
                "unet_rescued",
                "unet_rescued_split",
                "unet_rescued_short_high_confidence",
                "unet_rescued_low_ratio_high_confidence",
            ],
            "unet_mean_probability": [0.8, 0.9, 0.85, 0.95, 0.92],
        }
    )

    summary = saturn.summarize_unet_rescue_for_reports(frame)

    assert summary["unet_rescued"] == 3
    assert summary["unet_rescued_split"] == 1
    assert summary["unet_total_rescued"] == 4


def test_unet_instance_already_represented_by_saturn_is_not_fragmented():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "hybrid",
            "UM_PER_PX_XY": 0.4,
            "UNET_RESCUE_ENABLE": True,
            "UNET_RESCUE_THRESHOLD": 0.7,
            "UNET_RESCUE_MIN_COMPONENT_PX": 3,
            "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
            "UNET_RESCUE_EXCLUDE_DILATION_PX": 1,
            "UNET_INSTANCE_SPLIT_ENABLE": False,
        }
    )
    shape = (48, 48)
    skeleton = np.zeros(shape, dtype=bool)
    skeleton[24, 12:32] = True
    labels = np.zeros(shape, dtype=np.int32)
    labels[skeleton] = 1
    probability = np.zeros(shape, dtype=np.float32)
    probability[22:27, 10:34] = 0.99
    seg = {
        "skel_pruned": skeleton,
        "dist_clean": np.ones(shape, dtype=float),
        "skel_labeled": labels,
        "unet_probability": probability,
        "roi_mask": np.ones(shape, dtype=bool),
        "exclusion_mask": np.zeros(shape, dtype=bool),
    }

    measured = saturn.measure_spermatids(seg, cfg)

    assert len(measured["results"]) == 1
    assert measured["results"][0]["detection_source"] == "saturn_classical"
    assert np.count_nonzero(measured["unet_rescue_rejected_reason"]) == 0


def test_post_detection_qc_reports_one_final_population(tmp_path):
    saturn = load_saturn_v57()
    detections = pd.DataFrame({"track_link_method": ["new", "overlap", "overlap"]})
    tracks = pd.DataFrame(
        {
            "quality_flags": ["", "wide", "unusual_pitch"],
            "reference_morphology_pass": [True, True, False],
            "is_biological_candidate": [True, True, True],
            "technical_valid": [True, True, True],
            "morphology_warning": [False, True, True],
            "has_warning_only": [False, True, True],
            "total_3d_length_um": [9.0, 10.0, 11.0],
            "z_span_um": [0.0, 1.0, 2.0],
            "n_slices": [1, 2, 3],
        }
    )

    saturn.export_post_detection_qc(tmp_path, detections, tracks)
    report = (tmp_path / "post_detection_qc.txt").read_text(encoding="utf-8")

    assert "FINAL ANALYSIS POPULATION:" in report
    assert "estimated_unique_nuclei: 3" in report
    assert "nuclei_with_morphology_review_note: 2" in report
    assert "reference_morphology_pass" not in report
    assert "warning_free" not in report


def test_analysis_summary_uses_only_technical_valid_tracks(tmp_path):
    saturn = load_saturn_v57()
    detections = pd.DataFrame(
        {
            "length_um_geodesic": [9.0, 10.0, 30.0],
            "width_um": [1.8, 2.0, 8.0],
            "detection_source": [
                "saturn_classical",
                "unet_rescued_low_ratio_high_confidence",
                "unet_rescued_split",
            ],
            "unet_mean_probability": [0.8, 0.92, 0.85],
        }
    )
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "technical_valid": [True, True, False],
            "total_3d_length_um": [10.0, 12.0, 99.0],
            "max_length_2d": [9.0, 11.0, 90.0],
            "thickness_um": [1.8, 2.2, 20.0],
            "tortuosity_3d": [1.0, 1.2, 9.0],
            "z_span_um": [1.0, 3.0, 40.0],
            "morphology_warning": [False, True, False],
        }
    )

    summary = saturn.export_analysis_summary(
        tmp_path,
        detections,
        tracks,
        cfg={"SEGMENTATION_ENGINE": "hybrid", "UNET_MODEL_PATH": "best.pt"},
    )

    assert summary["analysis_population"] == "included estimated nuclei"
    assert summary["estimated_unique_nuclei"] == 2
    assert summary["median_3d_length_um"] == 11.0
    assert summary["median_maximum_2d_length_um"] == 10.0
    assert summary["technical_failure_track_count_qc"] == 1
    assert summary["morphology_review_note_count_qc"] == 1
    assert summary["segmentation_engine"] == "hybrid"
    assert summary["unet_checkpoint"] == "best.pt"
    assert summary["unet_rescued_2d_count"] == 2
    assert summary["unet_rescued_split_2d_count"] == 1
    assert summary["unet_rescued_low_ratio_high_confidence_2d_count"] == 1
    assert summary["unet_probability_supported_2d_count"] == 3
    assert (tmp_path / "analysis_summary.csv").exists()
    with (tmp_path / "analysis_summary.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["estimated_unique_nuclei"] == 2
    assert payload["median_3d_length_um"] == 11.0
    assert "unet_rescued_2d_count" not in payload
    with (tmp_path / "technical_qc_summary.json").open("r", encoding="utf-8") as handle:
        technical_payload = json.load(handle)
    assert technical_payload["unet_rescued_2d_count"] == 2
    assert technical_payload["unet_rescued_split_2d_count"] == 1
    assert technical_payload["unet_rescued_low_ratio_high_confidence_2d_count"] == 1


def test_single_slice_summary_cannot_be_mistaken_for_unique_nuclei(tmp_path):
    saturn = load_saturn_v57()
    detections = pd.DataFrame(
        {
            "length_um_geodesic": [8.0, 10.0],
            "width_um": [1.5, 2.0],
            "detection_source": [
                "saturn_classical",
                "unet_rescued_low_ratio_high_confidence",
            ],
            "unet_mean_probability": [0.7, 0.93],
        }
    )

    summary = saturn.export_analysis_summary(
        tmp_path,
        detections,
        run_scope="single_slice_preview",
        z_index=12,
        cfg={"SEGMENTATION_ENGINE": "hybrid", "UNET_MODEL_PATH": "best.pt"},
    )

    assert summary["run_scope"] == "single_slice_preview"
    assert summary["biological_count_available"] is False
    assert np.isnan(summary["estimated_unique_nuclei"])
    assert summary["candidate_2d_detection_count"] == 2
    assert summary["median_2d_length_um"] == 9.0
    assert summary["unet_rescued_2d_count"] == 1
    assert summary["unet_rescued_low_ratio_high_confidence_2d_count"] == 1
    assert summary["unet_probability_supported_2d_count"] == 2
    with (tmp_path / "analysis_summary.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["estimated_unique_nuclei"] is None
    assert "not unique nuclei" in payload["interpretation"].lower()
    assert "unet_rescued_2d_count" not in payload
    with (tmp_path / "technical_qc_summary.json").open("r", encoding="utf-8") as handle:
        technical_payload = json.load(handle)
    assert technical_payload["unet_rescued_2d_count"] == 1


def test_completed_tracking_with_no_detections_reports_zero_not_missing():
    saturn = load_saturn_v57()

    summary = saturn.build_analysis_summary(
        pd.DataFrame(),
        pd.DataFrame(),
        run_scope="full_stack_3d",
    )

    assert summary["biological_count_available"] is True
    assert summary["estimated_unique_nuclei"] == 0
    assert summary["analysis_population"] == "included estimated nuclei"


def test_analysis_overlay_shows_only_included_tracks():
    saturn = load_saturn_v57()
    image = np.zeros((24, 32), dtype=np.uint8)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[8, 5:12] = 1
    labels[16, 18:26] = 2
    slice_tracks = pd.DataFrame(
        {
            "sperm_id": [1, 2],
            "track_id": [10, 20],
        }
    )

    overlay = saturn.make_analysis_overlay(
        image,
        labels,
        slice_tracks,
        {10},
    )

    assert overlay[8, 8, 1] > overlay[8, 8, 0]
    assert np.array_equal(overlay[16, 21], [0, 0, 0])


def test_primary_summary_includes_width_and_elongation():
    saturn = load_saturn_v57()
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2],
            "technical_valid": [True, True],
            "total_3d_length_um": [9.5, 10.5],
            "max_length_2d": [9.0, 10.0],
            "median_width_2d": [2.0, 2.4],
            "median_length_width_ratio_2d": [4.5, 4.2],
            "thickness_um": [2.1, 2.3],
            "tortuosity_3d": [1.0, 1.1],
            "z_span_um": [1.0, 2.0],
        }
    )

    summary = saturn.build_analysis_summary(
        pd.DataFrame(),
        tracks,
        run_scope="full_stack_3d",
    )

    assert summary["median_2d_width_um"] == pytest.approx(2.2)
    assert summary["median_2d_length_width_ratio"] == pytest.approx(4.35)


def test_post_detection_qc_medians_exclude_technical_failures(tmp_path):
    saturn = load_saturn_v57()
    tracks = pd.DataFrame(
        {
            "technical_valid": [True, True, False],
            "total_3d_length_um": [10.0, 12.0, 99.0],
            "z_span_um": [1.0, 3.0, 50.0],
            "n_slices": [2, 4, 1],
            "morphology_warning": [False, True, False],
        }
    )

    saturn.export_post_detection_qc(tmp_path, pd.DataFrame(), tracks)
    report = (tmp_path / "post_detection_qc.txt").read_text(encoding="utf-8")

    assert "estimated_unique_nuclei: 2" in report
    assert "Median 3D length um: 11.000" in report
    assert "Median Z-span um: 2.000" in report
    assert "Median 3D length um: 12.000" not in report


def test_tracking_audit_records_rejections_without_claiming_stops():
    saturn = load_saturn_v57()
    detections = pd.DataFrame({"track_id": [1, 1, 2]})
    tracks = pd.DataFrame({"track_id": [1, 2]})
    events = {
        1: ["z=3, reason=width_jump=0.80", "z=4, reason=area_jump=0.75"],
    }

    detections, tracks = saturn._attach_tracking_audit(detections, tracks, events)

    assert tracks.loc[tracks.track_id == 1, "rejected_extension_count"].item() == 2
    assert tracks.loc[tracks.track_id == 1, "has_rejected_extension"].item()
    assert tracks["track_stop_reason"].eq("").all()
    assert detections.loc[detections.track_id == 1, "track_rejected_extension_count"].eq(2).all()


def test_hybrid_tracking_preserves_rejected_extension_history():
    saturn = load_saturn_v57()
    cfg = saturn.CONFIG.copy()
    cfg["TRACKING_BACKEND"] = "hybrid_repair"
    detections = pd.DataFrame(
        {
            "z_slice": [0, 1],
            "sperm_id": [1, 1],
            "centroid_x": [20.0, 20.5],
            "centroid_y": [20.0, 20.5],
            "width_um": [2.0, 2.0],
            "length_um_geodesic": [8.0, 18.0],
            "area_px": [20.0, 45.0],
            "orientation": [0.0, 0.0],
            "bbox_min_y": [18.0, 18.0],
            "bbox_min_x": [18.0, 18.0],
            "bbox_max_y": [23.0, 23.0],
            "bbox_max_x": [23.0, 23.0],
            "tortuosity": [1.0, 1.0],
        }
    )

    tracked, tracks = saturn.track_across_slices(detections, cfg)

    assert "rejected_extension_count" in tracks.columns
    assert tracks["rejected_extension_count"].sum() >= 1
    assert tracks["track_stop_reason"].eq("").all()
    assert "track_rejected_extension_count" in tracked.columns


def test_biologist_results_uses_only_technical_valid_tracks(tmp_path):
    saturn = load_saturn_v57()
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "technical_valid": [True, True, False],
            "total_3d_length_um": [10.0, 14.0, 30.0],
            "max_length_2d": [9.0, 13.0, 29.0],
            "median_width_2d": [2.0, 2.4, 8.0],
            "median_length_width_ratio_2d": [4.5, 5.4, 3.6],
            "thickness_um": [2.0, 2.4, 8.0],
            "tortuosity_3d": [1.0, 1.2, 4.0],
            "z_span_um": [2.0, 3.0, 10.0],
            "n_slices": [3, 4, 1],
            "morphology_warning": [False, True, False],
        }
    )

    paths = saturn.export_biologist_results(tmp_path, tracks, "test")
    summary = pd.read_csv(paths["summary"])
    nuclei = pd.read_csv(paths["nuclei"])

    assert summary.loc[0, "estimated_unique_nuclei"] == 2
    assert summary.loc[0, "median_3d_length_um"] == 12.0
    assert summary.loc[0, "median_2d_width_um"] == 2.2
    assert summary.loc[0, "median_2d_length_width_ratio"] == 4.95
    assert len(nuclei) == 2
    assert set(nuclei["estimated_nucleus_id"]) == {1, 2}
    assert "median_2d_width_um" in nuclei.columns


def test_track_exports_do_not_duplicate_analysis_population(tmp_path):
    saturn = load_saturn_v57()
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "technical_valid": [True, True, False],
            "is_biological_candidate": [True, True, False],
        }
    )

    paths = saturn.export_comparative_track_tables(tmp_path, tracks, "test")

    assert set(paths) == {"track_summary_technical_failures_test.csv"}
    assert not (tmp_path / "track_summary_all_test.csv").exists()
    assert not (tmp_path / "track_summary_technical_valid_test.csv").exists()
    assert not (tmp_path / "track_summary_biological_candidates_test.csv").exists()


def test_excel_uses_one_track_audit_sheet_without_candidate_duplicate(tmp_path):
    saturn = load_saturn_v57()
    tracks = pd.DataFrame(
        {
            "track_id": [1, 2],
            "technical_valid": [True, True],
            "total_3d_length_um": [10.0, 12.0],
            "max_length_2d": [9.0, 11.0],
            "thickness_um": [2.0, 2.2],
            "tortuosity_3d": [1.0, 1.1],
            "z_span_um": [2.0, 3.0],
            "z_covered_um": [2.0, 3.0],
            "volume_um3": [20.0, 24.0],
            "n_slices": [3, 4],
        }
    )

    saturn.generate_excel_report(tmp_path, pd.DataFrame(), pd.DataFrame(), tracks)
    workbook_path = tmp_path / f"batch_analysis_results_{saturn._VERSION}.xlsx"
    with zipfile.ZipFile(workbook_path) as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")

    assert 'name="Biologist_Results"' in workbook_xml
    assert 'name="Technical_QC"' in workbook_xml
    assert 'name="Population_Summary"' not in workbook_xml
    assert 'name="3D_Track_Audit"' in workbook_xml
    assert 'name="3D_Biological_Candidates"' not in workbook_xml
    assert 'name="3D_Morphometrics"' not in workbook_xml


def test_batch_pdf_accepts_primary_biology_contract(tmp_path):
    saturn = load_saturn_v57()
    detections = pd.DataFrame(
        {
            "z_slice": [0],
            "length_um_geodesic": [9.5],
            "width_um": [2.1],
            "length_width_ratio": [4.52],
            "detection_source": ["saturn_classical"],
        }
    )
    slice_summary = pd.DataFrame(
        {
            "z_slice": [0],
            "n_spermatids": [1],
            "median_length_um": [9.5],
        }
    )
    tracks = pd.DataFrame(
        {
            "track_id": [1],
            "technical_valid": [True],
            "z_start": [0],
            "z_end": [0],
            "total_3d_length_um": [9.5],
            "max_length_2d": [9.5],
            "median_width_2d": [2.1],
            "median_length_width_ratio_2d": [4.52],
            "thickness_um": [2.0],
            "tortuosity_3d": [1.0],
            "z_span_um": [0.0],
            "volume_um3": [28.0],
            "pitch_deg": [0.0],
            "taper_ratio": [1.0],
            "nearest_neighbor_um": [np.nan],
            "n_slices": [1],
        }
    )

    saturn.generate_batch_report(
        tmp_path,
        detections,
        slice_summary,
        0.25,
        tracks,
        generate_pptx=False,
    )

    assert (tmp_path / f"batch_report_{saturn._VERSION}.pdf").is_file()


def make_sample(root, group, sample_id, roi=True):
    folder = root / group / sample_id
    folder.mkdir(parents=True)
    for z_index in range(3):
        image = np.full((24, 32), 10 + z_index, dtype=np.uint8)
        tifffile.imwrite(folder / f"Project_Series002_z{z_index:02d}_ch00.tif", image)
    tifffile.imwrite(folder / "Project_Series002_z00_ch00 (1).tif", np.zeros((24, 32), dtype=np.uint8))
    tifffile.imwrite(folder / "Project_Series002_z00_ch01.tif", np.zeros((24, 32), dtype=np.uint8))
    if roi:
        mask = np.zeros((24, 32), dtype=bool)
        mask[4:20, 5:27] = True
        np.save(folder / "analysis_roi_v5_7.npy", mask)
    return folder


def test_parameter_editor_exposes_v57_hybrid_configuration():
    saturn = load_saturn_v57()
    expected = {
        "SEGMENTATION_ENGINE",
        "UNET_MODEL_PATH",
        "UNET_THRESHOLD_MODE",
        "UNET_RESCUE_ENABLE",
        "UNET_INSTANCE_SPLIT_ENABLE",
        "UNET_RESCUE_SPLIT_THRESHOLDS",
    }

    unet_keys = set(saturn.PARAM_SECTIONS["2.5D U-Net Integration"])
    assert expected <= unet_keys
    assert list(saturn.PARAM_SECTIONS).index("2.5D U-Net Integration") == 1
    assert all(saturn._parameter_editor_can_display(key, saturn.CONFIG) for key in expected)
    assert saturn.PARAM_ENUM_OPTIONS["SEGMENTATION_ENGINE"] == (
        "classical_saturn",
        "hybrid",
        "unet_assisted",
    )
    assert saturn._coerce_parameter_editor_value("hybrid", str) == "hybrid"
    assert saturn._coerce_parameter_editor_value(True, bool) is True
    assert saturn._coerce_parameter_editor_value("[0.7, 0.8, 0.9]", list) == [0.7, 0.8, 0.9]


def test_discovery_uses_exact_sources_and_validates_roi(tmp_path):
    saturn = load_saturn_v57()
    make_sample(tmp_path, "WT Test SV", "WT-1")
    make_sample(tmp_path, "SATNull Test SV", "SATNull-1")

    rows = saturn.discover_multisample_study(
        tmp_path,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
    )
    assert len(rows) == 2
    assert {row["group"] for row in rows} == {"WT", "SATNull"}
    assert all(row["slice_count"] == 3 for row in rows)
    assert all(row["file_pattern"] == "Project_Series002_z*_ch00.tif" for row in rows)

    validated, errors = saturn.validate_multisample_manifest(rows)
    assert errors == []
    assert all(row["status"] == "validated" for row in validated)
    assert all((row["z_min"], row["z_max"]) == (0, 2) for row in validated)


def test_discovery_accepts_numbered_project_and_single_named_roi(tmp_path):
    saturn = load_saturn_v57()
    folder = tmp_path / "KJ" / "KJ-1"
    folder.mkdir(parents=True)
    image = np.zeros((24, 32), dtype=np.uint8)
    for z_index in range(3):
        tifffile.imwrite(
            folder / f"Project001_Series013_z{z_index:02d}_ch00.tif",
            image,
        )
    np.save(folder / "roi_KJ_01.npy", np.ones(image.shape, dtype=bool))

    rows = saturn.discover_multisample_study(tmp_path)

    assert len(rows) == 1
    assert rows[0]["group"] == "KJ"
    assert rows[0]["file_pattern"] == "Project001_Series013_z*_ch00.tif"
    assert rows[0]["roi_path"].endswith("roi_KJ_01.npy")
    validated, errors = saturn.validate_multisample_manifest(rows)
    assert errors == []
    assert validated[0]["status"] == "validated"


def test_discovery_normalizes_kj_and_w1118_group_folder_names(tmp_path):
    saturn = load_saturn_v57()
    make_sample(tmp_path, "kj sv feb", "KJ-1")
    make_sample(tmp_path, "w1118 sv feb", "WT-1")

    rows = saturn.discover_multisample_study(tmp_path)

    assert {row["group"] for row in rows} == {"KJ", "WT"}


def test_discovery_accepts_generic_z_names_and_only_channel_zero(tmp_path):
    saturn = load_saturn_v57()
    folder = tmp_path / "Mutant" / "Sample-A"
    folder.mkdir(parents=True)
    image = np.zeros((24, 32), dtype=np.uint8)
    for z_index in range(3):
        tifffile.imwrite(folder / f"GraceSample-Z{z_index:03d}-C0.TIF", image)
        tifffile.imwrite(folder / f"GraceSample-Z{z_index:03d}-C1.TIF", image)
    np.save(folder / "roi_sample.npy", np.ones(image.shape, dtype=bool))

    rows = saturn.discover_multisample_study(tmp_path)

    assert len(rows) == 1
    assert rows[0]["slice_count"] == 3
    assert rows[0]["file_pattern"] == "GraceSample-Z[0-9]*-C0.TIF"
    validated, errors = saturn.validate_multisample_manifest(rows)
    assert errors == []
    assert validated[0]["status"] == "validated"


def test_discovery_accepts_trailing_numeric_slice_names(tmp_path):
    saturn = load_saturn_v57()
    folder = tmp_path / "WT" / "Sample-B"
    folder.mkdir(parents=True)
    image = np.zeros((24, 32), dtype=np.uint8)
    for z_index in range(4):
        tifffile.imwrite(folder / f"sampleB_{z_index:04d}.tiff", image)
    np.save(folder / "analysis_roi_v5_7.npy", np.ones(image.shape, dtype=bool))

    rows = saturn.discover_multisample_study(tmp_path)

    assert len(rows) == 1
    assert rows[0]["file_pattern"] == "sampleB_[0-9]*.tiff"
    assert rows[0]["slice_count"] == 4
    validated, errors = saturn.validate_multisample_manifest(rows)
    assert errors == []
    assert validated[0]["status"] == "validated"


def test_discovery_excludes_generated_output_directories(tmp_path):
    saturn = load_saturn_v57()
    source = tmp_path / "WT" / "Sample-C"
    output = source / "batch_output_1" / "overlays"
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    image = np.zeros((24, 32), dtype=np.uint8)
    for z_index in range(3):
        tifffile.imwrite(source / f"sampleC_z{z_index:03d}.tif", image)
        tifffile.imwrite(output / f"overlay_z{z_index:03d}.tif", image)
    np.save(source / "analysis_roi_v5_7.npy", np.ones(image.shape, dtype=bool))

    rows = saturn.discover_multisample_study(tmp_path)

    assert len(rows) == 1
    assert rows[0]["input_dir"] == str(source.resolve())


def test_organizer_creates_canonical_copy_and_preserves_group_and_roi(tmp_path):
    saturn = load_saturn_v57()
    source_root = tmp_path / "source"
    make_sample(source_root, "OriginalFolders", "Specimen-A")
    rows = saturn.discover_multisample_study(
        source_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
    )
    rows[0]["group"] = "WT"
    rows[0]["sample_id"] = "WT_01"
    original_files = sorted(Path(rows[0]["input_dir"]).glob("*.tif"))

    organized, summary = saturn.organize_multisample_study_copy(
        rows,
        tmp_path / "organized",
    )

    destination = tmp_path / "organized" / "WT" / "WT_01"
    assert [path.name for path in sorted(destination.glob("*.tif"))] == [
        "WT_01_z0000_ch00.tif",
        "WT_01_z0001_ch00.tif",
        "WT_01_z0002_ch00.tif",
    ]
    assert (destination / "analysis_roi_v5_7.npy").is_file()
    assert (tmp_path / "organized" / "organized_study_manifest.csv").is_file()
    assert (tmp_path / "organized" / "source_file_mapping.csv").is_file()
    assert (tmp_path / "organized" / "organization_summary.json").is_file()
    assert organized[0]["group"] == "WT"
    assert organized[0]["file_pattern"] == "WT_01_z[0-9]*_ch00.tif"
    assert summary["sample_count"] == 1
    assert summary["samples_missing_roi"] == []
    assert all(path.is_file() for path in original_files)


def test_organizer_reports_missing_roi_without_borrowing_one(tmp_path):
    saturn = load_saturn_v57()
    source_root = tmp_path / "source"
    make_sample(source_root, "Mutant", "Specimen-B", roi=False)
    rows = saturn.discover_multisample_study(
        source_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
    )
    rows[0]["group"] = "Mutant"
    rows[0]["sample_id"] = "Mutant_01"

    organized, summary = saturn.organize_multisample_study_copy(
        rows,
        tmp_path / "organized",
    )

    destination = tmp_path / "organized" / "Mutant" / "Mutant_01"
    assert len(list(destination.glob("*.tif"))) == 3
    assert not (destination / "analysis_roi_v5_7.npy").exists()
    assert summary["samples_missing_roi"] == ["Mutant_01"]
    validated, errors = saturn.validate_multisample_manifest(organized)
    assert any("ROI file missing" in error for error in errors)
    assert validated[0]["status"] == "invalid"


def test_organizer_refuses_nonempty_unrelated_output_folder(tmp_path):
    saturn = load_saturn_v57()
    source_root = tmp_path / "source"
    make_sample(source_root, "WT", "Specimen-C")
    rows = saturn.discover_multisample_study(source_root)
    output = tmp_path / "organized"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="empty output folder"):
        saturn.organize_multisample_study_copy(rows, output)


def test_validation_rejects_missing_roi_and_duplicate_sample_ids(tmp_path):
    saturn = load_saturn_v57()
    make_sample(tmp_path, "WT", "WT-1")
    make_sample(tmp_path, "WT", "WT-2", roi=False)
    rows = saturn.discover_multisample_study(tmp_path)
    rows[1]["sample_id"] = rows[0]["sample_id"]

    validated, errors = saturn.validate_multisample_manifest(rows)
    assert any("ROI file missing" in error for error in errors)
    assert any("duplicate sample ID" in error for error in errors)
    assert any(row["status"] == "invalid" for row in validated)


def test_leica_metadata_preserves_padded_series_and_physical_calibration(tmp_path):
    saturn = load_saturn_v57()
    metadata_dir = tmp_path / "MetaData"
    metadata_dir.mkdir()
    (metadata_dir / "Project_Series002.xml").write_text(
        """<Root>
        <DimensionDescription DimID="1" NumberOfElements="32" Length="0.000024" />
        <DimensionDescription DimID="3" NumberOfElements="3" Length="0.000006" />
        <ATLConfocalSettingDefinition Begin="0" End="0.000004" ObjectiveName="40x" Zoom="1" />
        <Detector IsActive="1" Gain="700" IsTimeGateActivated="0" />
        </Root>""",
        encoding="utf-8",
    )
    result = saturn._study_parse_leica_metadata(tmp_path, "", 2, 9.0, 9.0)
    assert result["xy_um_per_pixel"] == 0.75
    assert result["z_um_per_slice"] == 2.0
    assert "objective=40x" in result["acquisition_class"]


def test_study_run_isolates_samples_aggregates_and_resumes(tmp_path):
    saturn = load_saturn_v57()
    make_sample(tmp_path / "input", "WT", "WT-1")
    make_sample(tmp_path / "input", "SATNull", "SATNull-1")
    rows = saturn.discover_multisample_study(
        tmp_path / "input",
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
    )
    calls = []

    def fake_batch_runner(cfg):
        calls.append(cfg["INPUT_DIR"])
        output = Path(cfg["OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "detection_source": ["saturn_classical", "unet_rescued_core"],
                "length_um_geodesic": [9.0, 10.0],
                "width_um": [1.8, 2.0],
                "z_slice": [0, 2],
            }
        ).to_csv(output / "spermatid_measurements_v5.7.csv", index=False)
        pd.DataFrame(
            {
                "track_id": [1, 2],
                "detection_source": ["saturn_classical", "unet_rescued_core"],
                "z_slice": [0, 2],
                "length_um_geodesic": [9.0, 10.0],
                "width_um": [1.8, 2.0],
            }
        ).to_csv(output / "measurements_with_tracks_v5.7.csv", index=False)
        pd.DataFrame(
            {
                "track_id": [1, 2],
                "technical_valid": [True, False],
                "is_biological_candidate": [True, False],
                "is_quality_track": [True, False],
                "max_length_2d": [9.0, 10.0],
                "total_3d_length_um": [9.4, 99.0],
                "tortuosity_3d": [1.1, 1.2],
                "thickness_um": [1.7, 8.0],
                "volume_um3": [24.0, 400.0],
                "z_span_um": [0.0, 0.0],
                "z_start": [0, 2],
                "z_end": [0, 2],
            }
        ).to_csv(output / "track_summary_v5.7.csv", index=False)
        pd.DataFrame(columns=["track_id", "z_start", "z_end"]).to_csv(
            output / "track_summary_technical_failures_v5.7.csv", index=False
        )
        with (output / "stack_preprocessing_qc.json").open("w", encoding="utf-8") as handle:
            json.dump({"roi_pixel_count": 352}, handle)

    output_root = tmp_path / "study_output"
    state, summary = saturn.run_multisample_study(
        rows,
        output_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
        batch_runner=fake_batch_runner,
    )
    assert len(calls) == 2
    assert set(summary["status"]) == {"complete"}
    assert set(summary["raw_2d_detection_count"]) == {2}
    assert set(summary["estimated_unique_nuclei"]) == {1}
    assert set(summary["roi_area_um2"].round(6)) == {198.0}
    assert set(summary["sampled_depth_um"].round(6)) == {3.12}
    assert set(summary["stack_span_um"].round(6)) == {2.08}
    assert set(summary["sampled_roi_volume_um3"].round(6)) == {617.76}
    assert set(summary["estimated_nuclei_per_1000_um2"].round(6)) == {5.050505}
    assert set(summary["estimated_nuclei_per_100000_um3"].round(6)) == {161.875162}
    assert set(summary["qc_unet_associated_3d_track_count"]) == {1}
    assert set(summary["qc_analysis_population_unet_track_count"]) == {0}
    assert set(summary["unet_rescued_2d_count"]) == {1}
    assert set(summary["unet_rescued_other_2d_count"]) == {1}
    assert set(summary["estimated_unique_nuclei_classical_only"]) == {1}
    assert set(summary["estimated_unique_nuclei_with_unet_evidence"]) == {0}
    assert set(summary["estimated_unique_nuclei_unet_fraction"]) == {0.0}
    assert set(summary["median_2d_length_um"]) == {9.0}
    assert set(summary["median_2d_width_um"]) == {1.8}
    assert set(summary["median_3d_length_um"]) == {9.4}
    assert set(summary["median_3d_thickness_um"]) == {1.7}
    assert set(summary["median_3d_volume_um3"]) == {24.0}
    assert set(summary["z_boundary_track_count"]) == {2}
    assert set(summary["z_boundary_track_fraction"]) == {1.0}
    assert (output_root / "study_manifest.csv").exists()
    assert (output_root / "specimen_summary.csv").exists()
    assert (output_root / "specimen_technical_qc.csv").exists()
    assert (output_root / "group_summary.csv").exists()
    assert (output_root / "specimen_group_comparisons.csv").exists()
    assert (output_root / "specimen_group_comparison_qc.json").exists()
    assert (output_root / "specimen_group_comparison.pdf").exists()
    comparisons = pd.read_csv(output_root / "specimen_group_comparisons.csv")
    assert set(comparisons["analysis_unit"]) == {"biological specimen"}
    assert set(comparisons["inference_status"]) == {"insufficient_specimens"}
    with (output_root / "normalization_qc.json").open("r", encoding="utf-8") as handle:
        normalization_qc = json.load(handle)
    assert normalization_qc["roi_area_max_min_ratio"] == 1.0
    assert normalization_qc["high_z_boundary_specimen_count"] == 2
    assert normalization_qc["normalization_review_required"] is True
    tracks = pd.read_csv(output_root / "study_track_records.csv")
    assert tracks["study_track_id"].is_unique
    assert all(":" in value for value in tracks["study_track_id"])
    assert all(record["status"] == "complete" for record in state["samples"].values())
    assert all((Path(record["output_dir"]) / "sample_complete.json").exists() for record in state["samples"].values())
    specimen_primary = pd.read_csv(output_root / "specimen_summary.csv")
    assert "estimated_unique_nuclei" in specimen_primary.columns
    assert "unet_rescued_2d_count" not in specimen_primary.columns
    specimen_qc = pd.read_csv(output_root / "specimen_technical_qc.csv")
    assert "unet_rescued_2d_count" in specimen_qc.columns

    saturn.run_multisample_study(
        rows,
        output_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
        batch_runner=fake_batch_runner,
    )
    assert len(calls) == 2
    assert all(len(list((output_root / "samples" / row["sample_id"]).glob("attempt_*"))) == 1 for row in rows)


def test_specimen_group_comparison_is_deterministic_and_uses_specimens():
    saturn = load_saturn_v57()
    specimen_frame = pd.DataFrame(
        {
            "sample_id": ["WT-1", "WT-2", "WT-3", "KJ-1", "KJ-2", "KJ-3"],
            "group": ["WT", "WT", "WT", "KJ", "KJ", "KJ"],
            "status": ["complete"] * 6,
            "median_3d_length_um": [9.0, 9.5, 10.0, 11.0, 12.0, 13.0],
        }
    )

    first, qc_first = saturn._study_specimen_group_comparisons(
        specimen_frame,
        random_seed=123,
        bootstrap_resamples=250,
        permutation_resamples=999,
    )
    second, qc_second = saturn._study_specimen_group_comparisons(
        specimen_frame,
        random_seed=123,
        bootstrap_resamples=250,
        permutation_resamples=999,
    )

    pd.testing.assert_frame_equal(first, second)
    assert qc_first == qc_second
    row = first[first["metric"] == "median_3d_length_um"].iloc[0]
    assert row["analysis_unit"] == "biological specimen"
    assert row["reference_group"] == "WT"
    assert row["comparison_group"] == "KJ"
    assert row["reference_n"] == 3
    assert row["comparison_n"] == 3
    assert row["median_difference_comparison_minus_reference"] == 2.5
    assert row["cliffs_delta_comparison_minus_reference"] > 0
    assert row["inference_status"] == "exploratory_small_sample"


def test_study_run_stops_after_current_sample_and_resumes(tmp_path):
    saturn = load_saturn_v57()
    for sample_id in ("WT-1", "WT-2", "WT-3"):
        make_sample(tmp_path / "input", "WT", sample_id)
    rows = saturn.discover_multisample_study(
        tmp_path / "input",
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
    )
    calls = []
    events = []
    stop_state = {"requested": False}

    def fake_batch_runner(cfg):
        calls.append(Path(cfg["INPUT_DIR"]).name)
        output = Path(cfg["OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "detection_source": ["saturn_classical"],
                "length_um_geodesic": [9.5],
                "width_um": [1.9],
                "z_slice": [1],
            }
        ).to_csv(output / "spermatid_measurements_v5.7.csv", index=False)
        pd.DataFrame(
            {
                "track_id": [1],
                "detection_source": ["saturn_classical"],
                "z_slice": [1],
            }
        ).to_csv(output / "measurements_with_tracks_v5.7.csv", index=False)
        pd.DataFrame(
            {
                "track_id": [1],
                "is_biological_candidate": [True],
                "is_quality_track": [True],
                "total_3d_length_um": [9.7],
                "tortuosity_3d": [1.1],
                "z_start": [1],
                "z_end": [1],
            }
        ).to_csv(output / "track_summary_v5.7.csv", index=False)
        pd.DataFrame(columns=["track_id", "z_start", "z_end"]).to_csv(
            output / "track_summary_technical_failures_v5.7.csv", index=False
        )
        with (output / "stack_preprocessing_qc.json").open("w", encoding="utf-8") as handle:
            json.dump({"roi_pixel_count": 352}, handle)
        if len(calls) == 1:
            stop_state["requested"] = True

    output_root = tmp_path / "study_output"
    state, summary = saturn.run_multisample_study(
        rows,
        output_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
        progress_callback=events.append,
        batch_runner=fake_batch_runner,
        stop_requested=lambda: stop_state["requested"],
    )

    assert calls == ["WT-1"]
    assert state["run_status"] == "stopped"
    assert len(summary) == 3
    assert summary["status"].value_counts().to_dict() == {"validated": 2, "complete": 1}
    assert state["samples"]["WT-1"]["status"] == "complete"
    assert "WT-2" not in state["samples"]
    assert [event["event"] for event in events] == ["started", "complete", "stopped"]
    assert events[-1]["position"] == 1

    stop_state["requested"] = False
    events.clear()
    resumed_state, resumed_summary = saturn.run_multisample_study(
        rows,
        output_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
        progress_callback=events.append,
        batch_runner=fake_batch_runner,
        stop_requested=lambda: stop_state["requested"],
    )

    assert calls == ["WT-1", "WT-2", "WT-3"]
    assert resumed_state["run_status"] == "complete"
    assert len(resumed_summary) == 3
    assert all(record["status"] == "complete" for record in resumed_state["samples"].values())
    assert events[0]["event"] == "skipped"
    assert events[0]["sample_id"] == "WT-1"
