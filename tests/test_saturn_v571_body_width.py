import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.draw import polygon
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_body_width_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def width_cfg(module, **updates):
    cfg = module.CONFIG.copy()
    cfg.update(
        {
            "BODY_WIDTH_ENABLE": True,
            "BODY_WIDTH_ENDPOINT_TRIM_FRACTION": 0.125,
            "BODY_WIDTH_SAMPLE_SPACING_PX": 1.0,
            "BODY_WIDTH_SMOOTH_SIGMA_PX": 0.75,
            "BODY_WIDTH_MIN_SAMPLES": 5,
            "UM_PER_PX_XY": 1.0,
        }
    )
    cfg.update(updates)
    return cfg


def rotated_rectangle(shape, center, length, width, angle_deg):
    theta = np.deg2rad(angle_deg)
    tangent = np.array([np.sin(theta), np.cos(theta)])
    normal = np.array([-tangent[1], tangent[0]])
    corners = np.array(
        [
            center - tangent * length / 2 - normal * width / 2,
            center + tangent * length / 2 - normal * width / 2,
            center + tangent * length / 2 + normal * width / 2,
            center - tangent * length / 2 + normal * width / 2,
        ]
    )
    rr, cc = polygon(corners[:, 0], corners[:, 1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


@pytest.mark.parametrize("angle", [0, 20, 45, 70, 90])
def test_subpixel_body_width_is_rotation_stable(angle):
    saturn = load_saturn_v571()
    mask = rotated_rectangle((128, 128), np.array([64.0, 64.0]), 70, 9, angle)
    centerline = np.argwhere(skeletonize(mask))

    result = saturn.measure_subpixel_body_width(
        mask,
        centerline,
        width_cfg(saturn),
    )

    assert result["body_width_sample_count"] >= 30
    assert result["body_width_px"] == pytest.approx(9.0, abs=0.8)
    assert result["body_width_p90_px"] == pytest.approx(9.0, abs=1.0)
    assert result["body_width_method"].startswith("subpixel_mask_contour")


def test_endpoint_trimming_reduces_taper_influence():
    saturn = load_saturn_v571()
    mask = np.zeros((100, 140), dtype=bool)
    rr, cc = polygon(
        [50, 44, 44, 50, 56, 56],
        [20, 35, 105, 120, 105, 35],
        shape=mask.shape,
    )
    mask[rr, cc] = True
    centerline = np.argwhere(skeletonize(mask))

    trimmed = saturn.measure_subpixel_body_width(
        mask,
        centerline,
        width_cfg(saturn, BODY_WIDTH_ENDPOINT_TRIM_FRACTION=0.15),
    )
    untrimmed = saturn.measure_subpixel_body_width(
        mask,
        centerline,
        width_cfg(saturn, BODY_WIDTH_ENDPOINT_TRIM_FRACTION=0.0),
    )

    assert trimmed["body_width_px"] >= untrimmed["body_width_px"]
    assert trimmed["body_width_px"] == pytest.approx(13.0, abs=1.0)


def test_measurement_exports_legacy_and_new_width_fields():
    saturn = load_saturn_v571()
    mask = rotated_rectangle((96, 128), np.array([48.0, 64.0]), 60, 9, 25)
    labels = mask.astype(np.int32)
    centerline = skeletonize(mask).astype(np.int32)
    seg = {
        "unet_primary_instance_labels": labels,
        "unet_primary_centerline_labels": centerline,
        "unet_probability": np.where(mask, 0.95, 0.0).astype(np.float32),
        "unet_primary_parent_by_instance": {1: 1},
        "unet_primary_instance_sources": {1: "unet_primary"},
        "unet_primary_centerline_metadata": {1: {}},
        "unet_primary_technical_failures": [],
    }
    cfg = saturn.cfg_with_resolved_pixels(
        width_cfg(
            saturn,
            SEGMENTATION_ENGINE="unet_primary",
            MIN_SKEL_LEN_UM=1.0,
            MAX_GEODESIC_LEN_UM=100.0,
            MAX_WIDTH_UM=100.0,
        )
    )

    measured = saturn._measure_unet_primary_instances(seg, cfg)
    assert len(measured["results"]) == 1
    result = measured["results"][0]
    assert result["width_px_dt_median_legacy"] == result["width_px"]
    assert result["length_width_ratio_dt_legacy"] == result["length_width_ratio"]
    assert result["body_width_px"] == pytest.approx(9.0, abs=0.8)
    rows = saturn.rows_from_results(measured["results"], z_idx=4, um=1.0)
    assert rows[0]["width_um_dt_median_legacy"] > 0
    assert rows[0]["body_width_um"] == pytest.approx(result["body_width_px"], abs=1e-3)
    assert rows[0]["width_um"] == pytest.approx(rows[0]["body_width_um"], abs=1e-3)
    assert rows[0]["length_width_ratio"] == pytest.approx(
        rows[0]["length_body_width_ratio"], abs=1e-3
    )
    assert rows[0]["width_measurement_method"] == rows[0]["body_width_method"]
    assert rows[0]["body_width_sample_count"] >= 5
    assert rows[0]["area_length_width_um"] > 0


def test_body_width_uses_nonunit_resolved_xy_calibration():
    saturn = load_saturn_v571()
    mask = rotated_rectangle((96, 128), np.array([48.0, 64.0]), 60, 9, 0)
    labels = mask.astype(np.int32)
    segmentation = {
        "unet_primary_instance_labels": labels,
        "unet_primary_centerline_labels": skeletonize(mask).astype(np.int32),
        "unet_probability": np.where(mask, 0.95, 0.0).astype(np.float32),
        "unet_primary_parent_by_instance": {1: 1},
        "unet_primary_instance_sources": {1: "unet_primary"},
        "unet_primary_centerline_metadata": {1: {}},
        "unet_primary_technical_failures": [],
    }
    xy_um_per_pixel = 0.37841796875
    cfg = saturn.cfg_with_resolved_pixels(
        width_cfg(
            saturn,
            UM_PER_PX_XY=xy_um_per_pixel,
            SEGMENTATION_ENGINE="unet_primary",
            MIN_SKEL_LEN_UM=1.0,
            MAX_GEODESIC_LEN_UM=100.0,
            MAX_WIDTH_UM=100.0,
        )
    )

    measured = saturn._measure_unet_primary_instances(segmentation, cfg)
    rows = saturn.rows_from_results(
        measured["results"],
        z_idx=4,
        um=xy_um_per_pixel,
    )

    assert rows[0]["body_width_px"] == pytest.approx(9.0, abs=0.8)
    assert rows[0]["body_width_um"] == pytest.approx(
        rows[0]["body_width_px"] * xy_um_per_pixel,
        abs=1e-4,
    )


def test_short_centerline_reports_unavailable_without_fabricating_width():
    saturn = load_saturn_v571()
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:10, 6:10] = True
    result = saturn.measure_subpixel_body_width(
        mask,
        np.array([[7, 7], [7, 8]]),
        width_cfg(saturn),
    )
    assert np.isnan(result["body_width_px"])
    assert result["body_width_sample_count"] < 5
    assert np.isfinite(result["body_centerline_length_px"])


def test_representative_track_width_uses_largest_area_not_widest_plane():
    saturn = load_saturn_v571()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 2],
            "z_slice": [4, 5, 6, 9],
            "length_um_geodesic": [9.0, 9.6, 11.0, 7.5],
            "tortuosity": [1.01, 1.23, 1.80, 1.05],
            "instance_mask_area_px": [80, 120, 90, 70],
            "body_width_um": [2.4, 2.8, 4.5, 2.2],
            "body_width_p90_um": [2.8, 3.1, 5.0, 2.5],
            "body_width_iqr_um": [0.2, 0.3, 0.8, 0.2],
            "area_length_width_um": [2.3, 2.7, 4.2, 2.1],
            "body_width_sample_count": [20, 30, 25, 18],
            "body_width_method": ["chord"] * 4,
            "unet_mean_probability": [0.95, 0.90, 0.99, 0.92],
            "centerline_within_instance_mask": [True] * 4,
        }
    )
    tracks = pd.DataFrame(
        {"track_id": [1, 2], "max_length_2d": [10.0, 8.0]}
    )

    result = saturn._attach_representative_body_width(detections, tracks)

    first = result.set_index("track_id").loc[1]
    assert first["representative_width_z"] == 5
    assert first["representative_body_width_um"] == pytest.approx(2.8)
    assert first["representative_body_width_um"] != 4.5
    assert first["representative_body_length_um"] == pytest.approx(9.6)
    assert first["representative_section_tortuosity"] == pytest.approx(1.23)
    assert first["length_body_width_ratio"] == pytest.approx(9.6 / 2.8)
    assert first["length_body_width_ratio_cross_plane_legacy"] == pytest.approx(
        10.0 / 2.8
    )


def test_representative_width_tie_breaks_by_support_then_z():
    saturn = load_saturn_v571()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "z_slice": [8, 6, 7],
            "length_um_geodesic": [8.0, 8.4, 8.2],
            "instance_mask_area_px": [100, 100, 100],
            "body_width_um": [2.5, 2.6, 2.7],
            "unet_mean_probability": [0.90, 0.95, 0.95],
            "centerline_within_instance_mask": [True, True, True],
        }
    )
    tracks = pd.DataFrame({"track_id": [1], "max_length_2d": [9.0]})

    result = saturn._attach_representative_body_width(detections, tracks)

    assert result.loc[0, "representative_width_z"] == 6
    assert result.loc[0, "representative_body_width_um"] == pytest.approx(2.6)


def test_all_unavailable_track_width_retains_explicit_schema():
    saturn = load_saturn_v571()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1],
            "z_slice": [3, 4],
            "body_width_um": [np.nan, np.nan],
        }
    )
    tracks = pd.DataFrame({"track_id": [1], "max_length_2d": [8.0]})

    result = saturn._attach_representative_body_width(detections, tracks)

    assert np.isnan(result.loc[0, "representative_body_width_um"])
    assert np.isnan(result.loc[0, "representative_width_z"])
    assert result.loc[0, "representative_width_sample_count"] == 0
    assert result.loc[0, "representative_width_method"] == "unavailable"
    assert np.isnan(result.loc[0, "length_body_width_ratio"])


def test_track_ratio_pairs_length_and_width_from_same_representative_plane():
    saturn = load_saturn_v571()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1],
            "z_slice": [4, 5],
            "instance_mask_area_px": [120, 80],
            "length_um_geodesic": [8.0, 12.0],
            "body_width_um": [2.0, 3.0],
            "unet_mean_probability": [0.9, 0.95],
            "centerline_within_instance_mask": [True, True],
        }
    )
    tracks = pd.DataFrame({"track_id": [1], "max_length_2d": [12.0]})

    result = saturn._attach_representative_body_width(detections, tracks)

    assert result.loc[0, "representative_width_z"] == 4
    assert result.loc[0, "representative_body_length_um"] == pytest.approx(8.0)
    assert result.loc[0, "representative_body_width_um"] == pytest.approx(2.0)
    assert result.loc[0, "length_body_width_ratio"] == pytest.approx(4.0)
    assert result.loc[0, "length_body_width_ratio_cross_plane_legacy"] == (
        pytest.approx(6.0)
    )


def test_analysis_summary_uses_body_width_as_primary_and_labels_legacy():
    saturn = load_saturn_v571()
    tracks = pd.DataFrame(
        {
            "technical_valid": [True, True],
            "total_3d_length_um": [9.0, 11.0],
            "max_length_2d": [8.5, 10.5],
            "representative_body_width_um": [2.1, 2.5],
            "representative_body_width_p90_um": [2.4, 2.8],
            "length_body_width_ratio": [4.0, 4.4],
            "median_width_um_dt_legacy": [2.7, 2.7],
            "median_length_width_ratio_dt_legacy": [3.2, 3.8],
        }
    )
    summary = saturn.build_analysis_summary(
        track_summary=tracks,
        run_scope="full_stack_3d",
        cfg={"SEGMENTATION_ENGINE": "unet_primary"},
    )

    assert summary["median_body_width_um"] == pytest.approx(2.3)
    assert summary["median_body_width_p90_um"] == pytest.approx(2.6)
    assert summary["median_width_um_dt_legacy"] == pytest.approx(2.7)
    assert "median_2d_width_um" not in summary


def test_biological_group_comparison_exposes_one_width_metric():
    saturn = load_saturn_v571()

    width_metrics = [
        name
        for name in saturn._STUDY_COMPARISON_METRICS
        if "width" in name
    ]

    assert "median_body_width_um" in width_metrics
    assert "median_length_body_width_ratio" in width_metrics
    assert "median_body_width_p90_um" not in width_metrics
    assert "median_area_length_width_um" not in width_metrics


def test_primary_analysis_summary_hides_qc_width_variants(tmp_path):
    saturn = load_saturn_v571()
    tracks = pd.DataFrame(
        {
            "technical_valid": [True],
            "projection_z_extent_um": [9.0],
            "max_length_2d": [8.0],
            "representative_body_width_um": [2.0],
            "representative_body_width_p90_um": [2.7],
            "length_body_width_ratio": [4.0],
            "median_width_um_dt_legacy": [2.8],
            "median_length_width_ratio_dt_legacy": [3.0],
            "thickness_um": [1.0],
            "tortuosity_3d": [1.1],
            "z_span_um": [2.0],
        }
    )

    saturn.export_analysis_summary(
        tmp_path,
        track_summary=tracks,
        run_scope="full_stack_3d",
        cfg={"SEGMENTATION_ENGINE": "unet_primary"},
    )
    primary = pd.read_csv(tmp_path / "analysis_summary.csv")
    metrics = set(primary.columns)

    assert "median_body_width_um" in metrics
    assert "median_length_body_width_ratio" in metrics
    assert "median_body_width_p90_um" not in metrics
    assert "median_width_um_dt_legacy" not in metrics
