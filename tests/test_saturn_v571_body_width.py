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
    assert rows[0]["body_width_sample_count"] >= 5
    assert rows[0]["area_length_width_um"] > 0


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
    assert first["length_body_width_ratio"] == pytest.approx(10.0 / 2.8)


def test_representative_width_tie_breaks_by_support_then_z():
    saturn = load_saturn_v571()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "z_slice": [8, 6, 7],
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
            "median_width_2d": [2.7, 2.7],
            "median_length_width_ratio_2d": [3.2, 3.8],
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
