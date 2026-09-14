"""Contract for the signal-width availability bias validator.

The hardened intensity profile refuses to measure objects whose half-maximum
crossing is not seen on both sides inside their own mask. That is correct, but
it makes the measured set a subset. These tests pin the arithmetic that decides
whether the subset is biased, because a wrong answer here would be read as
permission to compare groups on a biased sample.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_v571_width_availability_bias.py"
SPEC = importlib.util.spec_from_file_location("v571_width_availability_bias", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _detection_frame(sample_id, group, widths, mask_widths, methods):
    """One specimen's detections on a single sampled plane."""
    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "group": group,
            "z_slice": 35,
            "intensity_fwhm_width_um": widths,
            "body_width_px": np.asarray(mask_widths, dtype=float) / 0.5,
            "um_per_px_xy": 0.5,
            "roi_area_um2": 1000.0,
            "intensity_width_method": methods,
            "centroid_x": np.arange(len(widths), dtype=float) * 10.0,
            "centroid_y": np.zeros(len(widths), dtype=float),
        }
    )


def test_sampled_planes_avoids_stack_extremes_and_honours_count():
    planes = MODULE.sampled_planes(0, 100, 3)
    assert len(planes) == 3
    assert min(planes) >= 20 and max(planes) <= 80
    assert planes == sorted(planes)


def test_sampled_planes_degrades_gracefully_on_a_tiny_stack():
    planes = MODULE.sampled_planes(4, 5, 3)
    assert planes
    assert all(4 <= p <= 5 for p in planes)


def test_unavailable_fraction_and_selection_bias_are_computed_correctly():
    # Two of four measured. The unavailable pair has narrower masks, so the
    # measured subset skews wide and the bias must be negative.
    frame = _detection_frame(
        "s1",
        "KJ",
        widths=[0.70, 0.80, np.nan, np.nan],
        mask_widths=[1.5, 1.7, 1.0, 1.2],
        methods=[
            "raw_intensity_fwhm_perpendicular_central_body",
            "raw_intensity_fwhm_perpendicular_central_body",
            "unavailable_boundary_clipped_profiles",
            "unavailable_short_centerline",
        ],
    )
    summary = MODULE.summarize(frame)
    row = summary.iloc[0]

    assert row["detections"] == 4
    assert row["width_available"] == 2
    assert row["width_unavailable_fraction"] == pytest.approx(0.5)
    assert row["boundary_clipped_fraction"] == pytest.approx(0.25)
    assert row["median_mask_width_um_available"] == pytest.approx(1.6)
    assert row["median_mask_width_um_unavailable"] == pytest.approx(1.1)
    # Negative means the dropped objects were narrower than the measured ones.
    assert row["mask_width_selection_bias_um"] == pytest.approx(-0.5)
    assert json.loads(row["unavailable_reasons"]) == {
        "unavailable_boundary_clipped_profiles": 1,
        "unavailable_short_centerline": 1,
    }


def test_selection_bias_is_positive_when_wide_objects_drop_out():
    """Adversarial: the dangerous direction is losing the wide, merged objects."""
    frame = _detection_frame(
        "s2",
        "WT",
        widths=[0.70, 0.72, np.nan, np.nan],
        mask_widths=[1.4, 1.5, 3.0, 4.0],
        methods=[
            "raw_intensity_fwhm_perpendicular_central_body",
            "raw_intensity_fwhm_perpendicular_central_body",
            "unavailable_boundary_clipped_profiles",
            "unavailable_boundary_clipped_profiles",
        ],
    )
    row = MODULE.summarize(frame).iloc[0]
    assert row["mask_width_selection_bias_um"] > 0
    assert row["mask_width_selection_bias_um"] == pytest.approx(2.05)


def test_group_contrast_reports_direction_and_refuses_tiny_groups():
    summary = pd.DataFrame(
        {
            "group": ["KJ", "KJ", "KJ", "WT", "WT", "WT"],
            "width_unavailable_fraction": [0.10, 0.12, 0.11, 0.30, 0.32, 0.31],
        }
    )
    contrast = MODULE.group_contrast(summary, "width_unavailable_fraction")
    # Groups sort alphabetically, so the difference is WT minus KJ.
    assert contrast["difference_of_means"] == pytest.approx(0.20, abs=1e-9)
    assert contrast["welch_p_value"] < 0.01

    too_small = pd.DataFrame(
        {"group": ["KJ", "WT"], "width_unavailable_fraction": [0.1, 0.3]}
    )
    assert "note" in MODULE.group_contrast(too_small, "width_unavailable_fraction")


def test_group_contrast_refuses_a_design_that_is_not_two_groups():
    summary = pd.DataFrame(
        {
            "group": ["KJ", "KJ", "WT", "WT", "RESCUE", "RESCUE"],
            "width_unavailable_fraction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    assert "note" in MODULE.group_contrast(summary, "width_unavailable_fraction")


def test_nearest_neighbour_distance_matches_known_spacing():
    frame = _detection_frame(
        "s3",
        "KJ",
        widths=[0.7, 0.7, 0.7],
        mask_widths=[1.5, 1.5, 1.5],
        methods=["raw_intensity_fwhm_perpendicular_central_body"] * 3,
    )
    # Centroids sit 10 px apart along x, at 0.5 um per pixel.
    assert MODULE.nearest_neighbour_um(frame) == pytest.approx(5.0)


def test_spearman_returns_none_below_the_minimum_sample():
    short = pd.Series([0.1, 0.2, 0.3])
    assert MODULE.spearman_or_none(short, short) is None


def test_validator_fails_closed_on_a_populated_output_directory(tmp_path):
    """Evidence directories are append-only; overwriting one destroys provenance."""
    output = tmp_path / "existing"
    output.mkdir()
    (output / "prior_evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        MODULE.main(["--output", str(output)])
