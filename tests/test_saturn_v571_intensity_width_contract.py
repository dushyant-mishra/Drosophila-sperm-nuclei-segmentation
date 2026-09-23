import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_intensity_width_contract_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_cfg(module, **updates):
    cfg = module.CONFIG.copy()
    cfg.update(
        {
            "INTENSITY_WIDTH_ENABLE": True,
            "INTENSITY_WIDTH_PROFILE_HALF_EXTENT_PX": 8.0,
            "INTENSITY_WIDTH_PROFILE_STEP_PX": 0.1,
            "INTENSITY_WIDTH_BACKGROUND_OFFSET_PX": 6.0,
            "BODY_WIDTH_ENDPOINT_TRIM_FRACTION": 0.1,
            "BODY_WIDTH_SAMPLE_SPACING_PX": 1.0,
            "BODY_WIDTH_SMOOTH_SIGMA_PX": 0.0,
            "BODY_WIDTH_MIN_SAMPLES": 5,
            "UM_PER_PX_XY": 0.5,
            "INTENSITY_WIDTH_PSF_CORRECTION_ENABLE": False,
        }
    )
    cfg.update(updates)
    return cfg


def filament_image(neighbor=False):
    rows, columns = np.indices((72, 96), dtype=float)
    image = np.exp(-0.5 * ((rows - 32.0) / 1.25) ** 2)
    image *= ((columns >= 16) & (columns <= 80))
    if neighbor:
        second = 1.4 * np.exp(-0.5 * ((rows - 39.0) / 1.25) ** 2)
        image += second * ((columns >= 16) & (columns <= 80))
    mask = np.zeros(image.shape, dtype=bool)
    mask[27:37, 14:83] = True
    centerline = np.column_stack(
        [np.full(55, 32, dtype=int), np.arange(20, 75, dtype=int)]
    )
    return image, mask, centerline


def test_profile_signal_is_instance_scoped_and_brightness_is_not_primary_width():
    saturn = load_saturn()
    image, mask, centerline = filament_image(neighbor=False)
    isolated = saturn.measure_intensity_profile_width(
        image, mask, centerline, profile_cfg(saturn)
    )
    neighboring, _, _ = filament_image(neighbor=True)
    with_neighbor = saturn.measure_intensity_profile_width(
        neighboring, mask, centerline, profile_cfg(saturn)
    )
    brighter = saturn.measure_intensity_profile_width(
        image * 2.0, mask, centerline, profile_cfg(saturn)
    )

    assert with_neighbor["intensity_fwhm_width_px"] == pytest.approx(
        isolated["intensity_fwhm_width_px"], abs=0.15
    )
    assert with_neighbor["intensity_profile_signal_au"] == pytest.approx(
        isolated["intensity_profile_signal_au"], rel=0.05
    )
    assert brighter["intensity_fwhm_width_px"] == pytest.approx(
        isolated["intensity_fwhm_width_px"], abs=0.05
    )
    assert brighter["intensity_profile_signal_au"] == pytest.approx(
        isolated["intensity_profile_signal_au"] * 2.0, rel=0.03
    )
    assert np.isnan(isolated["intensity_deconvolved_width_um"])
    assert isolated["intensity_signal_comparability"] == (
        "technical_qc_only_no_independent_staining_reference"
    )


def test_representative_signal_width_uses_same_largest_area_plane_for_ratio():
    saturn = load_saturn()
    detections = pd.DataFrame(
        {
            "track_id": [1, 1],
            "z_slice": [4, 5],
            "instance_mask_area_px": [120.0, 80.0],
            "length_um_geodesic": [8.0, 12.0],
            "intensity_fwhm_width_um": [0.8, 1.5],
            "intensity_width_sample_count": [20, 20],
            "intensity_width_method": ["fwhm", "fwhm"],
            "intensity_profile_signal_au": [10.0, 20.0],
            "tortuosity": [1.05, 1.40],
            "unet_mean_probability": [0.90, 0.95],
            "centerline_within_instance_mask": [True, True],
        }
    )
    tracks = pd.DataFrame({"track_id": [1], "max_length_2d": [12.0]})

    result = saturn._attach_representative_body_width(detections, tracks)

    assert result.loc[0, "representative_signal_width_z"] == 4
    assert result.loc[0, "representative_signal_profile_fwhm_width_um"] == pytest.approx(0.8)
    assert result.loc[0, "representative_signal_profile_length_um"] == pytest.approx(8.0)
    assert result.loc[0, "representative_signal_profile_tortuosity"] == pytest.approx(1.05)
    assert result.loc[0, "length_signal_width_ratio"] == pytest.approx(10.0)


def test_tracking_keeps_mask_volume_and_profile_footprint_proxy_separate():
    saturn = load_saturn()
    detections = pd.DataFrame(
        {
            "track_id": [3, 3],
            "sperm_id": [1, 1],
            "z_slice": [0, 1],
            "centroid_x": [0.0, 0.0],
            "centroid_y": [0.0, 0.0],
            "length_um_geodesic": [8.0, 10.0],
            "tortuosity": [1.0, 1.0],
            "width_um": [2.0, 2.0],
            "length_width_ratio": [4.0, 5.0],
            "area_px": [16.0, 20.0],
            "instance_mask_area_px": [30.0, 40.0],
            "profile_area_px": [8.0, np.nan],
            "suspected_multi_object_merge": [False, False],
        }
    )
    cfg = {**saturn.CONFIG, "UM_PER_PX_XY": 2.0, "UM_PER_SLICE_Z": 0.5}

    _, summary = saturn._summarize_tracked_detections(detections, {}, cfg)
    track = summary.iloc[0]

    assert track["observed_slice_mask_volume_um3"] == pytest.approx(140.0)
    assert track["observed_slice_profile_footprint_proxy_um3"] == pytest.approx(16.0)
    assert track["profile_footprint_observed_slice_count"] == 1
    assert track["profile_footprint_expected_slice_count"] == 2
    assert track["volume_method"] == "sum_filled_mask_area_observed_slices_no_interpolation"
    assert track["profile_footprint_method"] == (
        "sum_length_times_signal_fwhm_observed_slices_no_fallback"
    )


def test_primary_summary_routes_signal_width_and_keeps_mask_width_as_qc():
    saturn = load_saturn()
    tracks = pd.DataFrame(
        {
            "technical_valid": [True, True],
            "projection_z_extent_um": [9.0, 11.0],
            "max_length_2d": [8.0, 10.0],
            "representative_signal_profile_length_um": [7.5, 9.5],
            "representative_signal_profile_fwhm_width_um": [0.7, 0.9],
            "representative_signal_profile_tortuosity": [1.02, 1.04],
            "length_signal_width_ratio": [7.5 / 0.7, 9.5 / 0.9],
            "representative_body_length_um": [20.0, 30.0],
            "representative_body_width_um": [1.5, 1.7],
            "representative_section_tortuosity": [1.40, 1.60],
        }
    )

    summary = saturn.build_analysis_summary(
        track_summary=tracks,
        run_scope="full_stack_3d",
        cfg={"SEGMENTATION_ENGINE": "unet_primary"},
    )

    assert summary["median_signal_profile_fwhm_width_um"] == pytest.approx(0.8)
    assert summary["median_representative_section_length_um"] == pytest.approx(8.5)
    assert summary["median_representative_section_tortuosity"] == pytest.approx(1.03)
    assert summary["median_length_signal_width_ratio"] == pytest.approx(
        np.median([7.5 / 0.7, 9.5 / 0.9])
    )
    assert summary["median_body_mask_chord_width_um_qc"] == pytest.approx(1.6)
    assert "median_body_width_um" not in summary


def divergent_plane_tracks():
    return pd.DataFrame(
        {
            "track_id": [1, 2],
            "technical_valid": [True, True],
            "representative_body_length_um": [20.0, 30.0],
            "representative_signal_profile_length_um": [7.5, 9.5],
            "representative_signal_profile_fwhm_width_um": [0.7, 0.9],
            "length_signal_width_ratio": [7.5 / 0.7, 9.5 / 0.9],
            "representative_signal_profile_tortuosity": [1.02, 1.04],
            "tortuosity_3d": [1.10, 1.20],
            "projection_z_extent_um": [8.0, 10.0],
            "z_span_um": [1.0, 2.0],
            "z_covered_um": [2.0, 3.0],
            "observed_slice_mask_volume_um3": [20.0, 24.0],
            "observed_slab_effective_thickness_um": [1.4, 1.8],
            "thickness_um": [1.0, 1.2],
            "pitch_deg": [5.0, 10.0],
            "taper_ratio": [1.1, 1.2],
            "nearest_neighbor_um": [4.0, 5.0],
        }
    )


def test_excel_biologist_sheet_uses_signal_plane_length(tmp_path):
    saturn = load_saturn()
    saturn.generate_excel_report(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        divergent_plane_tracks(),
    )
    workbook = tmp_path / f"batch_analysis_results_{saturn._VERSION}.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    length_cell = sheet.find(".//x:c[@r='B5']/x:v", namespace)

    assert length_cell is not None
    assert float(length_cell.text) == pytest.approx(8.5)


def test_batch_pdf_uses_signal_plane_length(tmp_path):
    saturn = load_saturn()
    detections = pd.DataFrame(
        {
            "z_slice": [0],
            "length_um_geodesic": [8.0],
            "width_um": [1.0],
            "length_width_ratio": [8.0],
            "detection_source": ["unet_primary"],
        }
    )
    slice_summary = pd.DataFrame(
        {"z_slice": [0], "n_spermatids": [1], "median_length_um": [8.0]}
    )

    saturn.generate_batch_report(
        tmp_path,
        detections,
        slice_summary,
        {"xy": 1.0, "z": 1.0},
        divergent_plane_tracks(),
        generate_pptx=False,
    )
    report = (
        tmp_path
        / "technical_qc"
        / f"batch_technical_qc_report_{saturn._VERSION}.pdf"
    )
    text = "\n".join(page.extract_text() or "" for page in PdfReader(report).pages)

    assert "8.50 um" in text
    assert "25.00 um" not in text


def test_concise_report_contract_uses_one_signal_width(tmp_path):
    source = (ROOT / "scripts" / "generate_v57_biological_comparison.py").read_text(
        encoding="utf-8"
    )
    concise = source.split('if args.metric_profile == "concise_v571":', 1)[1]
    concise = concise.split("else:", 1)[0]

    assert '"median_signal_profile_fwhm_width_um"' in concise
    assert '"median_length_signal_width_ratio"' in concise
    assert '"median_body_width_um"' not in concise
    assert "integrated_density" not in concise
