import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_component_distance_transform_matches_full_frame():
    saturn = load_saturn_v57()
    mask = np.zeros((96, 112), dtype=bool)
    mask[15:35, 18:45] = True
    mask[20:30, 25:38] = False

    expected = distance_transform_edt(mask)
    actual = saturn._distance_transform_component(mask)

    np.testing.assert_array_equal(actual, expected)


def test_post_detection_qc_separates_warning_free_from_reference(tmp_path):
    saturn = load_saturn_v57()
    detections = pd.DataFrame({"track_link_method": ["new", "overlap", "overlap"]})
    tracks = pd.DataFrame(
        {
            "quality_flags": ["", "wide", "unusual_pitch"],
            "reference_morphology_pass": [True, True, False],
            "is_biological_candidate": [True, True, True],
            "has_warning_only": [False, True, True],
        }
    )

    saturn.export_post_detection_qc(tmp_path, detections, tracks)
    report = (tmp_path / "post_detection_qc.txt").read_text(encoding="utf-8")

    assert "warning_free: 1" in report
    assert "reference_morphology_pass: 2" in report
    assert "\n  quality:" not in report


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
    result = saturn._study_parse_leica_metadata(tmp_path, 2, 9.0, 9.0)
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
            }
        ).to_csv(output / "measurements_with_tracks_v5.7.csv", index=False)
        pd.DataFrame(
            {
                "track_id": [1, 2],
                "is_biological_candidate": [True, False],
                "is_quality_track": [True, False],
                "total_3d_length_um": [9.4, 10.2],
                "tortuosity_3d": [1.1, 1.2],
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
    assert set(summary["biological_candidate_track_count"]) == {1}
    assert set(summary["roi_area_um2"].round(6)) == {198.0}
    assert set(summary["sampled_depth_um"].round(6)) == {3.12}
    assert set(summary["stack_span_um"].round(6)) == {2.08}
    assert set(summary["sampled_roi_volume_um3"].round(6)) == {617.76}
    assert set(summary["all_3d_tracks_per_1000_um2"].round(6)) == {10.10101}
    assert set(summary["all_3d_tracks_per_100000_um3"].round(6)) == {323.750324}
    assert set(summary["unet_associated_3d_track_count"]) == {1}
    assert set(summary["biological_unet_associated_track_count"]) == {0}
    assert set(summary["z_boundary_track_count"]) == {2}
    assert set(summary["z_boundary_track_fraction"]) == {1.0}
    assert (output_root / "study_manifest.csv").exists()
    assert (output_root / "specimen_summary.csv").exists()
    assert (output_root / "group_summary.csv").exists()
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

    saturn.run_multisample_study(
        rows,
        output_root,
        base_cfg={"UM_PER_PX_XY": 0.75, "UM_PER_SLICE_Z": 1.04},
        batch_runner=fake_batch_runner,
    )
    assert len(calls) == 2
    assert all(len(list((output_root / "samples" / row["sample_id"]).glob("attempt_*"))) == 1 for row in rows)
