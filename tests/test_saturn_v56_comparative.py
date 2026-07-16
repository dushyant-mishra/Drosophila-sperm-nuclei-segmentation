import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


seg = import_file("saturnv56_comparative_test", ROOT / "sperm_segmentation_saturnv5.6.py")
tuner = import_file("saturnv56_comparative_tuner_test", ROOT / "utils" / "tune_parameters_Saturnv5_6.py")


def cfg():
    out = seg.CONFIG.copy()
    out.update({
        "ANALYSIS_MODE": "comparative",
        "AUDIT_MAX_LENGTH_UM": 15.0,
        "AUDIT_MAX_THICKNESS_UM": 2.0,
        "AUDIT_MAX_TAPER_RATIO": 1.5,
        "AUDIT_MAX_TORTUOSITY": 1.5,
        "MIN_SKEL_LEN_UM": 4.0,
        "MAX_GEODESIC_LEN_UM": 20.0,
        "MIN_LENGTH_WIDTH_RATIO": 2.5,
    })
    return out


def tracks():
    return pd.DataFrame([
        {
            "track_id": 1, "centroid_x": 10.0, "centroid_y": 10.0,
            "total_3d_length_um": 9.0, "thickness_um": 1.5, "taper_ratio": 1.1,
            "tortuosity_3d": 1.1, "length_width_ratio": 4.0, "n_slices": 3,
            "volume_um3": 10.0, "z_span_um": 2.0, "pitch_deg": 45.0,
            "nearest_neighbor_um": 3.0,
        },
        {
            "track_id": 2, "centroid_x": 20.0, "centroid_y": 10.0,
            "total_3d_length_um": 24.0, "thickness_um": 1.6, "taper_ratio": 1.2,
            "tortuosity_3d": 1.2, "length_width_ratio": 6.0, "n_slices": 3,
            "volume_um3": 12.0, "z_span_um": 2.0, "pitch_deg": 50.0,
            "nearest_neighbor_um": 3.0,
        },
        {
            "track_id": 3, "centroid_x": 30.0, "centroid_y": 10.0,
            "total_3d_length_um": 10.0, "thickness_um": 3.1, "taper_ratio": 2.4,
            "tortuosity_3d": 2.2, "length_width_ratio": 1.8, "n_slices": 3,
            "volume_um3": 14.0, "z_span_um": 2.0, "pitch_deg": 90.0,
            "nearest_neighbor_um": 3.0,
        },
        {
            "track_id": 4, "centroid_x": 40.0, "centroid_y": 10.0,
            "total_3d_length_um": 0.0, "thickness_um": 1.0, "taper_ratio": 1.0,
            "tortuosity_3d": 1.0, "length_width_ratio": 3.0, "n_slices": 3,
            "volume_um3": 8.0, "z_span_um": 1.0, "pitch_deg": 30.0,
            "nearest_neighbor_um": 3.0,
        },
    ])


def test_comparative_mode_preserves_long_mutant_population():
    audited = seg.flag_quality_tracks(tracks(), cfg())
    long_track = audited.loc[audited["track_id"] == 2].iloc[0]
    assert bool(long_track["technical_valid"])
    assert bool(long_track["morphology_warning"])
    assert "long" in long_track["morphology_warning_reasons"]


def test_wide_tapered_tortuous_low_lwr_tracks_are_warnings_not_removed():
    audited = seg.flag_quality_tracks(tracks(), cfg())
    unusual = audited.loc[audited["track_id"] == 3].iloc[0]
    assert bool(unusual["technical_valid"])
    assert bool(unusual["morphology_warning"])
    for token in ("wide", "high_taper", "high_tortuosity", "low_length_width_ratio"):
        assert token in unusual["morphology_warning_reasons"]


def test_lower_count_input_is_not_compensated_to_target_count():
    local_cfg = cfg()
    rows = [{"length_px_geodesic": 30.0, "width_px": 3.0, "length_width_ratio": 10.0}]
    segs = [
        ({"mask_clean": np.zeros((8, 8), dtype=bool), "mask_hyst": np.zeros((8, 8), dtype=bool),
          "bridge_stats": {"skeleton_pixels_before": 1, "skeleton_pixels_after": 1}},
         {"results": rows})
    ]
    summary = tuner.summarize_candidate(rows, segs, local_cfg)
    assert summary["count_median"] == 1.0
    assert summary["score"] == summary["technical_score"]


def test_identical_settings_and_stack_specific_photometry_are_separate():
    wt = cfg()
    mutant = cfg()
    wt["NORM_STACK_WEIGHT"] = 0.5
    mutant["NORM_STACK_WEIGHT"] = 0.9
    morphology_keys = ["AUDIT_MAX_LENGTH_UM", "AUDIT_MAX_THICKNESS_UM", "AUDIT_MAX_TAPER_RATIO", "AUDIT_MAX_TORTUOSITY"]
    assert all(wt[k] == mutant[k] for k in morphology_keys)
    assert wt["NORM_STACK_WEIGHT"] != mutant["NORM_STACK_WEIGHT"]


def test_reference_morphology_filter_does_not_change_technical_valid_table(tmp_path):
    audited = seg.flag_quality_tracks(tracks(), cfg())
    paths = seg.export_comparative_track_tables(str(tmp_path), audited, "test")
    technical = pd.read_csv(paths["track_summary_technical_valid_test.csv"])
    reference = pd.read_csv(paths["track_summary_reference_morphology_test.csv"])
    assert len(technical) == 3
    assert len(reference) < len(technical)


def test_sensitivity_outputs_from_presets():
    audited = seg.flag_quality_tracks(tracks(), cfg())
    summaries, sensitivity = seg.compare_preset_track_summaries({
        "conservative": audited.iloc[:2].copy(),
        "selected": audited.copy(),
        "intermediate": audited.copy(),
        "permissive": audited.copy(),
    })
    assert {"conservative", "selected", "intermediate", "permissive"} <= set(summaries["preset"])
    assert "detected_by_all_presets" in sensitivity


def test_blinded_segmentation_manifest_hides_genotype_labels():
    manifest = pd.DataFrame({
        "dataset_path": ["a", "b"],
        "roi_path": ["ra", "rb"],
        "genotype": ["WT", "mutant"],
    })
    blinded, reveal = seg.assign_blinded_dataset_ids(manifest, seed=1)
    assert "genotype" not in blinded.columns
    assert "genotype" in reveal.columns
    assert blinded["blinded_dataset_id"].str.startswith("anon_").all()


def test_blinded_review_sheet_has_required_manual_columns():
    sheet = seg.make_blinded_review_sheet([
        {"blinded_dataset_id": "anon_001", "crop_id": "c1", "z_index": 5, "x0": 0, "y0": 0, "x1": 10, "y1": 10}
    ])
    for col in ["true_detection", "missed_nucleus", "split_nucleus", "merged_nuclei",
                "tissue_edge_false_positive", "puncta_ring_false_positive", "uncertain"]:
        assert col in sheet.columns


def test_differential_error_checks_warn_without_correcting_groups():
    table, warnings = seg.differential_error_indicators([
        {"group": "anon_001", "technical_failure_fraction": 0.0, "morphology_warning_fraction": 0.1},
        {"group": "anon_002", "technical_failure_fraction": 0.4, "morphology_warning_fraction": 0.1},
    ], warning_threshold=0.15)
    assert len(table) == 2
    assert any("technical_failure_fraction" in w for w in warnings)


def test_technical_artifacts_are_still_removed():
    audited = seg.flag_quality_tracks(tracks(), cfg())
    artifact = audited.loc[audited["track_id"] == 4].iloc[0]
    assert not bool(artifact["technical_valid"])
    assert "invalid_length" in artifact["technical_failure_reasons"]
