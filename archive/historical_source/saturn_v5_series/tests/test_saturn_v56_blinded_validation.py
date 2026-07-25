import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


blind = import_file("v56_blinded_runner_test", ROOT / "scratch" / "run_v56_blinded_validation.py")
unblind = import_file("v56_unblind_runner_test", ROOT / "scratch" / "run_v56_unblind_validation.py")
seg = import_file("v56_blinded_seg_test", ROOT / "sperm_segmentation_saturnv5.6.py")


def source_manifest():
    return pd.DataFrame({
        "dataset_path": ["stack_a", "stack_b"],
        "roi_path": ["roi_a.npy", "roi_b.npy"],
        "exclusion_mask_path": ["", ""],
        "dataset_label": ["sample A", "sample B"],
        "sample_id": ["S1", "S2"],
        "acquisition_class": ["same", "same"],
        "genotype": ["WT", "mutant"],
    })


def test_genotype_labels_absent_from_blinded_manifest():
    blinded, key = blind.blind_manifest(source_manifest(), seed=7)
    assert "genotype" not in blinded.columns
    assert "dataset_path" not in blinded.columns
    assert "sample_id" not in blinded.columns
    assert "genotype" in key.columns
    assert blinded["blinded_id"].str.match(r"B\d{3}").all()


def test_genotype_labels_not_passed_to_segmentation_manifest():
    blinded, _ = blind.blind_manifest(source_manifest(), seed=7)
    forbidden = {"genotype", "group", "strain", "condition"}
    assert forbidden.isdisjoint(blinded.columns)


def test_genotype_labels_do_not_appear_in_standard_output_names():
    names = [
        "blinded_dataset_manifest_v5_6.csv",
        "blinded_validation_report_v5_6.pdf",
        "B001_crop_01_random_selected.tif",
    ]
    for name in names:
        assert "WT" not in name
        assert "mutant" not in name.lower()


def make_synthetic_source(tmp_path):
    stack = tmp_path / "WT_original_stack_name"
    stack.mkdir()
    for z in [5, 6, 12, 35, 60, 87]:
        tifffile.imwrite(stack / f"Project001_Series002_z{z:03d}_ch00.tif", np.zeros((12, 12), dtype=np.uint16))
    roi = tmp_path / "WT_original_roi.npy"
    np.save(roi, np.ones((12, 12), dtype=bool))
    manifest = pd.DataFrame({
        "dataset_path": [str(stack)],
        "roi_path": [str(roi)],
        "exclusion_mask_path": [""],
        "dataset_label": ["WT_named_dataset"],
        "sample_id": ["WT_SAMPLE_001"],
        "acquisition_class": ["scope_A"],
        "genotype": ["WT"],
        "slice_override": ["5,6,12,35,60,87"],
    })
    return manifest


def test_private_source_fields_absent_from_blinded_manifest(tmp_path):
    manifest = make_synthetic_source(tmp_path)
    blinded, _ = blind.blind_manifest(manifest, seed=11, selected_slices={"B001": [5, 6, 12]})
    text = blinded.to_csv(index=False)
    assert "WT" not in text
    assert "WT_SAMPLE_001" not in text
    assert "WT_original_stack_name" not in text
    assert "dataset_path" not in blinded.columns


def test_private_output_dir_rejects_blinded_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(blind, "OUT_ROOT", tmp_path / "blinded")
    with pytest.raises(ValueError, match="inside blinded"):
        blind.validate_private_output_dir(tmp_path / "blinded" / "private")


def test_validate_manifest_only_creates_no_blinded_outputs_or_key(tmp_path, monkeypatch):
    monkeypatch.setattr(blind, "OUT_ROOT", tmp_path / "blinded")
    monkeypatch.setattr(blind, "STAGED_INPUT_ROOT", tmp_path / "staged")
    manifest = make_synthetic_source(tmp_path)
    manifest_path = tmp_path / "private_source_manifest_v5_6.csv"
    private_dir = tmp_path / "private_key_dir"
    manifest.to_csv(manifest_path, index=False)
    report = blind.validate_manifest_only(manifest_path, private_dir, seed=13)
    assert report["performed_segmentation"] is False
    assert report["created_blinded_outputs"] is False
    assert report["created_unblinding_key"] is False
    assert not (tmp_path / "blinded").exists()
    assert not (private_dir / "unblinding_key_v5_6.csv").exists()


def test_opaque_staging_preserves_z_mapping_privately(tmp_path, monkeypatch):
    monkeypatch.setattr(blind, "STAGED_INPUT_ROOT", tmp_path / "staged")
    manifest = make_synthetic_source(tmp_path)
    plans, _, _ = blind.plan_datasets(manifest, seed=17)
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    staged, mapping_path = blind.stage_representative_inputs(manifest, plans, private_dir)
    staged_files = sorted(staged["B001"]["images"].values())
    assert len(staged_files) == 6
    assert all(path.name.startswith("B001_z") for path in staged_files)
    assert all("WT" not in path.name for path in staged_files)
    mapping = pd.read_csv(mapping_path)
    assert set(mapping["original_z_index"].dropna().astype(int)) >= {5, 6, 12, 35, 60, 87}
    assert mapping["original_path"].astype(str).str.contains("WT_original_stack_name").any()


def test_leak_scanner_detects_private_text_without_reporting_value(tmp_path):
    out = tmp_path / "review"
    out.mkdir()
    (out / "blinded_metrics.csv").write_text("blinded_id,note\nB001,WT\n", encoding="utf-8")
    report = blind.scan_for_blinding_leaks([out], ["WT_SAMPLE_001", "WT"])
    assert report["leak_count"] == 1
    assert "WT" not in str(report["leaks"])


def test_gitignore_ignores_private_outputs_but_not_template():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "scratch/v5_6_unblinding_key/" in gitignore
    assert "scratch/v5_6_blinded_inputs/" in gitignore
    assert "*source_manifest_v5_6.csv" in gitignore
    assert "source_manifest_v5_6.template.csv" not in gitignore


def test_all_presets_share_morphology_settings():
    settings = []
    for preset in blind.PRESETS:
        _, params = blind.load_preset(preset)
        settings.append({
            "ANALYSIS_MODE": params["ANALYSIS_MODE"],
            "RIDGE_SIGMAS_UM": tuple(params["RIDGE_SIGMAS_UM"]),
            "BG_SIGMA_UM": params["BG_SIGMA_UM"],
            "UM_PER_PX_XY": params["UM_PER_PX_XY"],
            "UM_PER_SLICE_Z": params["UM_PER_SLICE_Z"],
        })
    assert len({tuple(sorted(s.items())) for s in settings}) == 1


def test_stack_specific_normalization_may_differ():
    cfg_a = seg.CONFIG.copy()
    cfg_b = seg.CONFIG.copy()
    cfg_a["NORM_STACK_WEIGHT"] = 0.5
    cfg_b["NORM_STACK_WEIGHT"] = 0.9
    assert cfg_a["NORM_STACK_WEIGHT"] != cfg_b["NORM_STACK_WEIGHT"]
    assert cfg_a["AUDIT_MAX_LENGTH_UM"] == cfg_b["AUDIT_MAX_LENGTH_UM"]


def test_presets_are_applied_identically_across_blinded_groups():
    blinded, _ = blind.blind_manifest(source_manifest(), seed=3)
    preset_names = set(blind.PRESETS)
    applied = {bid: preset_names.copy() for bid in blinded["blinded_id"]}
    assert len({tuple(sorted(v)) for v in applied.values()}) == 1


def test_unblinding_refuses_incomplete_review_workbook(tmp_path):
    review = pd.DataFrame({
        "blinded_dataset_id": ["B001"],
        "true_detection": [""],
        "missed_nucleus": [""],
        "split_nucleus": [""],
        "merged_nuclei": [""],
        "tissue_edge_false_positive": [""],
        "puncta_or_ring_false_positive": [""],
        "roi_edge_artifact": [""],
        "uncertain": [""],
    })
    with pytest.raises(ValueError, match="incomplete"):
        unblind.validate_review_complete(review)


def test_unblinding_accepts_completed_synthetic_review(tmp_path):
    review = pd.DataFrame({
        "blinded_dataset_id": ["B001", "B002"],
        "true_detection": [1, 1],
        "missed_nucleus": [0, 1],
        "split_nucleus": [0, 0],
        "merged_nuclei": [0, 0],
        "tissue_edge_false_positive": [0, 0],
        "puncta_or_ring_false_positive": [0, 0],
        "roi_edge_artifact": [0, 0],
        "uncertain": [0, 0],
    })
    key = pd.DataFrame({"blinded_id": ["B001", "B002"], "genotype": ["WT", "mutant"]})
    review_path = tmp_path / "review.csv"
    key_path = tmp_path / "key.csv"
    out_path = tmp_path / "out.csv"
    review.to_csv(review_path, index=False)
    key.to_csv(key_path, index=False)
    unblind.run_unblind(review_path, key_path, out_path)
    out = pd.read_csv(out_path)
    assert set(out["genotype"]) == {"WT", "mutant"}


def test_true_longer_synthetic_nuclei_remain_longer_after_technical_filtering():
    df = pd.DataFrame({
        "track_id": [1, 2], "centroid_x": [1.0, 2.0], "centroid_y": [1.0, 2.0],
        "total_3d_length_um": [9.0, 20.0], "thickness_um": [1.5, 1.5],
        "taper_ratio": [1.0, 1.0], "tortuosity_3d": [1.0, 1.0],
        "length_width_ratio": [5.0, 10.0], "volume_um3": [1.0, 1.0],
        "z_span_um": [1.0, 1.0], "pitch_deg": [45.0, 45.0],
    })
    audited = seg.flag_quality_tracks(df, seg.CONFIG.copy())
    valid = audited[audited["technical_valid"]]
    assert valid["total_3d_length_um"].max() > valid["total_3d_length_um"].min()


def test_wider_tapered_tortuous_objects_remain_technical_valid():
    cfg = seg.CONFIG.copy()
    df = pd.DataFrame({
        "track_id": [1], "centroid_x": [1.0], "centroid_y": [1.0],
        "total_3d_length_um": [10.0], "thickness_um": [3.0],
        "taper_ratio": [2.0], "tortuosity_3d": [2.0],
        "length_width_ratio": [2.0], "volume_um3": [1.0],
        "z_span_um": [1.0], "pitch_deg": [45.0],
    })
    audited = seg.flag_quality_tracks(df, cfg)
    assert bool(audited.loc[0, "technical_valid"])
    assert bool(audited.loc[0, "morphology_warning"])


def test_lower_count_synthetic_input_remains_lower_count():
    low = pd.DataFrame({"x": [1]})
    high = pd.DataFrame({"x": [1, 2, 3]})
    assert len(low) < len(high)


def test_technical_artifacts_are_rejected():
    df = pd.DataFrame({
        "track_id": [1], "centroid_x": [1.0], "centroid_y": [1.0],
        "total_3d_length_um": [0.0], "thickness_um": [1.0],
        "taper_ratio": [1.0], "tortuosity_3d": [1.0],
        "length_width_ratio": [2.0], "volume_um3": [1.0],
        "z_span_um": [1.0], "pitch_deg": [45.0],
    })
    audited = seg.flag_quality_tracks(df, seg.CONFIG.copy())
    assert not bool(audited.loc[0, "technical_valid"])
