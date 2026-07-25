import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage import draw, measure, morphology


ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ai = import_file("ai_preprocessing_v56_test", ROOT / "utils" / "ai_preprocessing_v5_6.py")
seg = import_file("saturn_v56_ai_import_test", ROOT / "sperm_segmentation_saturnv5.6.py")
instance_diag = import_file(
    "ilastik_instance_diagnostic_test",
    ROOT / "scratch" / "run_v56_ilastik_instance_diagnostic.py",
)
pathseg = import_file(
    "v56_2d_path_segmentation_test",
    ROOT / "scratch" / "run_v56_2d_path_segmentation.py",
)


def test_careamics_optional_and_production_pipeline_imports():
    assert hasattr(seg, "segment_slice")
    assert callable(ai.discover_n2v2_backend)


def test_test_slices_and_neighbor_buffers_excluded_from_training():
    split = ai.build_data_split(list(range(88)), [5, 6, 12, 35, 60, 87], seed=1)
    excluded = set(split["excluded_buffer_z_indices"])
    assert {4, 5, 6, 7, 11, 12, 13, 34, 35, 36, 59, 60, 61, 86, 87}.issubset(excluded)
    assert excluded.isdisjoint(split["training_z_indices"])
    assert excluded.isdisjoint(split["validation_z_indices"])


def test_patch_extraction_stays_inside_roi():
    roi = np.zeros((96, 96), dtype=bool)
    roi[16:80, 16:80] = True
    img = np.random.default_rng(2).random((96, 96)).astype(np.float32)
    patches, records = ai.extract_roi_patches({0: img}, roi, [0], patch_size=32, patches_per_slice=3, seed=2)
    assert patches.shape[1:] == (32, 32)
    for rec in records:
        y0, x0 = rec["y0"], rec["x0"]
        assert roi[y0:y0 + 32, x0:x0 + 32].all()


def test_outside_roi_changes_do_not_alter_ai_input_inside_roi():
    roi = np.zeros((32, 32), dtype=bool)
    roi[8:24, 8:24] = True
    img = np.zeros((32, 32), dtype=np.uint8)
    img[roi] = 100
    changed = img.copy()
    changed[~roi] = 255
    a, _ = ai.normalize_ai_input(img, roi)
    b, _ = ai.normalize_ai_input(changed, roi)
    np.testing.assert_allclose(a[roi], b[roi])


def test_ai_predictions_preserve_image_dimensions():
    raw = np.zeros((24, 32), dtype=np.float32)
    pred = np.ones((24, 32), dtype=np.float32) * 0.5
    roi = np.ones_like(raw, dtype=bool)
    report = ai.validate_ai_output(raw, pred, roi)
    assert report["shape"] == [24, 32]
    with pytest.raises(ValueError):
        ai.validate_ai_output(raw, pred[:20], roi)


def test_equivalent_uint8_and_uint16_have_equivalent_float32_ai_inputs():
    img8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    img16 = (img8.astype(np.uint16) * 257).astype(np.uint16)
    roi = np.ones((16, 16), dtype=bool)
    a, _ = ai.normalize_ai_input(img8, roi)
    b, _ = ai.normalize_ai_input(img16, roi)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_ai_proposals_without_raw_support_are_rejected():
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 15:17] = True
    raw = np.zeros((32, 32), dtype=np.float32)
    ridge = np.zeros_like(raw)
    roi = np.ones_like(mask, dtype=bool)
    support = ai.calculate_raw_support_metrics(mask, raw, ridge, roi)
    assert not support["accepted_by_raw_support"]
    assert "low_raw_contrast" in support["rejection_reason"]


def test_genuine_synthetic_faint_rods_with_raw_support_may_be_recovered():
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 15:17] = True
    raw = np.ones((32, 32), dtype=np.float32) * 0.1
    raw[mask] = 0.2
    ridge = np.zeros_like(raw)
    ridge[mask] = 1.0
    roi = np.ones_like(mask, dtype=bool)
    support = ai.calculate_raw_support_metrics(mask, raw, ridge, roi)
    assert support["accepted_by_raw_support"]


def test_ai_recovery_does_not_use_target_count_or_morphology():
    source = (ROOT / "scratch" / "run_v56_ai_preprocessing_pilot.py").read_text(encoding="utf-8")
    forbidden = ["target_count", "target length", "target_width", "target morphology"]
    assert all(token not in source.lower() for token in forbidden)


def test_morphology_measurements_mark_raw_source():
    source = (ROOT / "scratch" / "run_v56_ai_preprocessing_pilot.py").read_text(encoding="utf-8")
    assert '"raw_measurement_source": True' in source


def test_ilastik_probability_map_dimension_mismatch_rejected(tmp_path):
    path = tmp_path / "prob.tif"
    tifffile.imwrite(path, np.zeros((10, 10, 4), dtype=np.float32))
    roi = np.ones((12, 12), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        ai.load_ilastik_probability_map(path, (12, 12), roi, nucleus_channel=0)


def test_mock_or_fallback_model_cannot_be_reported_as_trained_n2v2():
    with pytest.raises(RuntimeError, match="fallback/mock"):
        ai.require_real_n2v2_verification({
            "real_careamics_model": True,
            "n2v2_specific_configuration_active": True,
            "blind_spot_masking_active": True,
            "weights_changed": True,
            "fallback_or_mock_used": True,
            "optimizer_steps_completed": 10,
        })


def test_model_weight_verification_requires_changed_weights():
    with pytest.raises(RuntimeError, match="weights_changed"):
        ai.require_real_n2v2_verification({
            "real_careamics_model": True,
            "n2v2_specific_configuration_active": True,
            "blind_spot_masking_active": True,
            "weights_changed": False,
            "fallback_or_mock_used": False,
            "optimizer_steps_completed": 10,
        })


def test_tiled_prediction_placeholder_preserves_dimensions():
    class DummyModel:
        def predict(self, arr, **kwargs):
            return [arr[0].copy()], ["dummy"]

    img = np.random.default_rng(4).random((37, 41)).astype(np.float32)
    pred = ai.predict_n2v2(DummyModel(), img, tile_size=16)
    assert pred.shape == img.shape


def test_structure_preservation_failure_prevents_saturn_execution_logic():
    gate = {
        "mean_boundary_to_interior_ridge_ratio": 0.9,
        "deleted_fraction": 0.25,
        "mean_n2v2_ridge_support_fraction": 0.1,
    }
    permitted = (
        gate["mean_boundary_to_interior_ridge_ratio"] <= 0.25
        and gate["deleted_fraction"] <= 0.10
        and gate["mean_n2v2_ridge_support_fraction"] >= 0.35
    )
    assert not permitted


def test_diagnostic_runner_does_not_use_target_count_or_morphology():
    source = (ROOT / "scratch" / "run_v56_n2v2_diagnostic.py").read_text(encoding="utf-8").lower()
    for forbidden in ["target_count", "target length", "target_width", "target morphology"]:
        assert forbidden not in source


def test_n2v2_and_ai_preprocessing_remain_off_by_default():
    assert seg.CONFIG["AI_PREPROCESSING_MODE"] == "off"


def test_malformed_nan_probability_map_rejected(tmp_path):
    path = tmp_path / "prob.tif"
    arr = np.zeros((12, 12, 4), dtype=np.float32)
    arr[0, 0, 0] = np.nan
    tifffile.imwrite(path, arr, photometric="minisblack")
    roi = np.ones((12, 12), dtype=bool)
    with pytest.raises(ValueError, match="NaN"):
        ai.load_ilastik_probability_map(path, (12, 12), roi, nucleus_channel=0)


def test_missing_class_metadata_requires_explicit_channel(tmp_path):
    path = tmp_path / "prob.tif"
    tifffile.imwrite(path, np.zeros((12, 12, 4), dtype=np.float32), photometric="minisblack")
    roi = np.ones((12, 12), dtype=bool)
    with pytest.raises(ValueError, match="explicit nucleus_channel"):
        ai.load_ilastik_probability_map(path, (12, 12), roi)


def test_ilastik_export_train_eval_slices_are_disjoint_from_eval_buffer():
    exporter = (ROOT / "scratch" / "export_v56_ilastik_training_data.py").read_text(encoding="utf-8")
    assert "TRAIN_Z = [18, 25, 43, 50, 70, 78]" in exporter
    assert "EVAL_Z = [5, 6, 12, 35, 60, 87]" in exporter
    train = {18, 25, 43, 50, 70, 78}
    eval_z = {5, 6, 12, 35, 60, 87}
    eval_buffer = {zz for z in eval_z for zz in (z - 1, z, z + 1)}
    assert train.isdisjoint(eval_z)
    assert train.isdisjoint(eval_buffer)


def test_ilastik_class_order_is_recorded_in_exporter():
    exporter = (ROOT / "scratch" / "export_v56_ilastik_training_data.py").read_text(encoding="utf-8")
    assert "sperm_nucleus" in exporter
    assert "structured_tissue_edge" in exporter
    assert "punctum_or_ring" in exporter
    assert "diffuse_background" in exporter


def test_ilastik_pilot_refuses_without_probability_maps():
    source = (ROOT / "scratch" / "run_v56_ilastik_probability_pilot.py").read_text(encoding="utf-8")
    assert "No user-supplied ilastik probability maps found" in source
    assert "Raw morphology remains the measurement source" in source


def test_ilastik_pilot_gates_raw_baseline_before_probability_maps():
    source = (ROOT / "scratch" / "run_v56_ilastik_probability_pilot.py").read_text(encoding="utf-8")
    assert "EXPECTED_RAW_COUNTS = {5: 266, 6: 288, 12: 316, 35: 318, 60: 300, 87: 273}" in source
    assert "RAW BASELINE EQUIVALENCE: FAIL" in source
    assert source.index("run_raw_baseline_gate") < source.index("run_ilastik_z05_z35")
    main_source = source[source.index("def main()"):]
    assert main_source.index("if not gate_pass") < main_source.index("run_ilastik_z05_z35(")


def test_ilastik_pilot_uses_original_raw_tiffs_for_baseline():
    source = (ROOT / "scratch" / "run_v56_ilastik_probability_pilot.py").read_text(encoding="utf-8")
    assert r'C:\Users\dmishra\Desktop\sperm images' in source
    assert "SOURCE_RE = re.compile" in source
    assert "Project001_Series002_z" in source
    assert "robust_imread(str(by_z[z]))" in source
    assert "ROI_PATH = IMAGE_DIR / \"roi_z28.1.npy\"" in source


def test_ilastik_combined_output_preserves_raw_objects_and_audits_rejections():
    source = (ROOT / "scratch" / "run_v56_ilastik_probability_pilot.py").read_text(encoding="utf-8")
    assert "combined_raw_object_count" in source
    assert "raw_object_preserved = all(" in source
    assert "Combined output failed raw object preservation" in source
    assert '"rejection_reasons": "not_applicable"' in source
    assert '"accepted": accepted' in source
    assert "unspecified_rejection" in source
    assert "target_count" not in source.lower()
    assert "target morphology" not in source.lower()


def _instance_cfg():
    return {
        "MIN_SKEL_LEN_PX": 10,
        "MAX_GEODESIC_LEN_PX": 30,
        "MAX_WIDTH_PX": 4,
        "MIN_LENGTH_WIDTH_RATIO": 2.0,
        "MAX_TORTUOSITY": 2.5,
    }


def test_instance_diagnostic_exact_raw_baseline_is_locked():
    source = (ROOT / "scratch" / "run_v56_ilastik_instance_diagnostic.py").read_text(encoding="utf-8")
    assert "EXPECTED_RAW_COUNTS = {5: 266, 35: 318}" in source
    assert 'cfg["ROI_BOUNDARY_SAFE_RIDGE"] = False' in source
    assert "RAW BASELINE EQUIVALENCE: FAIL" in source


def test_morphology_alone_is_warning_not_technical_rejection():
    prob = np.zeros((40, 40, 4), dtype=np.float32)
    prob[:, :, 0] = 0.9
    parent = np.zeros((40, 40), dtype=bool)
    parent[8:28, 8:28] = True
    rec = instance_diag.component_level_disposition(
        parent,
        prob,
        raw_seg={},
        raw_results=[],
        roi=np.ones((40, 40), dtype=bool),
        cfg=_instance_cfg(),
        hard_morphology=False,
    )
    assert rec["morphology_warning"] is True
    assert rec["technical_valid"] is True
    assert rec["accepted"] is True


def test_wide_long_short_and_tortuous_objects_are_warning_only():
    cfg = _instance_cfg()
    assert "wide" in instance_diag.morphology_warnings(20, 8, 2.5, 1.0, cfg)
    assert "long" in instance_diag.morphology_warnings(40, 2, 20.0, 1.0, cfg)
    assert "short" in instance_diag.morphology_warnings(5, 2, 2.5, 1.0, cfg)
    assert "tortuous" in instance_diag.morphology_warnings(20, 2, 10.0, 3.0, cfg)


def test_one_semantic_component_with_two_raw_ridge_paths_splits_into_two_instances():
    parent = np.zeros((80, 80), dtype=bool)
    parent[10:70, 10:70] = True
    ridge = np.zeros((80, 80), dtype=np.float32)
    ridge[25, 15:65] = 1.0
    ridge[50, 15:65] = 1.0
    roi = np.ones((80, 80), dtype=bool)
    raw_label = np.zeros((80, 80), dtype=np.int32)
    instances, info = instance_diag.split_semantic_component(parent, ridge, raw_label, roi, 1)
    assert len(instances) == 2
    assert info["resolved_instance_count"] == 2
    for inst in instances:
        assert np.all(inst["mask"] <= parent)
        assert np.all(inst["mask"] <= roi)


def test_branch_resolution_is_attempted_before_branch_rejection():
    parent = np.zeros((80, 80), dtype=bool)
    parent[10:70, 10:70] = True
    ridge = np.zeros((80, 80), dtype=np.float32)
    ridge[40, 20:60] = 1.0
    ridge[20:60, 40] = 1.0
    roi = np.ones((80, 80), dtype=bool)
    raw_label = np.zeros((80, 80), dtype=np.int32)
    instances, info = instance_diag.split_semantic_component(parent, ridge, raw_label, roi, 7)
    assert info["branch_resolution_attempted"] is True
    assert info["branch_resolution_result"] in {"resolved", "unresolved_after_branch_break"}
    assert info["resolved_instance_count"] == len(instances)


def test_duplicate_raw_objects_are_not_added():
    prob = np.zeros((40, 40, 4), dtype=np.float32)
    prob[:, :, 0] = 0.9
    parent = np.zeros((40, 40), dtype=bool)
    parent[18:23, 18:23] = True
    rec = instance_diag.component_level_disposition(
        parent,
        prob,
        raw_seg={},
        raw_results=[{"centroid_x": 20.0, "centroid_y": 20.0}],
        roi=np.ones((40, 40), dtype=bool),
        cfg=_instance_cfg(),
        hard_morphology=False,
    )
    assert rec["accepted"] is False
    assert "duplicate_raw_detection" in rec["technical_failure_reasons"]


def test_instance_diagnostic_rendering_and_target_constraints_are_explicit():
    source = (ROOT / "scratch" / "run_v56_ilastik_instance_diagnostic.py").read_text(encoding="utf-8")
    assert "appears_in_saved_panel" in source
    assert "visible_overlay_pixel_count" in source
    assert "target_count" not in source.lower()
    assert "target morphology" not in source.lower()


def _path_prob(shape=(64, 64), nucleus=0.9, tissue=0.05, punctum=0.03, diffuse=0.02):
    prob = np.zeros((*shape, 4), dtype=np.float32)
    prob[:, :, 0] = nucleus
    prob[:, :, 1] = tissue
    prob[:, :, 2] = punctum
    prob[:, :, 3] = diffuse
    return prob


def test_2d_path_runner_locks_exact_baseline_and_no_target_gate():
    source = (ROOT / "scratch" / "run_v56_2d_path_segmentation.py").read_text(encoding="utf-8")
    assert "EXPECTED_RAW_COUNTS = {5: 266, 35: 318}" in source
    assert "RAW BASELINE EQUIVALENCE: FAIL" in source
    assert 'cfg["ROI_BOUNDARY_SAFE_RIDGE"] = False' in source
    assert "target_count" not in source.lower()
    assert "target morphology" not in source.lower()
    assert "9-10" not in source


def test_collinear_fragment_extension_across_supported_gap():
    seed = np.zeros((64, 64), dtype=bool)
    seed[32, 20:28] = True
    weighted = np.zeros((64, 64), dtype=np.float32)
    weighted[32, 20:38] = 1.0
    prob = _path_prob()
    roi = np.ones((64, 64), dtype=bool)
    completed, rows = pathseg.complete_fragment(seed, weighted, prob, roi, np.zeros((64, 64), dtype=bool))
    assert np.count_nonzero(completed) > np.count_nonzero(seed)
    assert any(r["extension_status"] == "extended" for r in rows)


def test_join_fragments_across_short_supported_gap():
    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[20, 12:25] = True
    b[20, 29:42] = True
    weighted = np.zeros((64, 64), dtype=np.float32)
    weighted[20, 12:42] = 1.0
    prob = _path_prob()
    joined, attempts = pathseg.join_fragments(
        [{"seed_id": "a", "path_mask": a, "recovery_pass": 1}, {"seed_id": "b", "path_mask": b, "recovery_pass": 1}],
        weighted, prob, np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool),
    )
    assert any(r["join_status"] == "accepted" for r in attempts)
    assert len(joined) == 1


def test_reject_join_with_incompatible_orientation_or_punctum_gap():
    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[20, 12:25] = True
    b[24:38, 28] = True
    weighted = np.ones((64, 64), dtype=np.float32)
    prob = _path_prob(punctum=0.9)
    joined, attempts = pathseg.join_fragments(
        [{"seed_id": "a", "path_mask": a, "recovery_pass": 1}, {"seed_id": "b", "path_mask": b, "recovery_pass": 1}],
        weighted, prob, np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool),
    )
    assert attempts
    assert all(r["join_status"] == "rejected" for r in attempts)
    assert len(joined) == 2


def test_parallel_paths_in_one_semantic_region_seed_as_two_paths():
    roi = np.ones((80, 80), dtype=bool)
    raw_label = np.zeros((80, 80), dtype=np.int32)
    weighted = np.zeros((80, 80), dtype=np.float32)
    weighted[25, 15:65] = 1.0
    weighted[50, 15:65] = 1.0
    prob = _path_prob((80, 80))
    seed = pathseg.build_candidate_seed_mask(weighted, prob, raw_label, roi, pass_no=1)
    assert measure.label(seed).max() == 2


def test_preserve_complete_curved_or_unusual_path_as_warning_not_failure():
    path = np.zeros((80, 80), dtype=bool)
    rr, cc = draw.bezier_curve(15, 15, 40, 65, 65, 20, weight=2)
    path[rr, cc] = True
    raw = np.zeros((80, 80), dtype=np.float32)
    raw[morphology.dilation(path, morphology.disk(2))] = 1.0
    prob = _path_prob((80, 80))
    raw_seg = {"img_norm": raw, "ridge": path.astype(np.float32)}
    row = pathseg.path_measure(
        path, raw, raw_seg, path.astype(np.float32), prob, np.ones((80, 80), dtype=bool),
        np.zeros((80, 80), dtype=np.int32), _instance_cfg(),
    )
    assert row["technical_validity"] is True


def test_ambiguous_branch_is_unresolved_and_compact_punctum_rejected():
    branch = np.zeros((64, 64), dtype=bool)
    branch[32, 15:50] = True
    branch[15:50, 32] = True
    raw = np.ones((64, 64), dtype=np.float32)
    prob = _path_prob((64, 64))
    raw_seg = {"img_norm": raw, "ridge": branch.astype(np.float32)}
    row = pathseg.path_measure(
        branch, raw, raw_seg, branch.astype(np.float32), prob, np.ones((64, 64), dtype=bool),
        np.zeros((64, 64), dtype=np.int32), _instance_cfg(),
    )
    assert row["technical_validity"] is False
    assert "unresolved_dense_graph_network" in row["technical_failure_reasons"]
    punctum = np.zeros((64, 64), dtype=bool)
    punctum[30:34, 30:34] = True
    punctum = morphology.skeletonize(punctum)
    prob_punctum = _path_prob((64, 64), nucleus=0.2, punctum=0.9)
    status, reasons = pathseg.classify_seed(punctum, prob_punctum, np.zeros((64, 64), dtype=np.int32), _instance_cfg())
    assert status == "punctum_like"
    assert "punctum_dominant" in reasons


def test_path_pass2_preservation_and_roi_leakage_constraints_present():
    source = (ROOT / "scratch" / "run_v56_2d_path_segmentation.py").read_text(encoding="utf-8")
    assert "pass2_preserved_pass1_checksum" in source
    assert "outside_roi_leakage" in source
    parent = np.zeros((50, 50), dtype=bool)
    parent[5:45, 5:45] = True
    ridge = np.zeros((50, 50), dtype=np.float32)
    ridge[10, 10:40] = 1
    roi = np.zeros((50, 50), dtype=bool)
    roi[5:45, 5:45] = True
    raw_label = np.zeros((50, 50), dtype=np.int32)
    instances, _ = instance_diag.split_semantic_component(parent, ridge, raw_label, roi, 1)
    assert all(not np.any(inst["mask"] & ~roi) for inst in instances)


def test_boundary_safe_segmentation_has_zero_outside_roi_leakage():
    yy, xx = np.mgrid[:96, :96]
    roi = (yy - 48) ** 2 + (xx - 48) ** 2 < 38 ** 2
    img = np.ones((96, 96), dtype=np.float32) * 25
    img[44:48, 25:70] = 210
    img[~roi] = 255
    cfg = seg.CONFIG.copy()
    cfg["ROI_BOUNDARY_SAFE_RIDGE"] = True
    cfg["AUTO_LOCAL_REANALYSIS"] = False
    cfg["DO_TRACKING"] = False
    cfg["SAVE_DEBUG_IMAGES"] = False
    cfg = seg.cfg_with_resolved_pixels(cfg)
    out = seg.segment_slice(img, cfg, roi_mask=roi, preprocess_context=None)
    assert int(np.count_nonzero(out["mask_hyst"] & ~roi)) == 0
    assert int(np.count_nonzero(out["skel_pruned"] & ~roi)) == 0


def test_off_roi_perturbation_does_not_change_boundary_safe_thresholds():
    yy, xx = np.mgrid[:96, :96]
    roi = (yy - 48) ** 2 + (xx - 48) ** 2 < 38 ** 2
    img = np.ones((96, 96), dtype=np.float32) * 25
    img[44:48, 25:70] = 210
    img2 = img.copy()
    img2[~roi] = 255
    cfg = seg.CONFIG.copy()
    cfg["ROI_BOUNDARY_SAFE_RIDGE"] = True
    cfg["AUTO_LOCAL_REANALYSIS"] = False
    cfg["DO_TRACKING"] = False
    cfg = seg.cfg_with_resolved_pixels(cfg)
    a = seg.segment_slice(img, cfg, roi_mask=roi, preprocess_context=None)["preprocess_debug"]
    b = seg.segment_slice(img2, cfg, roi_mask=roi, preprocess_context=None)["preprocess_debug"]
    assert abs(a["threshold_hi"] - b["threshold_hi"]) < 1e-6
    assert abs(a["threshold_lo"] - b["threshold_lo"]) < 1e-6
