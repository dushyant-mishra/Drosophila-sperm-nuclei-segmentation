import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


seg = import_file("saturnv56_pipeline_test", ROOT / "sperm_segmentation_saturnv5.6.py")
tuner = import_file("saturnv56_tuner_test", ROOT / "utils" / "tune_parameters_Saturnv5_6.py")


def base_cfg():
    cfg = seg.CONFIG.copy()
    cfg.update({
        "SAVE_DEBUG_IMAGES": False,
        "SHOW_PREVIEW_WINDOW": False,
        "CLAHE_MODE": "standard",
        "NORM_STACK_WEIGHT": 0.8,
        "THRESHOLD_HI": 88.0,
        "THRESHOLD_LO": 80.0,
        "MIN_OBJ_PX": 3,
        "MIN_SKEL_LEN_UM": 1.0,
        "MAX_GEODESIC_LEN_UM": 40.0,
        "MAX_WIDTH_UM": 10.0,
        "MIN_LENGTH_WIDTH_RATIO": 1.0,
        "MAX_TORTUOSITY": 5.0,
        "MAX_ENDPOINT_COUNT": 10,
        "MAX_BRANCH_NODES": 99,
        "MAX_BRIDGE_UM": 0.0,
        "DENOISE_SIGMA_UM": 0.0,
        "BG_SIGMA_UM": 4.0,
        "RIDGE_SIGMAS_UM": [0.6, 0.9],
    })
    return cfg


def synthetic_image(scale=1.0, offset=0.0, dtype=np.float32, off_roi=False):
    img = np.zeros((64, 64), dtype=np.float32) + 20
    for y in (22, 32, 42):
        img[y, 15:42] = 220
        img[y - 1:y + 2, 15:42] += 60
    if off_roi:
        img[50:53, 5:60] = 5000
    img = img * scale + offset
    if dtype == np.uint8:
        img = np.clip(img / max(img.max(), 1) * 255, 0, 255).astype(np.uint8)
    elif dtype == np.uint16:
        img = np.clip(img / max(img.max(), 1) * 65535, 0, 65535).astype(np.uint16)
    else:
        img = img.astype(dtype)
    return img


def roi_mask():
    m = np.zeros((64, 64), dtype=bool)
    m[10:50, 10:50] = True
    return m


def make_context(tmp_path, images, roi, cfg, exclusion=None):
    files = []
    for i, image in enumerate(images):
        p = tmp_path / f"z{i:02d}.tif"
        tifffile.imwrite(p, image)
        files.append(str(p))
    return seg.build_stack_preprocess_context(files, roi, cfg, exclusion_mask=exclusion), files


def run_seg(tmp_path, image, roi=None, exclusion=None, cfg=None):
    cfg = cfg or base_cfg()
    roi = roi if roi is not None else roi_mask()
    ctx, _ = make_context(tmp_path, [image, image * 0.9 + 5], roi, cfg, exclusion)
    return seg.segment_slice(image, cfg, roi_mask=roi, preprocess_context=ctx, exclusion_mask=exclusion)


def detection_summary(seg_out, cfg):
    meas = seg.measure_spermatids(seg_out, cfg)
    lengths = [r["length_px_geodesic"] for r in meas["results"]]
    return len(meas["results"]), float(np.median(lengths)) if lengths else 0.0


def test_off_roi_bright_object_invariance(tmp_path):
    cfg = base_cfg()
    a = run_seg(tmp_path, synthetic_image(), cfg=cfg)
    b = run_seg(tmp_path, synthetic_image(off_roi=True), cfg=cfg)
    ca, la = detection_summary(a, cfg)
    cb, lb = detection_summary(b, cfg)
    assert abs(ca - cb) <= 1
    assert abs(la - lb) <= 2.5


def test_roi_zero_leak(tmp_path):
    r = roi_mask()
    out = run_seg(tmp_path, synthetic_image(off_roi=True), roi=r)
    for key in ("mask_hyst", "mask_clean", "skel_clean", "skel_bridged", "skel_pruned", "skel_labeled"):
        assert np.count_nonzero(out[key] & ~r) == 0


def test_exclusion_mask_zero_leak_and_detection_rejection(tmp_path):
    r = roi_mask()
    exclusion = np.zeros((64, 64), dtype=bool)
    exclusion[28:36, 10:50] = True
    out = run_seg(tmp_path, synthetic_image(), roi=r, exclusion=exclusion)
    for key in ("mask_hyst", "mask_clean", "skel_clean", "skel_bridged", "skel_pruned", "skel_labeled"):
        assert np.count_nonzero(out[key] & exclusion) == 0


def test_roi_boundary_artifact_and_crop_mapping(tmp_path):
    r = roi_mask()
    out = run_seg(tmp_path, synthetic_image(), roi=r)
    boundary_band = np.zeros_like(r)
    boundary_band[9:11, 10:50] = True
    boundary_band[49:51, 10:50] = True
    assert out["ridge"].shape == r.shape
    assert np.count_nonzero(out["skel_pruned"] & boundary_band) <= 2


def test_multiplicative_and_additive_brightness_invariance(tmp_path):
    cfg = base_cfg()
    a = detection_summary(run_seg(tmp_path, synthetic_image(), cfg=cfg), cfg)
    b = detection_summary(run_seg(tmp_path, synthetic_image(scale=1.8), cfg=cfg), cfg)
    c = detection_summary(run_seg(tmp_path, synthetic_image(offset=120), cfg=cfg), cfg)
    assert abs(a[0] - b[0]) <= 1
    assert abs(a[0] - c[0]) <= 1


def test_bit_depth_invariance(tmp_path):
    cfg = base_cfg()
    a = detection_summary(run_seg(tmp_path, synthetic_image(dtype=np.uint8), cfg=cfg), cfg)
    b = detection_summary(run_seg(tmp_path, synthetic_image(dtype=np.uint16), cfg=cfg), cfg)
    assert abs(a[0] - b[0]) <= 1


def test_stack_wide_clahe_profile_once(tmp_path):
    cfg = base_cfg()
    cfg["CLAHE_MODE"] = "auto_stack"
    ctx, _ = make_context(tmp_path, [synthetic_image(scale=0.5), synthetic_image(scale=2.0)], roi_mask(), cfg)
    out1 = seg.segment_slice(synthetic_image(scale=0.5), cfg, roi_mask=roi_mask(), preprocess_context=ctx)
    out2 = seg.segment_slice(synthetic_image(scale=2.0), cfg, roi_mask=roi_mask(), preprocess_context=ctx)
    assert out1["preprocess_debug"]["profile"] == out2["preprocess_debug"]["profile"] == ctx.selected_clahe_profile


def test_physical_unit_conversion():
    cfg = base_cfg()
    cfg["UM_PER_PX_XY"] = 0.5
    cfg["MAX_BRIDGE_UM"] = 1.5
    cfg["MAX_WIDTH_UM"] = 4.0
    cfg["MIN_SKEL_LEN_UM"] = 6.0
    resolved = seg.resolve_pixel_parameters(cfg)
    assert resolved["pixels"]["MAX_BRIDGE_PX"] == 3
    assert resolved["pixels"]["MAX_WIDTH_PX"] == pytest.approx(8.0)
    assert resolved["pixels"]["MIN_SKEL_LEN_PX"] == pytest.approx(12.0)


def test_bridge_distance_orientation_and_roi():
    sk = np.zeros((20, 20), dtype=bool)
    sk[10, 2:5] = True
    sk[10, 15:18] = True
    lab = seg.measure.label(sk)
    out, stats = seg.bridge_skeleton_endpoints(sk, lab, 3, return_stats=True)
    assert stats["accepted"] == 0
    sk2 = np.zeros((20, 20), dtype=bool)
    sk2[10, 5:8] = True
    sk2[7:10, 10] = True
    lab2 = seg.measure.label(sk2)
    out2, stats2 = seg.bridge_skeleton_endpoints(sk2, lab2, 5, max_angle_deg=10, return_stats=True)
    assert stats2["accepted"] == 0
    valid = np.ones((20, 20), dtype=bool)
    valid[:, 8:10] = False
    sk3 = np.zeros((20, 20), dtype=bool)
    sk3[10, 5:8] = True
    sk3[10, 10:13] = True
    lab3 = seg.measure.label(sk3)
    out3, stats3 = seg.bridge_skeleton_endpoints(sk3, lab3, 5, valid_mask=valid, return_stats=True)
    assert np.count_nonzero(out3 & ~valid) == 0
    assert stats3["accepted"] == 0


def test_tuner_roi_passing_and_context_reuse(monkeypatch):
    calls = []
    ctx = object()
    roi = np.ones((8, 8), dtype=bool)
    excl = np.zeros((8, 8), dtype=bool)
    def fake_segment(img, cfg, **kwargs):
        calls.append(kwargs)
        z = np.zeros_like(img, dtype=bool)
        return {"mask_hyst": z, "mask_clean": z, "skel_clean": z, "skel_bridged": z,
                "skel_pruned": z, "skel_labeled": z.astype(np.int32), "dist_clean": z.astype(float)}
    monkeypatch.setattr(tuner.segmentation, "segment_slice", fake_segment)
    monkeypatch.setattr(tuner.segmentation, "measure_spermatids", lambda s, c: {"results": [], "skel_label": s["skel_labeled"]})
    tuner.images_to_eval = [np.zeros((8, 8)), np.zeros((8, 8))]
    tuner.z_values_eval = [0, 1]
    tuner.roi_mask_global = roi
    tuner.exclusion_mask_global = excl
    tuner.preprocess_context_global = ctx
    tuner.evaluate_segmentation_candidate({"THRESHOLD_HI": 90.0, "THRESHOLD_LO": 82.0})
    tuner.evaluate_segmentation_candidate({"THRESHOLD_HI": 91.0, "THRESHOLD_LO": 83.0})
    assert all(c["roi_mask"] is roi for c in calls)
    assert all(c["exclusion_mask"] is excl for c in calls)
    assert all(c["preprocess_context"] is ctx for c in calls)


def test_base_parameter_merge_auto_slices_and_seed(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"A": 1, "B": 1}', encoding="utf-8")
    b.write_text('{"B": 2}', encoding="utf-8")
    assert tuner.merge_base_params([str(a), str(b)]) == {"A": 1, "B": 2}
    slices = tuner.select_auto_slices(20, 6)
    assert slices == [0, 4, 8, 11, 15, 19]
    assert len(slices) == len(set(slices))
    c1 = tuner.sample_candidates(tuner.SEGMENTATION_PARAM_SPACE, 4, 7)
    c2 = tuner.sample_candidates(tuner.SEGMENTATION_PARAM_SPACE, 4, 7)
    assert c1 == c2


def test_version_isolation_hashes():
    expected = {
        ROOT / "sperm_segmentation_saturnv5.5.py": "bb687703d0a3ef004a36d9685417f7ceabb01da14debae1bf65231d7c973e529",
        ROOT / "utils" / "tune_parameters_Saturnv5_5.py": "2ac57472c5dc6185f831004897132f7983725ad2e245b6b99693550be2144344",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
