import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from skimage import draw, measure, morphology


ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graph = import_file("saturn_v56_2d_graph_test", ROOT / "utils" / "saturn_v56_2d_graph_segmentation.py")


def prob(shape=(64, 64), nucleus=0.9, tissue=0.05, punctum=0.03, diffuse=0.02):
    arr = np.zeros((*shape, 4), dtype=np.float32)
    arr[:, :, 0] = nucleus
    arr[:, :, 1] = tissue
    arr[:, :, 2] = punctum
    arr[:, :, 3] = diffuse
    return arr


def cfg():
    return {
        "MIN_SKEL_LEN_PX": 10,
        "MAX_GEODESIC_LEN_PX": 30,
        "MAX_WIDTH_PX": 4,
        "MIN_LENGTH_WIDTH_RATIO": 2.0,
        "MAX_TORTUOSITY": 2.5,
    }


def test_probability_map_channel_policy_is_explicit():
    assert graph.CLASS_ORDER == [
        "sperm_nucleus",
        "structured_tissue_edge",
        "punctum_or_ring",
        "diffuse_background",
    ]
    source = (ROOT / "utils" / "saturn_v56_2d_graph_segmentation.py").read_text(encoding="utf-8")
    assert "expected_class_order=CLASS_ORDER" in source
    assert "nucleus_channel=nucleus_channel" in source


def test_class_margin_weighted_ridge_calculation():
    ridge = np.ones((8, 8), dtype=np.float32)
    roi = np.ones((8, 8), dtype=bool)
    p = prob((8, 8), nucleus=0.8, tissue=0.2, punctum=0.1)
    forms = graph.weighted_ridge_formulations(ridge, p, roi)
    assert "C_raw_nucleus_not_tissue_or_punctum" in forms
    c = forms["C_raw_nucleus_not_tissue_or_punctum"]
    assert float(np.max(c)) <= 1.0
    assert float(np.min(c[roi])) >= 0.0
    assert np.all(c[~roi] == 0)


def test_punctum_and_tissue_seed_rejection():
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:16, 12:16] = True
    mask = morphology.skeletonize(mask)
    status, reasons = graph.classify_seed(mask, prob((32, 32), nucleus=0.2, punctum=0.9), np.zeros((32, 32), dtype=np.int32), cfg())
    assert status == "punctum_like"
    assert "punctum_dominant" in reasons
    status, reasons = graph.classify_seed(mask, prob((32, 32), nucleus=0.2, tissue=0.9), np.zeros((32, 32), dtype=np.int32), cfg())
    assert status == "tissue_like"
    assert "tissue_dominant" in reasons


def test_broken_collinear_path_extension_and_gap_join():
    seed = np.zeros((64, 64), dtype=bool)
    seed[30, 10:18] = True
    weighted = np.zeros((64, 64), dtype=np.float32)
    weighted[30, 10:28] = 1.0
    completed, rows = graph.complete_fragment(seed, weighted, prob(), np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool))
    assert np.count_nonzero(completed) > np.count_nonzero(seed)
    assert any(r["extension_status"] == "extended" for r in rows)

    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[20, 10:22] = True
    b[20, 26:38] = True
    weighted[:] = 0
    weighted[20, 10:38] = 1.0
    joined, attempts = graph.join_fragments(
        [{"seed_id": "a", "path_mask": a, "recovery_pass": 1}, {"seed_id": "b", "path_mask": b, "recovery_pass": 1}],
        weighted, prob(), np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool),
    )
    assert any(a["join_status"] == "accepted" for a in attempts)
    assert len(joined) == 1


def test_bad_joins_are_rejected_and_endpoint_exclusivity_is_a_gate():
    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[20, 10:22] = True
    b[24:38, 26] = True
    weighted = np.ones((64, 64), dtype=np.float32)
    joined, attempts = graph.join_fragments(
        [{"seed_id": "a", "path_mask": a, "recovery_pass": 1}, {"seed_id": "b", "path_mask": b, "recovery_pass": 1}],
        weighted, prob(punctum=0.9), np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool),
    )
    assert attempts
    assert all(row["join_status"] == "rejected" for row in attempts)
    gate = graph.quality_gate_result([], [{"source_seed_a": "e1", "source_seed_b": "e2", "join_status": "accepted"}, {"source_seed_a": "e1", "source_seed_b": "e3", "join_status": "accepted"}], [5], False)
    assert "endpoint_conflicts" in gate["failures"]


def test_parallel_paths_and_curved_path_preservation():
    roi = np.ones((80, 80), dtype=bool)
    raw_label = np.zeros((80, 80), dtype=np.int32)
    weighted = np.zeros((80, 80), dtype=np.float32)
    weighted[25, 15:65] = 1.0
    weighted[50, 15:65] = 1.0
    seeds = graph.build_candidate_seed_mask(weighted, prob((80, 80)), raw_label, roi, pass_no=1)
    assert measure.label(seeds).max() == 2

    path = np.zeros((80, 80), dtype=bool)
    rr, cc = draw.bezier_curve(15, 15, 40, 65, 65, 20, weight=2)
    path[rr, cc] = True
    raw = np.zeros((80, 80), dtype=np.float32)
    raw[morphology.dilation(path, morphology.disk(2))] = 1.0
    row = graph.path_measure(path, raw, {"img_norm": raw, "ridge": path.astype(np.float32)}, path.astype(np.float32), prob((80, 80)), roi, raw_label, cfg())
    assert row["technical_validity"] is True


def test_ambiguous_branch_unresolved_and_roi_leakage_gate():
    branch = np.zeros((64, 64), dtype=bool)
    branch[32, 15:50] = True
    branch[15:50, 32] = True
    raw = np.ones((64, 64), dtype=np.float32)
    row = graph.path_measure(branch, raw, {"img_norm": raw, "ridge": branch.astype(np.float32)}, branch.astype(np.float32), prob(), np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=np.int32), cfg())
    assert row["technical_validity"] is False
    assert "unresolved_dense_graph_network" in row["technical_failure_reasons"]
    gate = graph.quality_gate_result([{"final_accepted_status": True, "final_pixel_count": 1, "completeness_status": "complete_path", "technical_failure_reasons": "none"}], [], [5], False)
    assert "accepted_paths_too_often_tiny_fragments" in gate["failures"]


def test_historical_mapping_and_no_targets():
    baseline = {5: {"results": [{"label": 1, "centroid_x": 10.0, "centroid_y": 20.0}]}}
    paths = [{"z_index": 5, "stable_id": "new1", "final_accepted_status": True, "centroid_x": 11.0, "centroid_y": 20.0}]
    rows = graph.historical_to_new_mapping(baseline, paths)
    assert rows[0]["historical_disposition"] == "replaced_by_completed_path"
    source = (ROOT / "utils" / "saturn_v56_2d_graph_segmentation.py").read_text(encoding="utf-8").lower()
    assert "target_count" not in source
    assert "target morphology" not in source
    assert "9-10" not in source
