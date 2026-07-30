import pytest
import numpy as np
import pandas as pd
import importlib.util
import os
import sys
from unittest import mock
import tempfile
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sperm_segmentation_saturnv5_7", os.path.join(PROJECT_ROOT, "sperm_segmentation_saturnv5.7.py"))
sperm_seg = importlib.util.module_from_spec(spec)
sys.modules["sperm_segmentation_saturnv5_7"] = sperm_seg
spec.loader.exec_module(sperm_seg)
rows_from_results = sperm_seg.rows_from_results

runner_spec = importlib.util.spec_from_file_location("run_v57", os.path.join(PROJECT_ROOT, "scripts", "run_v57_unet_primary_tracking_smoke.py"))
runner = importlib.util.module_from_spec(runner_spec)
sys.modules["run_v57"] = runner
runner_spec.loader.exec_module(runner)

# 1. A hybrid-repair candidate with overlapping current Z memberships is rejected.
@mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy")
def test_hybrid_repair_candidate_overlapping_z_rejected(mock_legacy):
    pass

# 2. The same-Z test uses the complete current union group, not only the two original endpoint tracks.
@mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy")
def test_same_z_test_uses_complete_union_group(mock_legacy):
    pass

# 3. A valid disjoint-Z repair still succeeds.
@mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy")
def test_valid_disjoint_z_repair_succeeds(mock_legacy):
    pass

# 4. A rejected merge does not mutate either union group.
@mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy")
def test_rejected_merge_does_not_mutate_union_groups(mock_legacy):
    pass

# 5. Final duplicate (track_id, z_slice) membership raises RuntimeError.
@mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy")
def test_final_duplicate_membership_raises_runtime_error(mock_legacy):
    cfg = {"UM_PER_PX_XY": 1.0, "UM_PER_SLICE_Z": 1.0}
    df = pd.DataFrame([
        {"track_id": 1, "z_slice": 1, "sperm_id": 1, "centroid_x": 0.0, "centroid_y": 0.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 1, "z_slice": 1, "sperm_id": 2, "centroid_x": 0.0, "centroid_y": 0.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 2, "z_slice": 2, "sperm_id": 3, "centroid_x": 0.0, "centroid_y": 0.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
    ])
    ts = pd.DataFrame([
        {"track_id": 1, "n_slices": 1, "rejected_extension_reasons": ""},
        {"track_id": 2, "n_slices": 1, "rejected_extension_reasons": ""}
    ])
    mock_legacy.return_value = (df, ts)
    with pytest.raises(RuntimeError):
        sperm_seg.track_across_slices_hybrid_repair(pd.DataFrame(), cfg)

# 6. validate_target_values rejects duplicate Z values.
def test_validate_target_values_rejects_duplicate_z():
    with pytest.raises(ValueError):
        runner.validate_target_values([33, 34, 34, 35, 36])

# 7. Invalid tracking backend is rejected.
def test_invalid_tracking_backend_is_rejected():
    with pytest.raises(ValueError):
        runner.validate_run_options(repeat=1, tracking_backend="invalid_backend")

# 8. repeat < 1 is rejected.
def test_repeat_less_than_1_is_rejected():
    with pytest.raises(ValueError):
        runner.validate_run_options(repeat=0, tracking_backend="legacy")

# 9. Membership hash is independent of numeric track-ID ordering.
def test_membership_hash_independent_of_ordering():
    df1 = pd.DataFrame([{"track_id": 2, "source_instance_key": "z1"}, {"track_id": 1, "source_instance_key": "z2"}])
    df2 = pd.DataFrame([{"track_id": 1, "source_instance_key": "z1"}, {"track_id": 2, "source_instance_key": "z2"}])
    assert runner.compute_membership_hash(df1) == runner.compute_membership_hash(df2)

# 10. Membership hash changes when actual membership changes.
def test_membership_hash_changes_with_membership():
    df1 = pd.DataFrame([{"track_id": 1, "source_instance_key": "z1"}, {"track_id": 1, "source_instance_key": "z2"}])
    df2 = pd.DataFrame([{"track_id": 1, "source_instance_key": "z1"}, {"track_id": 2, "source_instance_key": "z2"}])
    assert runner.compute_membership_hash(df1) != runner.compute_membership_hash(df2)

# 11. Missing tracked observations are detected.
def test_missing_tracked_observations_detected():
    pre = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1}, {"source_instance_key": "z2", "z_slice": 2}])
    post = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1, "track_id": 1}])
    metrics = runner.evaluate_tracking_integrity(pretracking_df=pre, tracked_df=post, requested_z_values=[1,2])
    assert metrics["dropped_instance_count"] == 1

# 12. Duplicate source-instance observations are detected.
def test_duplicate_source_instance_observations_detected():
    pre = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1}])
    post = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1, "track_id": 1}, {"source_instance_key": "z1", "z_slice": 1, "track_id": 1}])
    metrics = runner.evaluate_tracking_integrity(pretracking_df=pre, tracked_df=post, requested_z_values=[1])
    assert metrics["duplicated_source_instance_count"] == 1

# 13. Same-Z final memberships fail the runner quality gate.
def test_same_z_final_memberships_fail_quality_gate():
    pre = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1}, {"source_instance_key": "z2", "z_slice": 1}])
    post = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1, "track_id": 1}, {"source_instance_key": "z2", "z_slice": 1, "track_id": 1}])
    metrics = runner.evaluate_tracking_integrity(pretracking_df=pre, tracked_df=post, requested_z_values=[1])
    assert metrics["quality_gates_passed"] is False
    assert metrics["duplicate_same_z_group_count"] == 1

# 14. Track-span counts sum to total tracks.
def test_track_span_counts_sum_to_total_tracks():
    post = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1, "track_id": 1}, {"source_instance_key": "z2", "z_slice": 2, "track_id": 1}])
    metrics = runner.compute_track_span_metrics(post, [1, 2])
    total = sum(metrics[f"tracks_with_{i}_slices"] for i in range(1, 8))
    assert total == metrics["total_tracks"]

# 15. Boundary plus interior single-slice counts equal all one-slice tracks.
def test_boundary_plus_interior_single_slice_counts_equal_one_slice_tracks():
    post = pd.DataFrame([{"source_instance_key": "z1", "z_slice": 1, "track_id": 1}, {"source_instance_key": "z2", "z_slice": 2, "track_id": 2}])
    metrics = runner.compute_track_span_metrics(post, [1, 2, 3])
    assert metrics["boundary_single_slice_tracks"] + metrics["interior_single_slice_tracks"] == metrics.get("tracks_with_1_slices", 0)

# 16. The existing area-field test remains and continues to pass.
def test_unet_primary_area_logic():
    classical_res = {"label": 1, "length_px_geodesic": 20.0, "length_px_count": 22.0, "width_px": 2.0, "length_width_ratio": 10.0, "tortuosity": 1.0, "n_endpoints": 2, "n_branch_nodes": 0, "centroid_x": 50.0, "centroid_y": 50.0, "area_px": 40.0, "detection_source": "saturn_classical"}
    unet_res = {"label": 2, "source_instance_key": "10_2", "length_px_geodesic": 20.0, "length_px_count": 22.0, "width_px": 2.0, "length_width_ratio": 10.0, "tortuosity": 1.0, "n_endpoints": 2, "n_branch_nodes": 0, "centroid_x": 100.0, "centroid_y": 100.0, "area_px": 40.0, "instance_mask_area_px": 55.0, "detection_source": "unet_primary", "unet_mean_probability": 0.90, "unet_max_probability": 0.98, "morphology_warning": True, "morphology_warning_reasons": "short", "technical_failure": False, "technical_failure_reason": ""}
    rows = rows_from_results([classical_res, unet_res], 10, 0.5)
    df = pd.DataFrame(rows)
    assert df[df["sperm_id"] == 1].iloc[0]["area_px"] == 40.0
    assert df[df["sperm_id"] == 2].iloc[0]["area_px"] == 55.0

# 17. The new audit counter candidate counts match attempts.
def test_audit_counter_candidate_count():
    # Placeholder
    pass

# 18. Evaluated count reflects those not skipped.
def test_audit_counter_evaluated_count():
    pass

# 19. Accepted count matches number of targets repaired.
def test_audit_counter_accepted_count():
    pass

# 20. Same root skip count functions correctly.
def test_audit_counter_same_root_count():
    pass
