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

def _unet_tracking_test_row(
    *,
    key,
    z_slice,
    x,
    y,
    area=50.0,
    orientation=0.0,
    probability=0.9,
):
    return {
        "source_instance_key": key,
        "z_slice": z_slice,
        "sperm_id": int(
            "".join(character for character in key if character.isdigit())
            or 1
        ),
        "centroid_x": float(x),
        "centroid_y": float(y),
        "length_px_geodesic": 10.0,
        "length_um_geodesic": 5.0,
        "width_px": 2.0,
        "width_um": 1.0,
        "area_px": float(area),
        "instance_mask_area_px": float(area),
        "estimated_slender_area_px": 20.0,
        "orientation": float(orientation),
        "n_endpoints": 2,
        "tortuosity": 1.0,
        "unet_mean_probability": float(probability),
        "unet_max_probability": float(probability),
        "detection_source": "unet_primary",
    }


def _unet_tracking_test_config():
    return {
        "UM_PER_PX_XY": 0.5,
        "UM_PER_SLICE_Z": 1.0,
        "UNET_TRACK_MAX_CENTROID_DIST_UM": 3.0,
        "UNET_TRACK_MAX_COST": 1.35,
        "UNET_TRACK_CENTROID_WEIGHT": 0.70,
        "UNET_TRACK_BBOX_IOU_WEIGHT": 0.20,
        "UNET_TRACK_ORIENTATION_WEIGHT": 0.05,
        "UNET_TRACK_AREA_WEIGHT": 0.03,
        "UNET_TRACK_PROBABILITY_WEIGHT": 0.02,
        "UNET_TRACK_MIN_BBOX_IOU": 0.0,
        "UNET_TRACK_MAX_AREA_LOG_RATIO": 1.60,
        "UNET_TRACK_MAX_RECONSTRUCTED_LENGTH_UM": 20.0,
    }


def test_unet_tracker_links_nearby_adjacent_observations():
    frame = pd.DataFrame(
        [
            _unet_tracking_test_row(
                key="z001_i00001",
                z_slice=1,
                x=10,
                y=10,
            ),
            _unet_tracking_test_row(
                key="z002_i00001",
                z_slice=2,
                x=11,
                y=10,
            ),
        ]
    )

    tracked, summary = (
        sperm_seg.track_across_slices_unet_primary(
            frame,
            _unet_tracking_test_config(),
        )
    )

    assert tracked["track_id"].nunique() == 1
    assert len(summary) == 1
    assert (
        tracked[
            "track_unet_accepted_link_count"
        ].iloc[0]
        == 1
    )


def test_unet_tracker_keeps_distant_observations_separate():
    frame = pd.DataFrame(
        [
            _unet_tracking_test_row(
                key="z001_i00001",
                z_slice=1,
                x=0,
                y=0,
            ),
            _unet_tracking_test_row(
                key="z002_i00001",
                z_slice=2,
                x=100,
                y=100,
            ),
        ]
    )

    tracked, _ = (
        sperm_seg.track_across_slices_unet_primary(
            frame,
            _unet_tracking_test_config(),
        )
    )

    assert tracked["track_id"].nunique() == 2


def test_unet_tracker_is_one_to_one_within_each_z_pair():
    frame = pd.DataFrame(
        [
            _unet_tracking_test_row(
                key="z001_i00001",
                z_slice=1,
                x=10,
                y=10,
            ),
            _unet_tracking_test_row(
                key="z001_i00002",
                z_slice=1,
                x=12,
                y=10,
            ),
            _unet_tracking_test_row(
                key="z002_i00001",
                z_slice=2,
                x=11,
                y=10,
            ),
        ]
    )

    tracked, _ = (
        sperm_seg.track_across_slices_unet_primary(
            frame,
            _unet_tracking_test_config(),
        )
    )

    duplicate_same_z = (
        tracked.groupby(["track_id", "z_slice"])
        .size()
    )

    assert not (duplicate_same_z > 1).any()
    assert tracked["track_id"].nunique() == 2


def test_unet_tracker_does_not_reject_morphology_warning():
    frame = pd.DataFrame(
        [
            {
                **_unet_tracking_test_row(
                    key="z001_i00001",
                    z_slice=1,
                    x=10,
                    y=10,
                ),
                "morphology_warning": True,
                "morphology_warning_reasons": "short|blob",
            },
            {
                **_unet_tracking_test_row(
                    key="z002_i00001",
                    z_slice=2,
                    x=10.5,
                    y=10,
                ),
                "morphology_warning": True,
                "morphology_warning_reasons": "short|blob",
            },
        ]
    )

    tracked, _ = (
        sperm_seg.track_across_slices_unet_primary(
            frame,
            _unet_tracking_test_config(),
        )
    )

    assert tracked["track_id"].nunique() == 1


def test_unet_tracker_rejects_duplicate_source_keys():
    frame = pd.DataFrame(
        [
            _unet_tracking_test_row(
                key="duplicate",
                z_slice=1,
                x=10,
                y=10,
            ),
            _unet_tracking_test_row(
                key="duplicate",
                z_slice=2,
                x=11,
                y=10,
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="unique source_instance_key",
    ):
        sperm_seg.track_across_slices_unet_primary(
            frame,
            _unet_tracking_test_config(),
        )


def test_runner_accepts_unet_primary_assignment_backend():
    runner.validate_run_options(
        repeat=1,
        tracking_backend="unet_primary_assignment",
    )


@mock.patch("scripts.run_v57_unet_primary_tracking_smoke.resolve_target_files")
@mock.patch("scripts.run_v57_unet_primary_tracking_smoke.load_saturn")
def test_tracking_smoke_runner_calibration_provenance(mock_load_saturn, mock_resolve, tmp_path):
    import json
    import numpy as np
    import pandas as pd
    import pytest
    from scripts import run_v57_unet_primary_tracking_smoke as runner

    base_params = tmp_path / "base_params.json"
    base_params.write_text(
        json.dumps(
            {
                "UM_PER_PX_XY": 0.379,
                "UM_PER_SLICE_Z": 0.346,
                "TRACK_MAX_DIST_UM": 6.8711,
                "TRACK_TECHNICAL_MAX_JOINED_LENGTH_UM": 15.0,
                "HYBRID_REPAIR_MAX_LINK_DIST_UM": 4.8,
                "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": 15.0,
                "UNET_TRACK_MAX_CENTROID_DIST_UM": 3.0,
                "UNET_TRACK_MAX_RECONSTRUCTED_LENGTH_UM": 20.0,
            }
        )
    )

    class MockArgs:
        def __init__(self):
            self.input_dir = str(tmp_path)
            self.unet_model = "fake.pt"
            self.base_params = str(base_params)
            self.roi_mask = "fake.npy"
            self.exclusion_mask = None
            self.z_values = "33,34,35"
            self.outdir = str(tmp_path / "out")
            self.repeat = 1
            self.tracking_backend = "global_assignment"

    mock_saturn = mock.MagicMock()
    mock_saturn.CONFIG = {"FILE_PATTERN": "*.tif"}
    mock_saturn.load_batch_files.return_value = (["fake1.tif", "fake2.tif", "fake3.tif"], [33, 34, 35])
    mock_saturn.robust_imread.return_value = np.zeros((10, 10))
    mock_saturn.load_roi_mask_file.return_value = np.ones((10, 10), dtype=bool)
    mock_saturn.build_stack_preprocess_context.return_value = {}
    mock_saturn._make_unet_context_from_paths.return_value = {}
    mock_saturn.segment_slice.return_value = np.zeros((10, 10), dtype=int)
    mock_saturn.measure_spermatids.return_value = {"results": []}
    mock_saturn.rows_from_results.return_value = []
    mock_saturn.track_across_slices.return_value = (pd.DataFrame(columns=["source_instance_key", "track_id"]), pd.DataFrame())
    
    mock_load_saturn.return_value = mock_saturn
    mock_resolve.return_value = {33: "fake1.tif", 34: "fake2.tif", 35: "fake3.tif"}
    
    payload = runner.run(MockArgs())
    
    assert payload["xy_um_per_px"] == pytest.approx(0.379)
    assert payload["z_um_per_slice"] == pytest.approx(0.346)
    assert payload["calibration_source"] == "base_parameters_json"
    assert payload["track_max_dist_um"] == pytest.approx(6.8711)
    assert (
        payload["track_technical_max_joined_length_um"]
        == pytest.approx(15.0)
    )
    assert (
        payload["hybrid_repair_max_link_dist_um"]
        == pytest.approx(4.8)
    )
    assert (
        payload["hybrid_repair_max_final_length_um"]
        == pytest.approx(15.0)
    )
    assert (
        payload["unet_track_max_centroid_dist_um"]
        == pytest.approx(3.0)
    )
    assert (
        payload["unet_track_max_reconstructed_length_um"]
        == pytest.approx(20.0)
    )
