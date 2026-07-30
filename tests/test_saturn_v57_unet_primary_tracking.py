import pytest
import numpy as np
import pandas as pd

import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sperm_segmentation_saturnv5_7", os.path.join(PROJECT_ROOT, "sperm_segmentation_saturnv5.7.py"))
sperm_seg = importlib.util.module_from_spec(spec)
sys.modules["sperm_segmentation_saturnv5_7"] = sperm_seg
spec.loader.exec_module(sperm_seg)
rows_from_results = sperm_seg.rows_from_results

def test_unet_primary_area_logic():
    um = 0.5
    z_idx = 10

    # Mock result from classical engine
    classical_res = {
        "label": 1,
        "length_px_geodesic": 20.0,
        "length_px_count": 22.0,
        "width_px": 2.0,
        "length_width_ratio": 10.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 50.0,
        "centroid_y": 50.0,
        "area_px": 40.0, # geodesic * width
        "detection_source": "saturn_classical",
    }

    # Mock result from unet_primary
    unet_res = {
        "label": 2,
        "source_instance_key": "10_2",
        "length_px_geodesic": 20.0,
        "length_px_count": 22.0,
        "width_px": 2.0,
        "length_width_ratio": 10.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 100.0,
        "centroid_y": 100.0,
        "area_px": 40.0, # classical slender area
        "instance_mask_area_px": 55.0, # filled mask area
        "detection_source": "unet_primary",
        "unet_mean_probability": 0.90,
        "unet_max_probability": 0.98,
        "morphology_warning": True,
        "morphology_warning_reasons": "short",
        "technical_failure": False,
        "technical_failure_reason": ""
    }

    rows = rows_from_results([classical_res, unet_res], z_idx, um)
    df = pd.DataFrame(rows)

    # Assert classical area uses estimated_slender_area_px
    classical_row = df[df["sperm_id"] == 1].iloc[0]
    assert classical_row["area_px"] == 40.0
    assert classical_row["estimated_slender_area_px"] == 40.0

    # Assert unet_primary area uses instance_mask_area_px
    unet_row = df[df["sperm_id"] == 2].iloc[0]
    assert unet_row["area_px"] == 55.0
    assert unet_row["estimated_slender_area_px"] == 40.0
    assert unet_row["instance_mask_area_px"] == 55.0

    # Assert other unet_primary fields are preserved
    assert unet_row["source_instance_key"] == "10_2"
    assert unet_row["unet_mean_probability"] == 0.90
    assert unet_row["morphology_warning"] == True
    assert unet_row["detection_source"] == "unet_primary"


from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runner_spec = importlib.util.spec_from_file_location("run_v57", os.path.join(PROJECT_ROOT, "scripts", "run_v57_unet_primary_tracking_smoke.py"))
runner = importlib.util.module_from_spec(runner_spec)
sys.modules["run_v57"] = runner
runner_spec.loader.exec_module(runner)

import pytest
import numpy as np
import pandas as pd

import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sperm_segmentation_saturnv5_7", os.path.join(PROJECT_ROOT, "sperm_segmentation_saturnv5.7.py"))
sperm_seg = importlib.util.module_from_spec(spec)
sys.modules["sperm_segmentation_saturnv5_7"] = sperm_seg
spec.loader.exec_module(sperm_seg)
rows_from_results = sperm_seg.rows_from_results

def test_unet_primary_area_logic():
    um = 0.5
    z_idx = 10

    # Mock result from classical engine
    classical_res = {
        "label": 1,
        "length_px_geodesic": 20.0,
        "length_px_count": 22.0,
        "width_px": 2.0,
        "length_width_ratio": 10.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 50.0,
        "centroid_y": 50.0,
        "area_px": 40.0, # geodesic * width
        "detection_source": "saturn_classical",
    }

    # Mock result from unet_primary
    unet_res = {
        "label": 2,
        "source_instance_key": "10_2",
        "length_px_geodesic": 20.0,
        "length_px_count": 22.0,
        "width_px": 2.0,
        "length_width_ratio": 10.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 100.0,
        "centroid_y": 100.0,
        "area_px": 40.0, # classical slender area
        "instance_mask_area_px": 55.0, # filled mask area
        "detection_source": "unet_primary",
        "unet_mean_probability": 0.90,
        "unet_max_probability": 0.98,
        "morphology_warning": True,
        "morphology_warning_reasons": "short",
        "technical_failure": False,
        "technical_failure_reason": ""
    }

    rows = rows_from_results([classical_res, unet_res], z_idx, um)
    df = pd.DataFrame(rows)

    # Assert classical area uses estimated_slender_area_px
    classical_row = df[df["sperm_id"] == 1].iloc[0]
    assert classical_row["area_px"] == 40.0
    assert classical_row["estimated_slender_area_px"] == 40.0

    # Assert unet_primary area uses instance_mask_area_px
    unet_row = df[df["sperm_id"] == 2].iloc[0]
    assert unet_row["area_px"] == 55.0
    assert unet_row["estimated_slender_area_px"] == 40.0
    assert unet_row["instance_mask_area_px"] == 55.0

    # Assert other unet_primary fields are preserved
    assert unet_row["source_instance_key"] == "10_2"
    assert unet_row["unet_mean_probability"] == 0.90
    assert unet_row["morphology_warning"] == True
    assert unet_row["detection_source"] == "unet_primary"


from unittest import mock

runner_spec = importlib.util.spec_from_file_location("run_v57", os.path.join(PROJECT_ROOT, "scripts", "run_v57_unet_primary_tracking_smoke.py"))
runner = importlib.util.module_from_spec(runner_spec)
sys.modules["run_v57"] = runner
runner_spec.loader.exec_module(runner)

@mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy")
def test_hybrid_repair_same_z_rejection(mock_legacy):
    cfg = {
        "HYBRID_REPAIR_MAX_GAP_SLICES": 1,
        "HYBRID_REPAIR_MAX_FRAGMENT_SLICES": 2,
        "HYBRID_REPAIR_MAX_COST": 100.0,
        "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": 100.0,
        "UM_PER_PX_XY": 1.0,
        "UM_PER_SLICE_Z": 1.0,
        "TRACK_MAX_DIST_UM": 10.0,
    }

    df_data = [
        {"track_id": 1, "z_slice": 1, "sperm_id": 1, "centroid_x": 1.0, "centroid_y": 1.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 2, "z_slice": 2, "sperm_id": 2, "centroid_x": 2.0, "centroid_y": 2.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 3, "z_slice": 3, "sperm_id": 3, "centroid_x": 3.0, "centroid_y": 3.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 4, "z_slice": 1, "sperm_id": 4, "centroid_x": 4.0, "centroid_y": 4.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
    ]
    df = pd.DataFrame(df_data)
    df["width_px"] = df["width_px"].astype(float)
    df["length_px_geodesic"] = df["length_px_geodesic"].astype(float)
    ts_data = [
        {"track_id": 1, "n_slices": 1, "rejected_extension_reasons": ""},
        {"track_id": 2, "n_slices": 1, "rejected_extension_reasons": ""},
        {"track_id": 3, "n_slices": 1, "rejected_extension_reasons": ""},
        {"track_id": 4, "n_slices": 1, "rejected_extension_reasons": ""},
    ]
    ts = pd.DataFrame(ts_data)

    def mock_cost(src, dst, cfg):
        # We mapped track_id to centroid_x exactly
        src_id = src.get("x")
        dst_id = dst.get("x")
        if src_id == 1.0 and dst_id == 2.0:
            return 1.0, 1.0, 1.0
        if src_id == 4.0 and dst_id == 2.0:
            return 2.0, 1.0, 1.0
        if src_id == 2.0 and dst_id == 3.0:
            return 3.0, 1.0, 1.0
        return float('inf'), 0, 0

    mock_legacy.return_value = (df, ts)

    with mock.patch("sperm_segmentation_saturnv5_7._hybrid_repair_cost", side_effect=mock_cost):
        final_df, final_ts = sperm_seg.track_across_slices_hybrid_repair(pd.DataFrame(), cfg)

    tid_1 = final_df[final_df["sperm_id"] == 1].iloc[0]["track_id"]
    tid_2 = final_df[final_df["sperm_id"] == 2].iloc[0]["track_id"]
    tid_3 = final_df[final_df["sperm_id"] == 3].iloc[0]["track_id"]
    tid_4 = final_df[final_df["sperm_id"] == 4].iloc[0]["track_id"]

    assert tid_1 == tid_2 == tid_3, "Valid disjoint-Z repairs should succeed"
    assert tid_4 != tid_2, "Overlapping Z membership (via complete union group) must be rejected"

    assert final_ts["track_hybrid_repair_attempt_count"].iloc[0] == 3
    assert final_ts["track_hybrid_repair_accepted_count"].iloc[0] == 2
    assert final_ts["track_hybrid_repair_rejected_same_z_count"].iloc[0] == 1


def test_hybrid_repair_runtime_error_on_duplicate_z():
    cfg = {
        "UM_PER_PX_XY": 1.0,
        "UM_PER_SLICE_Z": 1.0,
    }
    df_data = [
        {"track_id": 1, "z_slice": 1, "sperm_id": 1, "centroid_x": 0.0, "centroid_y": 0.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 1, "z_slice": 1, "sperm_id": 2, "centroid_x": 0.0, "centroid_y": 0.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
        {"track_id": 2, "z_slice": 2, "sperm_id": 3, "centroid_x": 0.0, "centroid_y": 0.0, "length_px_geodesic": 1.0, "width_px": 1.0, "n_endpoints": 2, "length_um_geodesic": 1.0, "width_um": 1.0, "area_px": 1.0, "tortuosity": 1.0},
    ]
    df = pd.DataFrame(df_data)
    df["width_px"] = df["width_px"].astype(float)
    df["length_px_geodesic"] = df["length_px_geodesic"].astype(float)
    ts = pd.DataFrame([
        {"track_id": 1, "n_slices": 1, "rejected_extension_reasons": ""},
        {"track_id": 2, "n_slices": 1, "rejected_extension_reasons": ""}
    ])

    with mock.patch("sperm_segmentation_saturnv5_7.track_across_slices_legacy", return_value=(df, ts)):
        with pytest.raises(RuntimeError, match="duplicate observations at the same z_slice"):
            sperm_seg.track_across_slices_hybrid_repair(pd.DataFrame(), cfg)

def test_validate_target_values():
    with pytest.raises(ValueError, match="duplicate Z values"):
        runner.validate_target_values([33, 34, 34, 35, 36])

def test_runner_args_validation():
    class Args:
        repeat = 0
        tracking_backend = "legacy"
    with pytest.raises(ValueError, match="--repeat must be >= 1"):
        runner.run(Args())

    class Args2:
        repeat = 1
        tracking_backend = "invalid_backend"
    with pytest.raises(ValueError, match="Invalid tracking backend"):
        runner.run(Args2())

def test_membership_hash():
    df1 = pd.DataFrame([
        {"track_id": 2, "source_instance_key": "z001_i00001"},
        {"track_id": 2, "source_instance_key": "z002_i00001"},
        {"track_id": 1, "source_instance_key": "z003_i00001"},
    ])
    df2 = pd.DataFrame([
        {"track_id": 1, "source_instance_key": "z001_i00001"},
        {"track_id": 1, "source_instance_key": "z002_i00001"},
        {"track_id": 2, "source_instance_key": "z003_i00001"},
    ])
    df3 = pd.DataFrame([
        {"track_id": 1, "source_instance_key": "z001_i00001"},
        {"track_id": 2, "source_instance_key": "z002_i00001"},
        {"track_id": 3, "source_instance_key": "z003_i00001"},
    ])
    hash1 = runner.compute_membership_hash(df1)
    hash2 = runner.compute_membership_hash(df2)
    hash3 = runner.compute_membership_hash(df3)

    assert hash1 == hash2, "Membership hash must be independent of track_id ordering"
    assert hash1 != hash3, "Membership hash must change when memberships change"

from pathlib import Path
import tempfile

def test_runner_integrity_failures_and_counts():
    class MockSaturn:
        CONFIG = {"FILE_PATTERN": "*.tif"}
        def load_batch_files(self, a, b):
            return (["f1", "f2", "f3", "f4", "f5"], [33, 34, 35, 36, 37])
        def robust_imread(self, p): return np.zeros((10,10))
        def load_roi_mask_file(self, *a, **kw): return np.ones((10,10), dtype=bool)
        def build_stack_preprocess_context(self, *a, **kw): return None
        def _make_unet_context_from_paths(self, *a, **kw): return None
        def segment_slice(self, *a, **kw): return None
        def measure_spermatids(self, *a, **kw):
            return {"results": [{"label": 1}, {"label": 2}]}
        def rows_from_results(self, res, z, um):
            return [
                {"z_slice": z, "centroid_x": 0, "centroid_y": 0, "source_instance_key": f"z{z:03d}_i00001"},
                {"z_slice": z, "centroid_x": 1, "centroid_y": 1, "source_instance_key": f"z{z:03d}_i00002"}
            ]
        def track_across_slices(self, df, cfg):
            df_out = df.copy()
            df_out["track_id"] = 1 # Force all into one track to test same-Z detection
            return df_out, pd.DataFrame()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with mock.patch.object(runner, "load_saturn") as mock_load_saturn:
            mock_load_saturn.return_value = MockSaturn()

            class Args:
                input_dir = "in"
                unet_model = "model"
                base_params = tmp_path / "params.json"
                roi_mask = "roi"
                exclusion_mask = ""
                z_values = "33,34,35,36,37"
                outdir = str(tmp_path / "out")
                repeat = 1
                tracking_backend = "legacy"

            Args.base_params.write_text('{"UM_PER_PX_XY": 1.0}')
            payload = runner.run(Args())

            # 13. Same-Z final memberships fail the runner quality gate.
            # 14. Track-span counts sum to total tracks.
            # 15. Boundary plus interior single-slice counts equal all one-slice tracks.
            assert payload["quality_gates_passed"] is False
            assert payload["duplicate_same_z_count"] > 0

            # Check that counts sum properly
            total_spans = sum(payload[f"tracks_with_{i}_slice{'s' if i > 1 else ''}"] for i in range(1, 6))
            assert total_spans == payload["total_tracks"]

            assert payload["boundary_single_slice_tracks"] + payload["interior_single_slice_tracks"] == payload["tracks_with_1_slice"]

            # 11, 12 missing/duplicate instances
            # We can test these by modifying track_across_slices slightly in a second test
            class MockSaturnDrop(MockSaturn):
                def track_across_slices(self, df, cfg):
                    df_out = df.copy().iloc[:-1] # Drop one row
                    df_out = pd.concat([df_out, df_out.iloc[-1:]], ignore_index=True) # Duplicate one row
                    df_out["track_id"] = range(len(df_out))
                    return df_out, pd.DataFrame()

            mock_load_saturn.return_value = MockSaturnDrop()
            payload2 = runner.run(Args())
            assert payload2["dropped_instance_count"] == 1
            assert payload2["quality_gates_passed"] is False
