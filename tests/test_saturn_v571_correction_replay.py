import importlib.util
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from utils.saturn_v571_gui_services import CorrectionEvent


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    path = ROOT / "sperm_segmentation_saturnv5.7.1.py"
    spec = importlib.util.spec_from_file_location("saturn_v571_correction_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correction_event(
    source_id="z0001:instance:2",
    action="exclude_false_detection",
    z_index=1,
):
    return CorrectionEvent(
        correction_uuid=str(uuid4()),
        revision=1,
        specimen_id="specimen-01",
        z_index=z_index,
        action=action,
        source_instance_ids=(source_id,),
        technical_reason="weak_isolated_noise" if action == "exclude_false_detection" else "",
        reviewer="reviewer-01",
        timestamp_utc="2026-08-28T16:00:00Z",
        base_run_manifest_sha256="a" * 64,
        software_version="v5.7.1",
        analysis_profile_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        calibration_provenance_sha256="d" * 64,
        evidence_references=("raw_evidence/instance_labels/z0001/evidence.json",),
        before_hash="e" * 64,
        after_hash="f" * 64,
    )


def detections():
    return pd.DataFrame(
        {
            "source_instance_key": [
                "z0000:instance:1",
                "z0001:instance:2",
                "z0001:instance:3",
                "z0002:instance:4",
            ],
            "z_slice": [0, 1, 1, 2],
            "centroid_x": [10.0, 10.2, 30.0, 10.4],
            "centroid_y": [20.0, 20.1, 40.0, 20.2],
        }
    )


def test_false_detection_correction_retracks_complete_specimen_without_mutating_source():
    saturn = load_saturn()
    original = detections()
    snapshot = original.copy(deep=True)

    def tracker(frame, _cfg):
        assert list(frame["source_instance_key"]) == [
            "z0000:instance:1",
            "z0001:instance:3",
            "z0002:instance:4",
        ]
        tracked = frame.copy()
        tracked["track_id"] = [1, 2, 1]
        return tracked, pd.DataFrame({"track_id": [1, 2]})

    tracked, tracks, manifest = saturn.retrack_false_detection_corrections(
        original,
        (correction_event(),),
        {},
        specimen_id="specimen-01",
        tracking_callable=tracker,
    )

    pd.testing.assert_frame_equal(original, snapshot)
    assert "z0001:instance:2" not in set(tracked["source_instance_key"])
    assert set(tracked["correction_revision"]) == {1}
    assert set(tracks["correction_revision"]) == {1}
    assert tracks["technical_valid"].all()
    assert manifest["full_specimen_retracking"] is True
    assert manifest["excluded_detection_count"] == 1


def test_correction_retracking_rejects_wrong_z_and_same_z_track_collision():
    saturn = load_saturn()
    wrong_z = correction_event(z_index=2)
    with pytest.raises(ValueError, match="Z plane"):
        saturn.retrack_false_detection_corrections(
            detections(),
            (wrong_z,),
            {},
            specimen_id="specimen-01",
            tracking_callable=lambda frame, _cfg: (frame, pd.DataFrame()),
        )

    def invalid_tracker(frame, _cfg):
        tracked = frame.copy()
        tracked["track_id"] = [2, 2, 1]
        return tracked, pd.DataFrame({"track_id": [1, 2]})

    with pytest.raises(ValueError, match="multiple observations"):
        saturn.retrack_false_detection_corrections(
            detections(),
            (correction_event(source_id="z0000:instance:1", z_index=0),),
            {},
            specimen_id="specimen-01",
            tracking_callable=invalid_tracker,
        )


def test_correction_retracking_rejects_dropped_sources_and_nondeterminism():
    saturn = load_saturn()

    def dropping_tracker(frame, _cfg):
        tracked = frame.iloc[:1].copy()
        tracked["track_id"] = 1
        return tracked, pd.DataFrame({"track_id": [1]})

    with pytest.raises(ValueError, match="conserve every surviving"):
        saturn.retrack_false_detection_corrections(
            detections(),
            (correction_event(),),
            {},
            specimen_id="specimen-01",
            tracking_callable=dropping_tracker,
        )

    calls = 0

    def nondeterministic_tracker(frame, _cfg):
        nonlocal calls
        calls += 1
        tracked = frame.copy()
        tracked["track_id"] = [1, 2, 1] if calls == 1 else [1, 3, 1]
        return tracked, pd.DataFrame({"track_id": sorted(tracked["track_id"].unique())})

    with pytest.raises(ValueError, match="not deterministic"):
        saturn.retrack_false_detection_corrections(
            detections(),
            (correction_event(),),
            {},
            specimen_id="specimen-01",
            tracking_callable=nondeterministic_tracker,
        )


def test_correction_retracking_retains_short_and_long_morphology_warnings():
    saturn = load_saturn()

    def morphology_tracker(frame, _cfg):
        tracked = frame.copy()
        tracked["track_id"] = [1, 2, 3]
        summary = pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "centroid_x": [10.0, 30.0, 10.4],
                "centroid_y": [20.0, 40.0, 20.2],
                "projection_z_extent_um": [1.0, 18.0, 8.0],
                "length_width_ratio": [1.2, 4.0, 3.5],
                "tortuosity_3d": [1.1, 1.2, 1.0],
            }
        )
        return tracked, summary

    _tracked, tracks, _manifest = saturn.retrack_false_detection_corrections(
        detections(),
        (correction_event(),),
        {
            "ANALYSIS_MODE": "comparative",
            "MIN_SKEL_LEN_UM": 2.0,
            "AUDIT_MAX_LENGTH_UM": 15.0,
            "MIN_LENGTH_WIDTH_RATIO": 2.5,
        },
        specimen_id="specimen-01",
        tracking_callable=morphology_tracker,
    )

    assert tracks["technical_valid"].all()
    assert tracks.loc[tracks["track_id"].isin([1, 2]), "morphology_warning"].all()
