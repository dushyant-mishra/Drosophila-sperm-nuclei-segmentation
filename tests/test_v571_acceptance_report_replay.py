import importlib.util
import zipfile

import pandas as pd
import pytest


def load_module():
    spec = importlib.util.spec_from_file_location(
        "acceptance_report_replay",
        "scripts/rebuild_v571_acceptance_report_from_replay.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replay_summary_binding_reconciles_counts(tmp_path):
    module = load_module()
    tracked = tmp_path / "tracked.csv"
    tracks = tmp_path / "tracks.csv"
    pd.DataFrame({"track_id": [1, 1, 2]}).to_csv(tracked, index=False)
    pd.DataFrame(
        {"track_id": [1, 2], "technical_valid": [True, False]}
    ).to_csv(tracks, index=False)
    summary = pd.DataFrame(
        {
            "specimen": ["S1"],
            "candidate": ["production_morphology_neutral"],
            "detections_2d": [3],
            "tracks": [2],
        }
    )
    result = module.assert_replay_summary_binding(summary, "S1", tracked, tracks)
    assert result == {
        "detections_2d": 3,
        "all_reconstructed_tracks": 2,
        "technical_valid_tracks": 1,
    }


def test_replay_summary_binding_rejects_mismatched_track_count(tmp_path):
    module = load_module()
    tracked = tmp_path / "tracked.csv"
    tracks = tmp_path / "tracks.csv"
    pd.DataFrame({"track_id": [1]}).to_csv(tracked, index=False)
    pd.DataFrame({"track_id": [1], "technical_valid": [True]}).to_csv(
        tracks, index=False
    )
    summary = pd.DataFrame(
        {
            "specimen": ["S1"],
            "candidate": ["production_morphology_neutral"],
            "detections_2d": [1],
            "tracks": [2],
        }
    )
    with pytest.raises(ValueError, match="Track replay count mismatch"):
        module.assert_replay_summary_binding(summary, "S1", tracked, tracks)


def test_retained_replay_archive_is_content_bound(tmp_path):
    module = load_module()
    payloads = {
        "S1/source_2d_detections.csv": b"detection_id\n1\n",
        "S1/tracked_detections.csv": b"track_id\n1\n",
        "S1/track_summary.csv": b"track_id\n1\n",
    }
    archive_path = tmp_path / "replay.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    manifest = {
        "artifacts": [
            {
                "specimen": "S1",
                "candidate": "production_morphology_neutral",
                "source_2d_detections_sha256": module.sha256_bytes(
                    payloads["S1/source_2d_detections.csv"]
                ),
                "tracked_csv_sha256": module.sha256_bytes(
                    payloads["S1/tracked_detections.csv"]
                ),
                "tracks_csv_sha256": module.sha256_bytes(
                    payloads["S1/track_summary.csv"]
                ),
            }
        ]
    }
    assert module.validate_retained_replay_archive(archive_path, manifest) == {
        name: module.sha256_bytes(payload) for name, payload in payloads.items()
    }


def test_retained_replay_archive_rejects_changed_content(tmp_path):
    module = load_module()
    archive_path = tmp_path / "replay.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("S1/source_2d_detections.csv", b"changed")
        archive.writestr("S1/tracked_detections.csv", b"tracked")
        archive.writestr("S1/track_summary.csv", b"tracks")
    manifest = {
        "artifacts": [
            {
                "specimen": "S1",
                "candidate": "production_morphology_neutral",
                "source_2d_detections_sha256": module.sha256_bytes(b"original"),
                "tracked_csv_sha256": module.sha256_bytes(b"tracked"),
                "tracks_csv_sha256": module.sha256_bytes(b"tracks"),
            }
        ]
    }
    with pytest.raises(ValueError, match="Retained replay hash mismatch"):
        module.validate_retained_replay_archive(archive_path, manifest)
