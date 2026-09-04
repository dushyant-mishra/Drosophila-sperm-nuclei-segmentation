import importlib.util
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_v571_width_z_repeatability.py"
    spec = importlib.util.spec_from_file_location(
        "v571_width_z_repeatability_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv_bytes(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def make_archive(path, duplicate_z=False):
    specimens = {
        "kj_sv_40xx0.75-1": {
            "track_id": 1,
            "widths": [2.0, 2.1, 2.0],
            "areas": [10.0, 20.0, 12.0],
            "representative_z": 1,
        },
        "w1118_sv_feb_40xx0.75-1": {
            "track_id": 10,
            "widths": [1.0, 2.0, 1.0],
            "areas": [10.0, 30.0, 10.0],
            "representative_z": 1,
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        for specimen, values in specimens.items():
            detections = pd.DataFrame(
                {
                    "track_id": [values["track_id"]] * 3,
                    "z_slice": [0, 1, 2],
                    "body_width_um": values["widths"],
                    "instance_mask_area_px": values["areas"],
                }
            )
            if duplicate_z and specimen.startswith("kj_"):
                detections = pd.concat([detections, detections.iloc[[1]]], ignore_index=True)
            summaries = pd.DataFrame(
                {
                    "track_id": [values["track_id"], values["track_id"] + 1],
                    "technical_valid": [True, False],
                    "observed_slice_count": [3, 4],
                    "representative_width_z": [values["representative_z"], 1],
                    "morphology_warning": [True, False],
                }
            )
            archive.writestr(
                f"{specimen}/tracked_detections.csv", _csv_bytes(detections)
            )
            archive.writestr(
                f"{specimen}/track_summary.csv", _csv_bytes(summaries)
            )


def test_analyze_archive_uses_technical_valid_observed_width_planes(tmp_path):
    validator = load_validator()
    archive = tmp_path / "replay.zip"
    make_archive(archive)

    frame, summary = validator.analyze_archive(archive)

    assert list(frame["specimen_id"]) == ["KJ-01", "WT-01"]
    assert set(frame["width_available_plane_count"]) == {3}
    assert frame.loc[frame["specimen_id"] == "KJ-01", "width_cv"].iloc[0] == pytest.approx(
        pd.Series([2.0, 2.1, 2.0]).std(ddof=1) / pd.Series([2.0, 2.1, 2.0]).mean()
    )
    assert frame.loc[frame["specimen_id"] == "KJ-01", "width_range_um"].iloc[0] == pytest.approx(0.1)
    assert not bool(
        frame.loc[
            frame["specimen_id"] == "KJ-01",
            "representative_materially_differs_from_adjacent",
        ].iloc[0]
    )
    assert bool(
        frame.loc[
            frame["specimen_id"] == "WT-01",
            "representative_materially_differs_from_adjacent",
        ].iloc[0]
    )
    assert summary["analysis_type"] == "repeatability_qc"
    assert summary["biological_truth_status"] == "not_established"
    assert "biological" not in summary["primary_conclusion"].lower()


def test_outputs_are_compact_and_state_the_non_biological_scope(tmp_path):
    validator = load_validator()
    archive = tmp_path / "replay.zip"
    output = tmp_path / "output"
    make_archive(archive)
    frame, summary = validator.analyze_archive(archive)

    validator.write_outputs(frame, summary, output)

    assert {path.name for path in output.iterdir()} == {
        "v571_width_z_repeatability_tracks.csv",
        "v571_width_z_repeatability_summary.json",
        "V5_7_1_WIDTH_Z_REPEATABILITY_QC.md",
        "COMPLETED.json",
    }
    saved = json.loads(
        (output / "v571_width_z_repeatability_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["width_field"] == "body_width_um"
    assert "legacy" not in saved["width_field"]
    assert saved["generated_at_utc"]
    assert saved["generator_sha256"]
    assert saved["environment"]["python_version"]
    artifacts = {item["path"]: item["sha256"] for item in saved["artifacts"]}
    assert artifacts == {
        "v571_width_z_repeatability_tracks.csv": hashlib.sha256(
            (output / "v571_width_z_repeatability_tracks.csv").read_bytes()
        ).hexdigest(),
        "V5_7_1_WIDTH_Z_REPEATABILITY_QC.md": hashlib.sha256(
            (output / "V5_7_1_WIDTH_Z_REPEATABILITY_QC.md").read_bytes()
        ).hexdigest(),
    }
    report = (output / "V5_7_1_WIDTH_Z_REPEATABILITY_QC.md").read_text(
        encoding="utf-8"
    )
    assert report.count("## Primary conclusion") == 1
    assert "repeatability QC" in report
    assert "not biological truth or group inference" in report
    assert "does not invalidate a track" in report
    completion = json.loads(
        (output / "COMPLETED.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "complete"
    assert completion["biological_truth_status"] == "not_established"
    assert completion["summary_sha256"] == hashlib.sha256(
        (output / "v571_width_z_repeatability_summary.json").read_bytes()
    ).hexdigest()
    assert not list(output.glob(".*.tmp"))


def test_duplicate_observations_from_one_plane_fail_closed(tmp_path):
    validator = load_validator()
    archive = tmp_path / "duplicate.zip"
    make_archive(archive, duplicate_z=True)

    with pytest.raises(ValueError, match="multiple width observations from the same Z plane"):
        validator.analyze_archive(archive)


def test_repeatability_outputs_fail_closed_if_directory_exists(tmp_path):
    validator = load_validator()
    archive = tmp_path / "replay.zip"
    make_archive(archive)
    frame, summary = validator.analyze_archive(archive)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validator.write_outputs(frame, summary, empty)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    old = occupied / "old.txt"
    old.write_text("old evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        validator.write_outputs(frame, summary, occupied)
    assert old.read_text(encoding="utf-8") == "old evidence"
