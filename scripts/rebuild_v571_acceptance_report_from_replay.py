"""Rebuild v5.7.1 acceptance reports from frozen 2D detections and tracking replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "sperm_segmentation_saturnv5.7.1.py"
REPORT_SCRIPT = ROOT / "scripts" / "generate_v57_biological_comparison.py"
SENSITIVITY_SCRIPT = ROOT / "scripts" / "generate_specimen_sensitivity_artifact.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_retained_replay_archive(archive_path, replay_manifest):
    """Verify that durable replay evidence contains the exact files used."""
    expected = {}
    for record in replay_manifest.get("artifacts", []):
        if record.get("candidate") != "production_morphology_neutral":
            continue
        specimen = str(record["specimen"])
        expected[f"{specimen}/source_2d_detections.csv"] = record[
            "source_2d_detections_sha256"
        ]
        expected[f"{specimen}/tracked_detections.csv"] = record[
            "tracked_csv_sha256"
        ]
        expected[f"{specimen}/track_summary.csv"] = record["tracks_csv_sha256"]
    if not expected:
        raise ValueError("Replay manifest has no production evidence records")

    observed = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = sorted(set(expected) - names)
        if missing:
            raise ValueError(f"Retained replay archive is missing: {missing}")
        for name, expected_hash in expected.items():
            observed_hash = sha256_bytes(archive.read(name))
            if observed_hash != expected_hash:
                raise ValueError(f"Retained replay hash mismatch for {name}")
            observed[name] = observed_hash
    return observed


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_report_replay", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def technical_valid_count(path):
    frame = pd.read_csv(path)
    if "technical_valid" not in frame:
        return int(len(frame))
    valid = (
        frame["technical_valid"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    return int(valid.sum())


def assert_replay_summary_binding(summary, specimen, tracked_path, tracks_path):
    rows = summary[
        (summary["specimen"].astype(str) == str(specimen))
        & (summary["candidate"].astype(str) == "production_morphology_neutral")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one production replay summary for {specimen}; found {len(rows)}")
    row = rows.iloc[0]
    tracked = pd.read_csv(tracked_path)
    tracks = pd.read_csv(tracks_path)
    if int(row["detections_2d"]) != len(tracked):
        raise ValueError(f"2D replay count mismatch for {specimen}")
    if int(row["tracks"]) != len(tracks):
        raise ValueError(f"Track replay count mismatch for {specimen}")
    return {
        "detections_2d": int(len(tracked)),
        "all_reconstructed_tracks": int(len(tracks)),
        "technical_valid_tracks": technical_valid_count(tracks_path),
    }


def copy_if_present(source, destination):
    source = Path(source)
    if source.is_file():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-output", type=Path, required=True)
    parser.add_argument("--tracking-replay", type=Path, required=True)
    parser.add_argument("--output-study", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--retained-replay-archive", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_output.resolve()
    replay_root = args.tracking_replay.resolve()
    output_study = args.output_study.resolve()
    output_report = args.output_report.resolve()
    retained_archive = args.retained_replay_archive.resolve()
    output_study.mkdir(parents=True, exist_ok=True)
    output_report.mkdir(parents=True, exist_ok=True)

    manifest_path = study_root / "study_manifest.csv"
    state_path = study_root / "study_run_state.json"
    rows = pd.read_csv(manifest_path).to_dict(orient="records")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    replay_summary = pd.read_csv(replay_root / "tracking_replay_summary.csv")
    replay_manifest_path = replay_root / "tracking_replay_manifest.json"
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    archived_replay_hashes = validate_retained_replay_archive(
        retained_archive, replay_manifest
    )

    for name in (
        "study_manifest.csv",
        "runtime_parameters.json",
        "normalization_qc.json",
        "specimen_group_comparison_qc.json",
    ):
        copy_if_present(study_root / name, output_study / name)

    source_records = {
        record["specimen"]: record
        for record in replay_manifest.get("artifacts", [])
        if record.get("candidate") == "production_morphology_neutral"
    }
    binding_records = []
    rebuilt_state = dict(state)
    rebuilt_state["source_study_output"] = str(study_root)
    rebuilt_state["tracking_replay"] = str(replay_root)
    rebuilt_state["samples"] = dict(state.get("samples", {}))

    for row in rows:
        specimen = str(row["sample_id"])
        record = rebuilt_state["samples"].get(specimen, {})
        if record.get("status") != "complete":
            continue
        if specimen not in source_records:
            raise ValueError(f"Missing production replay provenance for {specimen}")
        source_attempt = Path(record["output_dir"])
        destination = output_study / "samples" / specimen / "attempt_001"
        destination.mkdir(parents=True, exist_ok=True)

        detections = next(source_attempt.glob("spermatid_measurements_*.csv"))
        tracked = replay_root / f"{specimen}_production_morphology_neutral_tracked.csv"
        tracks = replay_root / f"{specimen}_production_morphology_neutral_tracks.csv"
        counts = assert_replay_summary_binding(
            replay_summary, specimen, tracked, tracks
        )

        detections_copy = destination / "spermatid_measurements_v5.7.1-body-width.csv"
        tracked_copy = destination / "measurements_with_tracks_v5.7.1-body-width.csv"
        tracks_copy = destination / "track_summary_v5.7.1-body-width.csv"
        shutil.copy2(detections, detections_copy)
        shutil.copy2(tracked, tracked_copy)
        shutil.copy2(tracks, tracks_copy)
        copy_if_present(
            source_attempt / "stack_preprocessing_qc.json",
            destination / "stack_preprocessing_qc.json",
        )
        copy_if_present(
            source_attempt / "calibration_used.json",
            destination / "calibration_used.json",
        )

        record = dict(record)
        record["output_dir"] = str(destination)
        rebuilt_state["samples"][specimen] = record
        binding_records.append(
            {
                "specimen": specimen,
                **counts,
                "frozen_2d_detections_sha256": sha256(detections_copy),
                "replay_tracked_sha256": sha256(tracked_copy),
                "replay_tracks_sha256": sha256(tracks_copy),
            }
        )

    (output_study / "study_run_state.json").write_text(
        json.dumps(rebuilt_state, indent=2), encoding="utf-8"
    )
    saturn = load_pipeline()
    saturn._write_study_aggregates(output_study, rows, rebuilt_state)

    subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--study-output",
            str(output_study),
            "--output-folder",
            str(output_report),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(SENSITIVITY_SCRIPT),
            str(output_study / "specimen_technical_qc.csv"),
            str(output_report / "02_quality_control" / "data" / "specimen_sensitivity_artifact.csv"),
            str(output_report / "02_quality_control" / "data" / "specimen_sensitivity_artifact.json"),
        ],
        check=True,
        cwd=ROOT,
    )

    report_specimens = pd.read_csv(
        output_report / "01_biological_results" / "data" / "specimen_biological_measurements.csv"
    )
    sensitivity = json.loads(
        (output_report / "02_quality_control" / "data" / "specimen_sensitivity_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    sensitivity_by_specimen = {
        row["sample_id"]: row for row in sensitivity.get("rows", []) if row.get("analysis_included")
    }
    for binding in binding_records:
        specimen = binding["specimen"]
        report_row = report_specimens[report_specimens["sample_id"].astype(str) == specimen]
        if len(report_row) != 1:
            raise ValueError(f"Report does not contain exactly one row for {specimen}")
        sensitivity_count = int(sensitivity_by_specimen[specimen]["technical_valid_count"])
        if sensitivity_count != binding["technical_valid_tracks"]:
            raise ValueError(f"Sensitivity count mismatch for {specimen}")

    binding = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_study_output": str(study_root),
        "tracking_replay": str(replay_root),
        "tracking_replay_manifest_sha256": sha256(replay_manifest_path),
        "tracking_generation_commit": replay_manifest.get("git_commit_at_generation"),
        "tracking_pipeline_working_copy_sha256": replay_manifest.get(
            "pipeline_working_copy_sha256"
        ),
        "tracking_pipeline_git_blob_sha256": replay_manifest.get(
            "pipeline_git_blob_sha256"
        ),
        "report_generation_pipeline_working_copy_sha256": sha256(PIPELINE_PATH),
        "report_rebuild_script_sha256": sha256(Path(__file__)),
        "retained_replay_archive": display_path(retained_archive),
        "retained_replay_archive_sha256": sha256(retained_archive),
        "retained_replay_members": archived_replay_hashes,
        "records": binding_records,
        "assertions": {
            "report_and_sensitivity_use_replay_tracks": True,
            "all_counts_reconciled": True,
            "retained_replay_archive_verified": True,
            "unet_inference_rerun": False,
        },
    }
    (output_report / "report_source_binding.json").write_text(
        json.dumps(binding, indent=2), encoding="utf-8"
    )
    print(json.dumps(binding, indent=2))


if __name__ == "__main__":
    main()
