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

import numpy as np
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


def git_blob_sha256(path, commit="HEAD"):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    ).stdout
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


def reconcile_below_2_um_sensitivity(sensitivity_path, track_paths):
    sensitivity = pd.read_csv(sensitivity_path)
    records = []
    numeric_fields = (
        "below_2_um_fraction",
        "primary_median_length_um",
        "sensitivity_median_length_um",
        "primary_median_body_width_um",
        "sensitivity_median_body_width_um",
        "primary_width_missing_fraction",
        "sensitivity_width_missing_fraction",
    )
    for specimen, track_path in track_paths.items():
        rows = sensitivity[sensitivity["sample_id"].astype(str) == str(specimen)]
        if len(rows) != 1:
            raise ValueError(f"Expected one below-2-um sensitivity row for {specimen}")
        observed = rows.iloc[0]
        tracks = pd.read_csv(track_path)
        valid = (
            tracks["technical_valid"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        length = pd.to_numeric(tracks["projection_z_extent_um"], errors="coerce")
        valid_length = length.loc[valid]
        if (
            valid_length.isna().any()
            or (~np.isfinite(valid_length)).any()
            or (valid_length < 0).any()
        ):
            raise ValueError(
                f"Invalid technical-valid projection_z_extent_um for {specimen}"
            )
        primary = tracks.loc[valid]
        without_short = tracks.loc[valid & (length >= 2.0)]

        def median(frame, field):
            return float(pd.to_numeric(frame[field], errors="coerce").median())

        def availability(frame, field):
            values = pd.to_numeric(frame[field], errors="coerce")
            available = int(np.isfinite(values).sum())
            missing_fraction = (
                float(1.0 - available / len(frame)) if len(frame) else float("nan")
            )
            return available, missing_fraction

        primary_width_n, primary_width_missing = availability(
            primary, "representative_body_width_um"
        )
        sensitivity_width_n, sensitivity_width_missing = availability(
            without_short, "representative_body_width_um"
        )

        expected = {
            "primary_technical_valid_count": int(valid.sum()),
            "below_2_um_count": int((valid & (length < 2.0)).sum()),
            "sensitivity_count_without_below_2_um": int(len(without_short)),
            "below_2_um_fraction": float(
                (valid & (length < 2.0)).sum() / max(valid.sum(), 1)
            ),
            "primary_median_length_um": median(primary, "projection_z_extent_um"),
            "sensitivity_median_length_um": median(
                without_short, "projection_z_extent_um"
            ),
            "primary_median_body_width_um": median(
                primary, "representative_body_width_um"
            ),
            "sensitivity_median_body_width_um": median(
                without_short, "representative_body_width_um"
            ),
            "primary_width_available_n": primary_width_n,
            "primary_width_missing_fraction": primary_width_missing,
            "sensitivity_width_available_n": sensitivity_width_n,
            "sensitivity_width_missing_fraction": sensitivity_width_missing,
        }
        for field in expected:
            if field in numeric_fields:
                if not np.isclose(
                    float(observed[field]), expected[field], equal_nan=True
                ):
                    raise ValueError(
                        f"Below-2-um sensitivity mismatch for {specimen}: {field}"
                    )
            elif int(observed[field]) != expected[field]:
                raise ValueError(
                    f"Below-2-um sensitivity mismatch for {specimen}: {field}"
                )
        if expected["primary_technical_valid_count"] != (
            expected["below_2_um_count"]
            + expected["sensitivity_count_without_below_2_um"]
        ):
            raise ValueError(f"Below-2-um count partition mismatch for {specimen}")
        records.append(
            {
                "sample_id": specimen,
                **expected,
                "interpretation": str(observed["interpretation"]),
                "track_summary_sha256": sha256(track_path),
            }
        )
    return records


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
    parser.add_argument("--reference-group", required=True)
    parser.add_argument("--comparison-group", required=True)
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
    groups = {str(row.get("group", "")).strip() for row in rows}
    if args.reference_group not in groups or args.comparison_group not in groups:
        raise ValueError("Explicit report groups must exist in the study manifest")
    if args.reference_group == args.comparison_group:
        raise ValueError("Reference and comparison groups must differ")
    for row in rows:
        group = str(row.get("group", "")).strip()
        row["group_role"] = (
            "reference"
            if group == args.reference_group
            else "comparison"
            if group == args.comparison_group
            else ""
        )
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
    retained_track_paths = {}
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
        retained_track_paths[specimen] = tracks_copy
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

    below_2_source = (
        output_study / "technical_qc" / "below_2_um_specimen_sensitivity.csv"
    )
    below_2_records = reconcile_below_2_um_sensitivity(
        below_2_source, retained_track_paths
    )

    subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--study-output",
            str(output_study),
            "--output-folder",
            str(output_report),
            "--reference-group",
            args.reference_group,
            "--comparison-group",
            args.comparison_group,
        ],
        check=True,
        cwd=ROOT,
    )
    below_2_csv = (
        output_report
        / "02_quality_control"
        / "data"
        / "below_2_um_specimen_sensitivity.csv"
    )
    below_2_json = below_2_csv.with_suffix(".json")
    shutil.copy2(below_2_source, below_2_csv)
    below_2_payload = {
        "schema_version": "1.0",
        "analysis_unit": "biological_specimen",
        "inference_performed": False,
        "population": "technical_valid reconstructed tracks",
        "length_field": "projection_z_extent_um",
        "threshold_um": 2.0,
        "primary_population_preserved": True,
        "sensitivity_population_role": "descriptive omission scenario only",
        "formula": (
            "primary_technical_valid_count = below_2_um_count + "
            "sensitivity_count_without_below_2_um"
        ),
        "records": below_2_records,
        "assertions": {
            "all_counts_and_medians_recomputed_from_track_summaries": True,
            "short_tracks_remain_in_primary_population": True,
            "no_second_accepted_population": True,
        },
    }
    below_2_json.write_text(json.dumps(below_2_payload, indent=2), encoding="utf-8")
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
        "report_rebuild_script_git_blob_sha256": git_blob_sha256(Path(__file__)),
        "report_rebuild_script_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "reference_group": args.reference_group,
        "comparison_group": args.comparison_group,
        "group_direction_source": (
            "explicit command arguments bound to study manifest groups"
        ),
        "retained_replay_archive": display_path(retained_archive),
        "retained_replay_archive_sha256": sha256(retained_archive),
        "retained_replay_members": archived_replay_hashes,
        "below_2_um_sensitivity_csv": display_path(below_2_csv),
        "below_2_um_sensitivity_csv_sha256": sha256(below_2_csv),
        "below_2_um_sensitivity_json": display_path(below_2_json),
        "below_2_um_sensitivity_json_sha256": sha256(below_2_json),
        "records": binding_records,
        "assertions": {
            "report_and_sensitivity_use_replay_tracks": True,
            "all_counts_reconciled": True,
            "retained_replay_archive_verified": True,
            "below_2_um_sensitivity_reconciled": True,
            "unet_inference_rerun": False,
        },
    }
    (output_report / "report_source_binding.json").write_text(
        json.dumps(binding, indent=2), encoding="utf-8"
    )
    print(json.dumps(binding, indent=2))


if __name__ == "__main__":
    main()
