import hashlib
import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import tifffile

from utils.saturn_v571_correction_materializer import (
    create_correction_base_manifest,
    materialize_false_detection_revision,
    validate_correction_revision,
)
from utils.saturn_v571_gui_services import (
    CorrectionEvent,
    active_correction_events,
    append_correction_event,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_run(tmp_path):
    output = tmp_path / "completed"
    settings = output / "settings"
    evidence = output / "raw_evidence" / "instance_labels" / "z0001"
    settings.mkdir(parents=True)
    evidence.mkdir(parents=True)
    (output / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    runtime = settings / "runtime_parameters.json"
    profile = settings / "analysis_profile_used.json"
    checkpoint = settings / "model.pt"
    calibration = settings / "calibration_used.json"
    source_manifest = settings / "source_image_manifest.json"
    runtime.write_text(
        json.dumps({"DO_TRACKING": True, "UM_PER_PX_XY": 0.5}),
        encoding="utf-8",
    )
    profile.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"model")
    calibration.write_text(json.dumps({"xy_um_per_pixel": 0.5}), encoding="utf-8")
    source_manifest.write_text(
        json.dumps(
            {
                "ordered_source_images": [
                    {"z_index": 1, "sha256": "1" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )
    records = []
    for role, path in (
        ("runtime_parameters", runtime),
        ("loaded_analysis_profile", profile),
        ("unet_checkpoint", checkpoint),
        ("resolved_calibration", calibration),
        ("source_image_manifest", source_manifest),
    ):
        records.append({"role": role, "copied_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = settings / "settings_manifest.json"
    manifest.write_text(json.dumps({"pipeline_version": "v5.7.1", "files": records}), encoding="utf-8")
    instances = np.zeros((8, 8), dtype=np.uint32)
    instances[1:3, 1:3] = 1
    instances[4:7, 4:7] = 2
    centerlines = np.zeros_like(instances)
    centerlines[1, 1:3] = 1
    centerlines[5, 4:7] = 2
    ip = evidence / "instance_labels.tif"
    cp = evidence / "centerline_labels.tif"
    tifffile.imwrite(ip, instances)
    tifffile.imwrite(cp, centerlines)
    (evidence / "evidence.json").write_text(json.dumps({"instance_labels_sha256": sha(ip), "centerline_labels_sha256": sha(cp)}), encoding="utf-8")
    pd.DataFrame({
        "source_instance_key": ["z0001:instance:1", "z0001:instance:2"],
        "z_slice": [1, 1],
        "unet_mean_probability": [0.10, 0.90],
        "unet_max_probability": [0.20, 0.99],
    }).to_csv(output / "spermatid_measurements_v5.7.1.csv", index=False)
    base_manifest, _base_hash = create_correction_base_manifest(
        output, specimen_id="sample-01"
    )
    return output, base_manifest, profile, checkpoint, calibration, instances, centerlines


def event_for(output, manifest, profile, checkpoint, calibration, instances, centerlines):
    from utils.saturn_v571_gui_services import preview_false_detection_exclusion
    _a, _b, before, after = preview_false_detection_exclusion(instances, centerlines, source_instance_id="z0001:instance:1", source_label_map={"z0001:instance:1": 1, "z0001:instance:2": 2})
    evidence = output / "corrections" / "review_evidence" / "exclude_z0001_instance1.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "saturn_v571_exclusion_evidence/1.0",
                "source_instance_id": "z0001:instance:1",
                "z_index": 1,
                "technical_reason": "weak_isolated_noise",
                "label_state_before_sha256": before,
                "observed_unet_mean_probability": 0.10,
                "observed_unet_max_probability": 0.20,
                "source_image_sha256": "1" * 64,
                "reviewer_confirmed_non_nuclear_signal": True,
            }
        ),
        encoding="utf-8",
    )
    return CorrectionEvent(
        correction_uuid=str(uuid4()), revision=1, specimen_id="sample-01", z_index=1,
        action="exclude_false_detection", source_instance_ids=("z0001:instance:1",),
        technical_reason="weak_isolated_noise", reviewer="reviewer", timestamp_utc="2026-08-31T12:00:00Z",
        base_run_manifest_sha256=sha(manifest), software_version="v5.7.1",
        analysis_profile_sha256=sha(profile), checkpoint_sha256=sha(checkpoint),
        calibration_provenance_sha256=sha(calibration),
        evidence_references=(
            "raw_evidence/instance_labels/z0001/evidence.json",
            "corrections/review_evidence/exclude_z0001_instance1.json",
        ),
        before_hash=before, after_hash=after,
    )


def tracker(frame, _cfg):
    tracked = frame.copy()
    tracked["track_id"] = range(1, len(tracked) + 1)
    return tracked, pd.DataFrame({"track_id": list(tracked["track_id"])})


def retrack(frame, events, cfg, specimen_id, tracking_callable=None):
    excluded = {
        source
        for event in active_correction_events(tuple(events))
        for source in event.source_instance_ids
    }
    kept = frame[~frame["source_instance_key"].isin(excluded)].copy()
    tracked, tracks = (tracking_callable or tracker)(kept, cfg)
    return tracked, tracks, {"full_specimen_retracking": True}


def test_materializer_verifies_provenance_and_promotes_complete_revision(tmp_path):
    output, manifest, profile, checkpoint, calibration, instances, centerlines = base_run(tmp_path)
    log = tmp_path / "events.jsonl"
    append_correction_event(log, event_for(output, manifest, profile, checkpoint, calibration, instances, centerlines))

    def reports(temp, *_args):
        path = temp / "biologist_results" / "summary.pdf"
        path.parent.mkdir()
        path.write_bytes(b"pdf")
        return [path]

    revision = materialize_false_detection_revision(output, log, retrack_callable=retrack, tracking_callable=tracker, artifact_callback=reports)
    assert revision.name == "revision_0001"
    assert (revision / "revision_complete.json").is_file()
    validate_correction_revision(revision, expected_base_run_manifest_sha256=sha(output / "corrections" / "correction_base_manifest.json"))
    corrected = tifffile.imread(revision / "instance_labels" / "z0001" / "instance_labels.tif")
    assert 1 not in np.unique(corrected)
    assert 2 in np.unique(corrected)
    assert (output / "raw_evidence" / "instance_labels" / "z0001" / "instance_labels.tif").is_file()


def test_materializer_failure_is_atomic_and_tampering_is_rejected(tmp_path):
    output, manifest, profile, checkpoint, calibration, instances, centerlines = base_run(tmp_path)
    log = tmp_path / "events.jsonl"
    append_correction_event(log, event_for(output, manifest, profile, checkpoint, calibration, instances, centerlines))

    def fails(*_args):
        raise RuntimeError("synthetic report failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        materialize_false_detection_revision(output, log, retrack_callable=retrack, tracking_callable=tracker, artifact_callback=fails)
    assert not (output / "corrections" / "revision_0001").exists()
    assert not list((output / "corrections").glob(".revision_*.tmp-*"))

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="no longer matches"):
        materialize_false_detection_revision(output, log, retrack_callable=retrack, tracking_callable=tracker, artifact_callback=lambda *_: [])


def test_revision_validator_rejects_post_promotion_tampering(tmp_path):
    output, manifest, profile, checkpoint, calibration, instances, centerlines = base_run(tmp_path)
    log = tmp_path / "events.jsonl"
    append_correction_event(log, event_for(output, manifest, profile, checkpoint, calibration, instances, centerlines))

    def reports(temp, *_args):
        path = temp / "summary.pdf"
        path.write_bytes(b"pdf")
        return [path]

    revision = materialize_false_detection_revision(output, log, retrack_callable=retrack, tracking_callable=tracker, artifact_callback=reports)
    (revision / "summary.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash changed"):
        validate_correction_revision(revision)


def test_pipeline_coordinator_wires_corrected_reports_and_overlays(
    tmp_path, monkeypatch
):
    output, manifest, profile, checkpoint, calibration, instances, centerlines = base_run(tmp_path)
    log = tmp_path / "events.jsonl"
    append_correction_event(log, event_for(output, manifest, profile, checkpoint, calibration, instances, centerlines))
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_materializer_integration",
        Path(__file__).resolve().parents[1] / "sperm_segmentation_saturnv5.7.1.py",
    )
    saturn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saturn)
    monkeypatch.setattr(saturn, "retrack_false_detection_corrections", retrack)

    def write_file(path, content=b"artifact"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    monkeypatch.setattr(
        saturn,
        "_corrected_revision_overlays",
        lambda _base, temp: [write_file(Path(temp) / "analysis_overlays" / "z0001.png")],
    )
    monkeypatch.setattr(
        saturn,
        "export_comparative_track_tables",
        lambda temp, *_args: [write_file(Path(temp) / "technical_qc" / "tracks.csv")],
    )

    def biological(temp, *_args):
        return {
            "summary": str(write_file(Path(temp) / "biologist_results" / "sample_summary.csv")),
            "nuclei": str(write_file(Path(temp) / "biologist_results" / "nuclei.csv")),
            "readme": str(write_file(Path(temp) / "biologist_results" / "README.txt")),
        }

    monkeypatch.setattr(saturn, "export_biologist_results", biological)
    monkeypatch.setattr(saturn, "export_analysis_summary", lambda *_a, **_k: {"estimated_unique_nuclei": 1})
    monkeypatch.setattr(
        saturn,
        "generate_concise_biologist_pdf",
        lambda temp, *_a: str(write_file(Path(temp) / "biologist_results" / "report.pdf")),
    )
    monkeypatch.setattr(
        saturn,
        "generate_concise_biologist_pptx",
        lambda temp: str(write_file(Path(temp) / "biologist_results" / "report.pptx")),
    )
    monkeypatch.setattr(
        saturn,
        "generate_batch_report",
        lambda temp, *_a, **_k: write_file(Path(temp) / "technical_report.pdf"),
    )
    monkeypatch.setattr(
        saturn,
        "generate_excel_report",
        lambda temp, *_a: write_file(Path(temp) / "technical_report.xlsx"),
    )

    revision = saturn.materialize_reviewed_false_detection_revision(output, log)

    validated = validate_correction_revision(revision)
    assert validated["manifest"]["correction_revision"] == 1
    assert (revision / "correction_summary.json").is_file()
    assert (revision / "biologist_results" / "report.pdf").is_file()
    assert (revision / "analysis_overlays" / "z0001.png").is_file()


def test_second_revision_can_undo_latest_exclusion_without_changing_revision_one(tmp_path):
    output, manifest, profile, checkpoint, calibration, instances, centerlines = base_run(tmp_path)
    log = tmp_path / "events.jsonl"
    exclusion = event_for(output, manifest, profile, checkpoint, calibration, instances, centerlines)
    append_correction_event(log, exclusion)

    def reports(temp, *_args):
        path = temp / "summary.pdf"
        path.write_bytes(b"pdf")
        return [path]

    first_revision = materialize_false_detection_revision(
        output, log, retrack_callable=retrack, tracking_callable=tracker, artifact_callback=reports
    )
    first_hash = sha(first_revision / "instance_labels" / "z0001" / "instance_labels.tif")
    undo = CorrectionEvent(
        correction_uuid=str(uuid4()), revision=2, specimen_id="sample-01", z_index=1,
        action="undo", source_instance_ids=(), technical_reason="", reviewer="reviewer",
        timestamp_utc="2026-08-31T12:05:00Z", base_run_manifest_sha256=sha(manifest),
        software_version="v5.7.1", analysis_profile_sha256=sha(profile),
        checkpoint_sha256=sha(checkpoint), calibration_provenance_sha256=sha(calibration),
        evidence_references=(), before_hash=exclusion.after_hash,
        after_hash=exclusion.before_hash, supersedes=exclusion.correction_uuid,
    )
    append_correction_event(log, undo)
    second_revision = materialize_false_detection_revision(
        output, log, retrack_callable=retrack, tracking_callable=tracker, artifact_callback=reports
    )

    restored = tifffile.imread(second_revision / "instance_labels" / "z0001" / "instance_labels.tif")
    assert np.array_equal(restored, instances)
    assert sha(first_revision / "instance_labels" / "z0001" / "instance_labels.tif") == first_hash
    current = json.loads((output / "corrections" / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["correction_revision"] == 2
