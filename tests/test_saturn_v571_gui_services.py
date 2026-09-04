from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from utils.saturn_v571_gui_services import (
    CorrectionEvent,
    PreflightIssue,
    PreflightReport,
    PRODUCTION_REQUIRED_CLAIM_IDS,
    active_correction_events,
    append_correction_event,
    correction_label_state_sha256,
    load_correction_events,
    preview_false_detection_exclusion,
    production_audit_gate_state,
    reduce_study_progress,
    replay_false_detection_exclusions,
)


def test_production_audit_gate_requires_accepted_latest_verdicts(tmp_path):
    registry = tmp_path / "audits" / "claims_registry.json"
    registry.parent.mkdir()

    def write(status="accepted", gate_passed=True):
        registry.write_text(
            json.dumps(
                {
                    "claims": [
                        {
                            "claim_id": claim_id,
                            "status": status,
                            "latest_audit": {
                                "gate_passed": gate_passed,
                                "decision": "accepted" if gate_passed else "not_accepted",
                            },
                        }
                        for claim_id in PRODUCTION_REQUIRED_CLAIM_IDS
                    ]
                }
            ),
            encoding="utf-8",
        )

    write()
    assert production_audit_gate_state(tmp_path)[0] is True
    write(status="validated", gate_passed=False)
    ready, detail = production_audit_gate_state(tmp_path)
    assert ready is False
    assert "not accepted" in detail


def _event(**overrides):
    values = {
        "correction_uuid": str(uuid4()),
        "revision": 1,
        "specimen_id": "specimen-01",
        "z_index": 12,
        "action": "confirm_instance",
        "source_instance_ids": ("specimen-01:z12:i7",),
        "technical_reason": "",
        "reviewer": "reviewer-01",
        "timestamp_utc": "2026-08-28T15:30:00Z",
        "base_run_manifest_sha256": "a" * 64,
        "software_version": "v5.7.1",
        "analysis_profile_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "calibration_provenance_sha256": "d" * 64,
        "evidence_references": ("raw_evidence/instance_labels/z0012/evidence.json",),
        "notes": "Reviewed against the raw image.",
        "before_hash": "e" * 64,
        "after_hash": "f" * 64,
        "supersedes": None,
    }
    values.update(overrides)
    return CorrectionEvent(**values)


def _label_evidence():
    instances = np.zeros((8, 9), dtype=np.uint32)
    instances[1:4, 1:4] = 7
    instances[4:7, 5:8] = 12
    centerlines = np.zeros_like(instances)
    centerlines[2, 1:4] = 7
    centerlines[5, 5:8] = 12
    return instances, centerlines


def test_false_detection_exclusion_replays_from_hash_verified_filled_labels():
    instances, centerlines = _label_evidence()
    source_id = "specimen-01:z0012:instance:7"
    source_map = {source_id: 7}
    preview_instances, preview_centerlines, before_hash, after_hash = (
        preview_false_detection_exclusion(
            instances,
            centerlines,
            source_instance_id=source_id,
            source_label_map=source_map,
        )
    )
    event = _event(
        action="exclude_false_detection",
        source_instance_ids=(source_id,),
        technical_reason="weak_isolated_noise",
        evidence_references=("raw_evidence/instance_labels/z0012/evidence.json",),
        before_hash=before_hash,
        after_hash=after_hash,
    )
    corrected_instances, corrected_centerlines = replay_false_detection_exclusions(
        instances,
        centerlines,
        (event,),
        specimen_id="specimen-01",
        z_index=12,
        source_label_map=source_map,
    )

    assert np.array_equal(corrected_instances, preview_instances)
    assert np.array_equal(corrected_centerlines, preview_centerlines)
    assert np.any(corrected_instances == 12)
    assert not np.any(corrected_instances == 7)
    assert not np.any(corrected_centerlines == 7)
    assert np.any(instances == 7), "authoritative input evidence must remain unchanged"
    assert correction_label_state_sha256(corrected_instances, corrected_centerlines) == after_hash


def test_undo_last_exclusion_replays_from_raw_evidence():
    instances, centerlines = _label_evidence()
    source_id = "specimen-01:z0012:instance:7"
    source_map = {source_id: 7}
    _preview_i, _preview_c, before_hash, excluded_hash = (
        preview_false_detection_exclusion(
            instances,
            centerlines,
            source_instance_id=source_id,
            source_label_map=source_map,
        )
    )
    exclusion = _event(
        action="exclude_false_detection",
        source_instance_ids=(source_id,),
        technical_reason="weak_isolated_noise",
        before_hash=before_hash,
        after_hash=excluded_hash,
    )
    undo = _event(
        correction_uuid=str(uuid4()),
        revision=2,
        action="undo",
        source_instance_ids=(),
        technical_reason="",
        evidence_references=(),
        before_hash=excluded_hash,
        after_hash=before_hash,
        supersedes=exclusion.correction_uuid,
    )

    restored_instances, restored_centerlines = replay_false_detection_exclusions(
        instances,
        centerlines,
        (exclusion, undo),
        specimen_id="specimen-01",
        z_index=12,
        source_label_map=source_map,
    )

    assert np.array_equal(restored_instances, instances)
    assert np.array_equal(restored_centerlines, centerlines)
    assert active_correction_events((exclusion, undo)) == ()


def test_undo_rejects_nonlatest_target_and_undo_of_undo():
    first = _event(action="confirm_instance")
    second = _event(
        correction_uuid=str(uuid4()),
        revision=2,
        action="confirm_instance",
    )
    nonlatest = _event(
        correction_uuid=str(uuid4()),
        revision=3,
        action="undo",
        source_instance_ids=(),
        evidence_references=(),
        supersedes=first.correction_uuid,
    )
    with pytest.raises(ValueError, match="most recent active"):
        active_correction_events((first, second, nonlatest))


def test_correction_replay_rejects_stale_hash_wrong_plane_and_unsupported_action():
    instances, centerlines = _label_evidence()
    source_id = "specimen-01:z0012:instance:7"
    source_map = {source_id: 7}
    _, _, before_hash, after_hash = preview_false_detection_exclusion(
        instances,
        centerlines,
        source_instance_id=source_id,
        source_label_map=source_map,
    )
    valid_values = {
        "action": "exclude_false_detection",
        "source_instance_ids": (source_id,),
        "technical_reason": "weak_isolated_noise",
        "evidence_references": ("raw_evidence/instance_labels/z0012/evidence.json",),
        "before_hash": before_hash,
        "after_hash": after_hash,
    }
    with pytest.raises(ValueError, match="before_hash"):
        stale_values = dict(valid_values)
        stale_values["before_hash"] = "0" * 64
        replay_false_detection_exclusions(
            instances,
            centerlines,
            (_event(**stale_values),),
            specimen_id="specimen-01",
            z_index=12,
            source_label_map=source_map,
        )
    with pytest.raises(ValueError, match="different Z plane"):
        replay_false_detection_exclusions(
            instances,
            centerlines,
            (_event(**valid_values),),
            specimen_id="specimen-01",
            z_index=11,
            source_label_map=source_map,
        )
    with pytest.raises(ValueError, match="not executable"):
        replay_false_detection_exclusions(
            instances,
            centerlines,
            (_event(before_hash=before_hash, after_hash=before_hash),),
            specimen_id="specimen-01",
            z_index=12,
            source_label_map=source_map,
        )


def test_false_detection_replay_does_not_allow_morphology_as_exclusion_reason():
    instances, centerlines = _label_evidence()
    source_id = "specimen-01:z0012:instance:7"
    _, _, before_hash, after_hash = preview_false_detection_exclusion(
        instances,
        centerlines,
        source_instance_id=source_id,
        source_label_map={source_id: 7},
    )
    with pytest.raises(ValueError, match="morphology alone"):
        _event(
            action="exclude_false_detection",
            source_instance_ids=(source_id,),
            technical_reason="short",
            before_hash=before_hash,
            after_hash=after_hash,
        )


def test_authoritative_label_hash_rejects_float_negative_and_uncontained_centerline():
    instances, centerlines = _label_evidence()
    with pytest.raises(ValueError, match="integer labels"):
        correction_label_state_sha256(instances.astype(float), centerlines)
    negative = instances.astype(np.int32)
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        correction_label_state_sha256(negative, centerlines)
    outside = centerlines.copy()
    outside[0, 0] = 7
    with pytest.raises(ValueError, match="contained"):
        correction_label_state_sha256(instances, outside)


def test_preflight_report_is_immutable_and_summarizes_issues():
    blocking = PreflightIssue(
        "ROI_MISSING", "block", "ROI missing", "No ROI was resolved.", "Select an ROI."
    )
    warning = PreflightIssue(
        "CAL_FALLBACK",
        "warning",
        "Fallback calibration",
        "Metadata calibration was unavailable.",
        "Review the calibration source.",
    )
    report = PreflightReport([blocking, warning])

    assert report.ready is False
    assert report.blocking_issues == (blocking,)
    assert report.issues_summary == {"block": 1, "warning": 1, "info": 0, "total": 2}
    assert report.to_dict()["blocking_summary"] == "ROI_MISSING: ROI missing"
    with pytest.raises(FrozenInstanceError):
        blocking.title = "changed"
    with pytest.raises(FrozenInstanceError):
        report.issues = ()


def test_preflight_report_without_blocks_is_ready():
    report = PreflightReport(
        (PreflightIssue("OK", "info", "Ready", "Checks passed.", "Continue."),)
    )
    assert report.ready is True
    assert report.blocking_summary == "No blocking preflight issues."


@pytest.mark.parametrize(
    ("event_type", "expected_run_status", "expected_specimen_status"),
    [
        ("started", "running", "started"),
        ("slice_progress", "running", "slice_progress"),
        ("postprocess_progress", "running", "postprocess_progress"),
        ("complete", "running", "complete"),
        ("failed", "running", "failed"),
        ("skipped", "running", "skipped"),
    ],
)
def test_progress_reducer_covers_specimen_transitions(
    event_type, expected_run_status, expected_specimen_status
):
    original = {"status": "idle", "specimens": {"existing": {"status": "complete"}}}
    reduced = reduce_study_progress(
        original,
        {"event": event_type, "specimen_id": "specimen-01", "message": "Status changed."},
    )

    assert reduced is not original
    assert reduced["specimens"] is not original["specimens"]
    assert original == {"status": "idle", "specimens": {"existing": {"status": "complete"}}}
    assert reduced["status"] == expected_run_status
    assert reduced["specimens"]["specimen-01"] == {
        "status": expected_specimen_status,
        "message": "Status changed.",
    }
    assert reduced["counts"][expected_specimen_status] >= 1


def test_progress_reducer_covers_reporting_and_stopped():
    reporting = reduce_study_progress({}, {"event": "reporting", "message": "Building reports."})
    stopped = reduce_study_progress(reporting, {"event": "stopped", "message": "Stopped safely."})

    assert reporting["status"] == "reporting"
    assert reporting["current_specimen_id"] is None
    assert stopped["status"] == "stopped"
    assert stopped["message"] == "Stopped safely."
    assert reporting["status"] == "reporting"


def test_correction_event_is_immutable_and_round_trips():
    event = _event()
    restored = CorrectionEvent.from_dict(event.to_dict())
    assert restored == event
    assert isinstance(restored.source_instance_ids, tuple)
    with pytest.raises(FrozenInstanceError):
        event.action = "undo"


def test_invalid_correction_action_is_rejected():
    with pytest.raises(ValueError, match="action must be one of"):
        _event(action="delete_everything")


def test_exclusion_requires_technical_reason():
    with pytest.raises(ValueError, match="requires a technical_reason"):
        _event(action="exclude_false_detection", technical_reason="")


@pytest.mark.parametrize(
    "reason",
    ["short", "wide", "single_slice", "low-length-to-width-ratio", "mutant like irregularity"],
)
def test_morphology_only_technical_reasons_are_forbidden(reason):
    with pytest.raises(ValueError, match="morphology alone"):
        _event(action="exclude_false_detection", technical_reason=reason)


def test_free_text_technical_reason_is_rejected():
    with pytest.raises(ValueError, match="controlled technical code"):
        _event(
            action="exclude_false_detection",
            technical_reason="looks like a false positive",
        )


def test_duplicate_source_instance_ids_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        _event(source_instance_ids=("z0001:instance:1", "z0001:instance:1"))


@pytest.mark.parametrize(
    "action",
    ["add_instance", "replace_boundary", "split_instance", "merge_instances", "must_link", "cannot_link"],
)
def test_technical_edits_require_controlled_reason(action):
    source_ids = {
        "add_instance": (),
        "merge_instances": ("z1:i1", "z1:i2"),
        "must_link": ("z1:i1", "z2:i1"),
        "cannot_link": ("z1:i1", "z2:i1"),
    }.get(action, ("z1:i1",))
    with pytest.raises(ValueError, match="requires a technical_reason"):
        _event(action=action, technical_reason="", source_instance_ids=source_ids)


def test_action_reason_compatibility_is_enforced():
    with pytest.raises(ValueError, match="not compatible"):
        _event(
            action="exclude_false_detection",
            technical_reason="confirmed_multi_object_join",
        )


def test_hashes_and_objective_evidence_are_required():
    with pytest.raises(ValueError, match="64-character SHA-256"):
        _event(checkpoint_sha256="not-a-hash")
    with pytest.raises(ValueError, match="objective evidence_references"):
        _event(
            action="exclude_false_detection",
            technical_reason="weak_isolated_noise",
            evidence_references=(),
        )


def test_append_is_byte_preserving_and_load_validates(tmp_path: Path):
    path = tmp_path / "corrections" / "operations.jsonl"
    first = _event(revision=1)
    second = _event(revision=2, action="mark_uncertain", source_instance_ids=("s:z12:i8",))

    append_correction_event(path, first)
    original_bytes = path.read_bytes()
    append_correction_event(path, second)
    final_bytes = path.read_bytes()

    assert final_bytes[: len(original_bytes)] == original_bytes
    assert final_bytes.startswith(original_bytes)
    assert load_correction_events(path) == (first, second)


def test_append_rejects_duplicate_uuid_and_revision_gap(tmp_path: Path):
    path = tmp_path / "operations.jsonl"
    first = _event(revision=1)
    append_correction_event(path, first)

    with pytest.raises(ValueError, match="duplicate correction_uuid"):
        append_correction_event(path, _event(correction_uuid=first.correction_uuid, revision=2))
    with pytest.raises(ValueError, match="expected revision 2"):
        append_correction_event(path, _event(revision=3))


def test_append_rejects_provenance_drift_and_unknown_supersedes(tmp_path: Path):
    path = tmp_path / "operations.jsonl"
    first = _event(revision=1)
    append_correction_event(path, first)

    with pytest.raises(ValueError, match="provenance cannot change"):
        append_correction_event(
            path,
            _event(revision=2, checkpoint_sha256="9" * 64),
        )
    with pytest.raises(ValueError, match="must reference an earlier"):
        append_correction_event(path, _event(revision=2, supersedes=str(uuid4())))


def test_load_rejects_invalid_existing_record(tmp_path: Path):
    path = tmp_path / "operations.jsonl"
    path.write_text('{"action":"not_valid"}\n', encoding="ascii")
    with pytest.raises(ValueError, match="invalid correction record at line 1"):
        load_correction_events(path)
