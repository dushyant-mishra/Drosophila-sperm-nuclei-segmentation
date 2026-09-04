"""Pure GUI-facing services for Saturn v5.7.1.

This module deliberately has no Tk or pipeline dependencies. It provides
serializable state and append-only correction records that GUI adapters can
consume without changing frozen v5.7 behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
from uuid import UUID

import numpy as np


PREFLIGHT_SEVERITIES = frozenset({"block", "warning", "info"})
STUDY_PROGRESS_EVENTS = frozenset(
    {
        "started",
        "slice_progress",
        "postprocess_progress",
        "complete",
        "failed",
        "skipped",
        "reporting",
        "stopped",
    }
)
PRODUCTION_REQUIRED_CLAIM_IDS = (
    "PIPELINE-V571-PRODUCTION-001",
    "MEAS-BODY-WIDTH-001",
    "REPORT-BIOLOGIST-CONCISE-001",
    "WORKFLOW-GUI-PRIMARY-001",
)


def production_audit_gate_state(
    project_root: os.PathLike[str] | str,
    required_claim_ids=PRODUCTION_REQUIRED_CLAIM_IDS,
) -> Tuple[bool, str]:
    """Return a fail-closed production verdict from the durable claim registry."""
    registry_path = Path(project_root) / "audits" / "claims_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Claims registry is unavailable: {type(exc).__name__}: {exc}"
    claims = {
        str(claim.get("claim_id", "")): claim
        for claim in registry.get("claims", [])
        if isinstance(claim, dict)
    }
    blockers = []
    for claim_id in required_claim_ids:
        claim = claims.get(str(claim_id))
        if claim is None:
            blockers.append(f"{claim_id}: missing")
            continue
        audit = claim.get("latest_audit") or {}
        if not (
            str(claim.get("status", "")).lower() == "accepted"
            and bool(audit.get("gate_passed", False))
        ):
            decision = str(audit.get("decision", "not audited"))
            blockers.append(
                f"{claim_id}: {claim.get('status', 'unknown')} ({decision})"
            )
    if blockers:
        return False, "Required scientific claims are not accepted: " + "; ".join(blockers)
    return True, "All required production claims are accepted by their latest audit."
CORRECTION_ACTIONS = frozenset(
    {
        "confirm_instance",
        "exclude_false_detection",
        "add_instance",
        "replace_boundary",
        "split_instance",
        "merge_instances",
        "mark_uncertain",
        "must_link",
        "cannot_link",
        "confirm_single_slice",
        "mark_track_uncertain",
        "undo",
    }
)

TECHNICAL_REASON_CODES = frozenset(
    {
        "segmentation_leakage",
        "duplicate_source_detection",
        "outside_roi",
        "exclusion_mask_overlap",
        "corrupted_instance_mask",
        "nonfinite_measurement",
        "weak_isolated_noise",
        "confirmed_multi_object_join",
        "confirmed_incorrect_split",
        "tracking_same_z_conflict",
        "tracking_cross_specimen_conflict",
        "tracking_impossible_displacement",
    }
)

CORRECTION_ACTIONS_REQUIRING_TECHNICAL_REASON = frozenset(
    {
        "exclude_false_detection",
        "add_instance",
        "replace_boundary",
        "split_instance",
        "merge_instances",
        "cannot_link",
        "must_link",
    }
)

ACTION_TECHNICAL_REASONS = {
    "exclude_false_detection": frozenset(
        {
            "segmentation_leakage",
            "duplicate_source_detection",
            "outside_roi",
            "exclusion_mask_overlap",
            "corrupted_instance_mask",
            "nonfinite_measurement",
            "weak_isolated_noise",
        }
    ),
    "add_instance": frozenset({"confirmed_incorrect_split"}),
    "replace_boundary": frozenset(
        {"segmentation_leakage", "corrupted_instance_mask"}
    ),
    "split_instance": frozenset({"confirmed_multi_object_join"}),
    "merge_instances": frozenset({"confirmed_incorrect_split"}),
    "must_link": frozenset({"confirmed_incorrect_split"}),
    "cannot_link": frozenset(
        {
            "confirmed_multi_object_join",
            "tracking_same_z_conflict",
            "tracking_cross_specimen_conflict",
            "tracking_impossible_displacement",
        }
    ),
}

# Morphology may be annotated, but it is not an acceptable technical reason
# for excluding a detection in a comparative biological analysis.
FORBIDDEN_MORPHOLOGY_REASONS = frozenset(
    {
        "short",
        "long",
        "wide",
        "thin",
        "curved",
        "tortuous",
        "low length to width ratio",
        "unusual pitch",
        "unusual taper",
        "single slice",
        "fragmented looking",
        "fused looking",
        "irregular",
        "mutant like irregularity",
        "orientation change",
    }
)


def _require_nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_reason(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def _validate_timestamp_utc(value: str) -> str:
    value = _require_nonempty(value, "timestamp_utc")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("timestamp_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("timestamp_utc must include a UTC offset")
    return value


def _validate_sha256(value: str, field_name: str) -> str:
    value = _require_nonempty(str(value), field_name).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
    return value


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    severity: str
    title: str
    detail: str
    action: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_nonempty(self.code, "code"))
        if self.severity not in PREFLIGHT_SEVERITIES:
            raise ValueError(
                "severity must be one of: " + ", ".join(sorted(PREFLIGHT_SEVERITIES))
            )
        object.__setattr__(self, "title", _require_nonempty(self.title, "title"))
        object.__setattr__(self, "detail", _require_nonempty(self.detail, "detail"))
        object.__setattr__(self, "action", _require_nonempty(self.action, "action"))

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass(frozen=True)
class PreflightReport:
    issues: Tuple[PreflightIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = tuple(self.issues)
        if not all(isinstance(issue, PreflightIssue) for issue in normalized):
            raise TypeError("issues must contain only PreflightIssue values")
        object.__setattr__(self, "issues", normalized)

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "block" for issue in self.issues)

    @property
    def blocking_issues(self) -> Tuple[PreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "block")

    @property
    def blocking_summary(self) -> str:
        if not self.blocking_issues:
            return "No blocking preflight issues."
        return "; ".join(f"{issue.code}: {issue.title}" for issue in self.blocking_issues)

    @property
    def issues_summary(self) -> dict[str, int]:
        counts = {severity: 0 for severity in ("block", "warning", "info")}
        for issue in self.issues:
            counts[issue.severity] += 1
        counts["total"] = len(self.issues)
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "summary": self.issues_summary,
            "blocking_summary": self.blocking_summary,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def reduce_study_progress(
    state: Optional[Mapping[str, Any]], event: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a new serializable study-progress state for one event."""

    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    event_type = event.get("event", event.get("type"))
    if event_type not in STUDY_PROGRESS_EVENTS:
        raise ValueError(
            "event must be one of: " + ", ".join(sorted(STUDY_PROGRESS_EVENTS))
        )

    result: dict[str, Any] = copy.deepcopy(dict(state or {}))
    specimens = result.get("specimens", {})
    if not isinstance(specimens, Mapping):
        raise ValueError("state['specimens'] must be a mapping")
    result["specimens"] = copy.deepcopy(dict(specimens))

    specimen_id_value = event.get("specimen_id")
    specimen_id = str(specimen_id_value).strip() if specimen_id_value is not None else ""
    message = str(event.get("message", "")).strip()

    if event_type in {
        "started",
        "slice_progress",
        "postprocess_progress",
        "complete",
        "failed",
        "skipped",
    } and not specimen_id:
        raise ValueError(f"{event_type} events require specimen_id")

    if specimen_id:
        previous = result["specimens"].get(specimen_id, {})
        if not isinstance(previous, Mapping):
            raise ValueError("specimen progress state must be a mapping")
        specimen_state = copy.deepcopy(dict(previous))
        specimen_state["status"] = event_type
        specimen_state["message"] = message
        result["specimens"][specimen_id] = specimen_state

    run_status = {
        "started": "running",
        "slice_progress": "running",
        "postprocess_progress": "running",
        "complete": "running",
        "failed": "running",
        "skipped": "running",
        "reporting": "reporting",
        "stopped": "stopped",
    }[event_type]
    result["status"] = run_status
    result["message"] = message
    result["last_event"] = event_type
    result["current_specimen_id"] = specimen_id or None
    result["counts"] = {
        status: sum(
            1
            for specimen in result["specimens"].values()
            if specimen.get("status") == status
        )
        for status in (
            "started",
            "slice_progress",
            "postprocess_progress",
            "complete",
            "failed",
            "skipped",
        )
    }

    # Fail immediately if an adapter accidentally supplied non-serializable data.
    json.dumps(result, sort_keys=True)
    return result


@dataclass(frozen=True)
class CorrectionEvent:
    correction_uuid: str
    revision: int
    specimen_id: str
    z_index: Optional[int]
    action: str
    source_instance_ids: Tuple[str, ...]
    technical_reason: str
    reviewer: str
    timestamp_utc: str
    base_run_manifest_sha256: str
    software_version: str
    analysis_profile_sha256: str
    checkpoint_sha256: str
    calibration_provenance_sha256: str
    evidence_references: Tuple[str, ...]
    notes: str = ""
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    supersedes: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            UUID(str(self.correction_uuid))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("correction_uuid must be a valid UUID") from exc
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        object.__setattr__(self, "specimen_id", _require_nonempty(self.specimen_id, "specimen_id"))
        if self.z_index is not None and (
            isinstance(self.z_index, bool) or not isinstance(self.z_index, int) or self.z_index < 0
        ):
            raise ValueError("z_index must be a non-negative integer or None")
        if self.action not in CORRECTION_ACTIONS:
            raise ValueError("action must be one of: " + ", ".join(sorted(CORRECTION_ACTIONS)))

        source_ids = tuple(str(value).strip() for value in self.source_instance_ids)
        if any(not value for value in source_ids):
            raise ValueError("source_instance_ids cannot contain empty values")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_instance_ids must be unique within one correction")
        object.__setattr__(self, "source_instance_ids", source_ids)
        source_count = len(source_ids)
        if self.action in {"confirm_instance", "exclude_false_detection", "replace_boundary", "split_instance", "confirm_single_slice"} and source_count != 1:
            raise ValueError(f"{self.action} requires exactly one source instance")
        if self.action in {"merge_instances", "must_link", "cannot_link"} and source_count < 2:
            raise ValueError(f"{self.action} requires at least two source instances")
        if self.action == "add_instance" and source_count != 0:
            raise ValueError("add_instance cannot reference an existing source instance")
        if self.action == "undo" and source_count != 0:
            raise ValueError("undo cannot reference source_instance_ids directly")

        reason = self.technical_reason.strip()
        normalized_reason = _normalize_reason(reason)
        if normalized_reason in FORBIDDEN_MORPHOLOGY_REASONS:
            raise ValueError("morphology alone cannot be used as a technical correction reason")
        if self.action in CORRECTION_ACTIONS_REQUIRING_TECHNICAL_REASON and not reason:
            raise ValueError(f"{self.action} requires a technical_reason")
        if reason and reason not in TECHNICAL_REASON_CODES:
            raise ValueError(
                "technical_reason must be a controlled technical code: "
                + ", ".join(sorted(TECHNICAL_REASON_CODES))
            )
        allowed_reasons = ACTION_TECHNICAL_REASONS.get(self.action)
        if reason and (allowed_reasons is None or reason not in allowed_reasons):
            raise ValueError(
                f"technical_reason {reason!r} is not compatible with action {self.action!r}"
            )
        object.__setattr__(self, "technical_reason", reason)
        object.__setattr__(self, "reviewer", _require_nonempty(self.reviewer, "reviewer"))
        object.__setattr__(self, "timestamp_utc", _validate_timestamp_utc(self.timestamp_utc))
        object.__setattr__(
            self,
            "base_run_manifest_sha256",
            _validate_sha256(self.base_run_manifest_sha256, "base_run_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "analysis_profile_sha256",
            _validate_sha256(self.analysis_profile_sha256, "analysis_profile_sha256"),
        )
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _validate_sha256(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "calibration_provenance_sha256",
            _validate_sha256(
                self.calibration_provenance_sha256,
                "calibration_provenance_sha256",
            ),
        )
        object.__setattr__(
            self,
            "software_version",
            _require_nonempty(str(self.software_version), "software_version"),
        )
        evidence = tuple(str(value).strip() for value in self.evidence_references)
        if any(not value for value in evidence):
            raise ValueError("evidence_references cannot contain empty values")
        if self.action in CORRECTION_ACTIONS_REQUIRING_TECHNICAL_REASON and not evidence:
            raise ValueError(f"{self.action} requires objective evidence_references")
        object.__setattr__(self, "evidence_references", evidence)
        object.__setattr__(self, "notes", str(self.notes))

        for field_name in ("before_hash", "after_hash", "supersedes"):
            value = getattr(self, field_name)
            if value is not None:
                value = _require_nonempty(str(value), field_name)
                object.__setattr__(self, field_name, value)
        if self.before_hash is None or self.after_hash is None:
            raise ValueError("every correction event requires before_hash and after_hash")
        object.__setattr__(self, "before_hash", _validate_sha256(self.before_hash, "before_hash"))
        object.__setattr__(self, "after_hash", _validate_sha256(self.after_hash, "after_hash"))
        if self.action == "undo" and self.supersedes is None:
            raise ValueError("undo requires supersedes")
        if self.supersedes is not None:
            try:
                UUID(self.supersedes)
            except ValueError as exc:
                raise ValueError("supersedes must be a valid UUID") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_uuid": self.correction_uuid,
            "revision": self.revision,
            "specimen_id": self.specimen_id,
            "z_index": self.z_index,
            "action": self.action,
            "source_instance_ids": list(self.source_instance_ids),
            "technical_reason": self.technical_reason,
            "reviewer": self.reviewer,
            "timestamp_utc": self.timestamp_utc,
            "base_run_manifest_sha256": self.base_run_manifest_sha256,
            "software_version": self.software_version,
            "analysis_profile_sha256": self.analysis_profile_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibration_provenance_sha256": self.calibration_provenance_sha256,
            "evidence_references": list(self.evidence_references),
            "notes": self.notes,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorrectionEvent":
        if not isinstance(value, Mapping):
            raise TypeError("correction event record must be a mapping")
        expected = {
            "correction_uuid",
            "revision",
            "specimen_id",
            "z_index",
            "action",
            "source_instance_ids",
            "technical_reason",
            "reviewer",
            "timestamp_utc",
            "base_run_manifest_sha256",
            "software_version",
            "analysis_profile_sha256",
            "checkpoint_sha256",
            "calibration_provenance_sha256",
            "evidence_references",
            "notes",
            "before_hash",
            "after_hash",
            "supersedes",
        }
        unknown = set(value) - expected
        if unknown:
            raise ValueError("unknown correction event fields: " + ", ".join(sorted(unknown)))
        return cls(**dict(value))


def correction_label_state_sha256(
    instance_labels: np.ndarray,
    centerline_labels: np.ndarray,
) -> str:
    """Hash one authoritative filled-label and centerline-label state."""
    instances = np.asarray(instance_labels)
    centerlines = np.asarray(centerline_labels)
    if instances.ndim != 2 or centerlines.ndim != 2:
        raise ValueError("correction label images must be two-dimensional")
    if instances.shape != centerlines.shape:
        raise ValueError("instance and centerline label images must have the same shape")
    if not np.issubdtype(instances.dtype, np.integer) or not np.issubdtype(
        centerlines.dtype, np.integer
    ):
        raise ValueError("correction label images must use integer labels")
    if np.any(instances < 0) or np.any(centerlines < 0):
        raise ValueError("correction label images cannot contain negative labels")
    centerline_pixels = centerlines != 0
    if np.any(instances[centerline_pixels] != centerlines[centerline_pixels]):
        raise ValueError(
            "every centerline pixel must be contained in the same filled instance label"
        )
    digest = hashlib.sha256()
    for name, array in (("instances", instances), ("centerlines", centerlines)):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def preview_false_detection_exclusion(
    instance_labels: np.ndarray,
    centerline_labels: np.ndarray,
    *,
    source_instance_id: str,
    source_label_map: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Preview removal of one technical false detection without changing inputs."""
    instances = np.asarray(instance_labels)
    centerlines = np.asarray(centerline_labels)
    before_hash = correction_label_state_sha256(instances, centerlines)
    source_id = str(source_instance_id).strip()
    labels = tuple(source_label_map.values())
    if len(set(int(value) for value in labels)) != len(labels):
        raise ValueError("source label map values must be unique")
    if source_id not in source_label_map:
        raise KeyError(f"unknown source_instance_id: {source_id}")
    label = source_label_map[source_id]
    if isinstance(label, bool) or not isinstance(label, (int, np.integer)) or int(label) <= 0:
        raise ValueError("source label map values must be positive integer labels")
    label = int(label)
    if not np.any(instances == label):
        raise ValueError(
            f"source instance {source_id!r} label {label} is absent from the current state"
        )
    corrected_instances = np.array(instances, copy=True)
    corrected_centerlines = np.array(centerlines, copy=True)
    corrected_instances[corrected_instances == label] = 0
    corrected_centerlines[corrected_centerlines == label] = 0
    after_hash = correction_label_state_sha256(
        corrected_instances,
        corrected_centerlines,
    )
    return corrected_instances, corrected_centerlines, before_hash, after_hash


def replay_false_detection_exclusions(
    instance_labels: np.ndarray,
    centerline_labels: np.ndarray,
    events: tuple[CorrectionEvent, ...],
    *,
    specimen_id: str,
    z_index: int,
    source_label_map: Mapping[str, int],
    allow_other_planes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the first audited correction subset against authoritative labels.

    Only ``exclude_false_detection`` is executable in this initial replay
    contract. Other actions remain fail-closed until their geometry and
    downstream regeneration paths have independent audit evidence.
    """
    corrected_instances = np.array(instance_labels, copy=True)
    corrected_centerlines = np.array(centerline_labels, copy=True)
    correction_label_state_sha256(corrected_instances, corrected_centerlines)
    validated_events = tuple(events)
    _validate_correction_history(validated_events)
    expected_specimen = str(specimen_id).strip()
    if not expected_specimen:
        raise ValueError("specimen_id must not be blank")
    if isinstance(z_index, bool) or not isinstance(z_index, int) or z_index < 0:
        raise ValueError("z_index must be a non-negative integer")

    original_instances = np.array(instance_labels, copy=True)
    original_centerlines = np.array(centerline_labels, copy=True)
    active_on_plane: list[CorrectionEvent] = []
    event_by_uuid = {event.correction_uuid: event for event in validated_events}
    for event in validated_events:
        if event.action not in {"exclude_false_detection", "undo"}:
            raise ValueError(
                f"correction action {event.action!r} is not executable by this replay version"
            )
        if event.specimen_id != expected_specimen:
            raise ValueError("correction event belongs to a different specimen")
        if event.z_index != z_index:
            # A specimen correction log is global and may contain operations on
            # several planes. The materializer explicitly enables per-plane
            # filtering; direct callers retain the fail-closed wrong-Z check.
            if allow_other_planes:
                continue
            raise ValueError("correction event belongs to a different Z plane")
        if event.action == "undo":
            target = event_by_uuid[event.supersedes]
            before_hash = correction_label_state_sha256(
                corrected_instances, corrected_centerlines
            )
            active_on_plane = [
                active
                for active in active_on_plane
                if active.correction_uuid != target.correction_uuid
            ]
            next_instances = np.array(original_instances, copy=True)
            next_centerlines = np.array(original_centerlines, copy=True)
            for active in active_on_plane:
                next_instances, next_centerlines, _before, _after = (
                    preview_false_detection_exclusion(
                        next_instances,
                        next_centerlines,
                        source_instance_id=active.source_instance_ids[0],
                        source_label_map=source_label_map,
                    )
                )
            after_hash = correction_label_state_sha256(
                next_instances, next_centerlines
            )
            if event.before_hash != before_hash:
                raise ValueError("undo before_hash does not match the current label state")
            if event.after_hash != after_hash:
                raise ValueError("undo after_hash does not match replay from raw evidence")
            corrected_instances = next_instances
            corrected_centerlines = next_centerlines
            continue
        source_id = event.source_instance_ids[0]
        (
            next_instances,
            next_centerlines,
            before_hash,
            after_hash,
        ) = preview_false_detection_exclusion(
            corrected_instances,
            corrected_centerlines,
            source_instance_id=source_id,
            source_label_map=source_label_map,
        )
        if event.before_hash != before_hash:
            raise ValueError("correction before_hash does not match the current label state")
        if event.after_hash != after_hash:
            raise ValueError("correction after_hash does not match the replayed label state")
        corrected_instances = next_instances
        corrected_centerlines = next_centerlines
        active_on_plane.append(event)
    return corrected_instances, corrected_centerlines


def append_correction_event(path: os.PathLike[str] | str, event: CorrectionEvent) -> None:
    """Append one validated JSONL event without rewriting existing bytes."""

    if not isinstance(event, CorrectionEvent):
        raise TypeError("event must be a CorrectionEvent")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_correction_events(destination)
    _validate_correction_history(existing + (event,))
    record = (json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    descriptor = os.open(str(destination), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        remaining = memoryview(record)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("failed to append correction event")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_correction_events(path: os.PathLike[str] | str) -> Tuple[CorrectionEvent, ...]:
    """Load and validate every append-only JSONL correction record."""

    source = Path(path)
    if not source.exists():
        return ()
    events = []
    with source.open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank correction record at line {line_number}")
            try:
                payload = json.loads(line)
                events.append(CorrectionEvent.from_dict(payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid correction record at line {line_number}: {exc}") from exc
    validated = tuple(events)
    _validate_correction_history(validated)
    return validated


def _validate_correction_history(events: Tuple[CorrectionEvent, ...]) -> None:
    """Validate immutable provenance and the append-only revision chain."""

    if not events:
        return
    seen_uuids: set[str] = set()
    seen_events: dict[str, CorrectionEvent] = {}
    active_state_events: list[str] = []
    expected_revision = 1
    provenance_fields = (
        "specimen_id",
        "base_run_manifest_sha256",
        "software_version",
        "analysis_profile_sha256",
        "checkpoint_sha256",
        "calibration_provenance_sha256",
    )
    baseline = tuple(getattr(events[0], name) for name in provenance_fields)
    for event in events:
        if event.correction_uuid in seen_uuids:
            raise ValueError(f"duplicate correction_uuid: {event.correction_uuid}")
        if event.revision != expected_revision:
            raise ValueError(
                f"correction revision {event.revision} does not follow "
                f"expected revision {expected_revision}"
            )
        current = tuple(getattr(event, name) for name in provenance_fields)
        if current != baseline:
            raise ValueError("correction-log provenance cannot change within one log")
        if event.supersedes is not None and event.supersedes not in seen_uuids:
            raise ValueError("supersedes must reference an earlier correction_uuid")
        if event.action == "undo":
            target = seen_events[event.supersedes]
            if target.action == "undo":
                raise ValueError("undo cannot supersede another undo")
            if not active_state_events or active_state_events[-1] != target.correction_uuid:
                raise ValueError("undo may supersede only the most recent active correction")
            if event.z_index != target.z_index:
                raise ValueError("undo must use the same Z plane as its target")
            if event.technical_reason:
                raise ValueError("undo cannot introduce a technical exclusion reason")
            active_state_events.pop()
        else:
            active_state_events.append(event.correction_uuid)
        seen_uuids.add(event.correction_uuid)
        seen_events[event.correction_uuid] = event
        expected_revision += 1


def active_correction_events(
    events: Tuple[CorrectionEvent, ...],
) -> Tuple[CorrectionEvent, ...]:
    """Return active domain corrections after validated latest-only undo events."""
    validated = tuple(events)
    _validate_correction_history(validated)
    active: list[CorrectionEvent] = []
    for event in validated:
        if event.action == "undo":
            if not active or active[-1].correction_uuid != event.supersedes:
                raise ValueError("undo history is inconsistent")
            active.pop()
        else:
            active.append(event)
    return tuple(active)
