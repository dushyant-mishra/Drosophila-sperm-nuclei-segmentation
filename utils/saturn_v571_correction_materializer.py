"""Atomic, provenance-bound correction revisions for Saturn v5.7.1.

Only confirmed technical false-detection exclusion is executable here. Other
correction actions remain fail-closed until their geometry and reporting paths
have independent audit evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import tifffile

from utils.saturn_v571_gui_services import (
    CorrectionEvent,
    active_correction_events,
    load_correction_events,
    replay_false_detection_exclusions,
)


ArtifactCallback = Callable[
    [Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]],
    Sequence[os.PathLike[str] | str] | None,
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, value: os.PathLike[str] | str, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the completed specimen output: {path}") from exc
    return path


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{description} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _settings_identity(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "settings" / "settings_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"settings manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path, "settings manifest")
    roles: dict[str, dict[str, Any]] = {}
    for record in manifest.get("files", []):
        if not isinstance(record, dict):
            raise ValueError("settings manifest files must be JSON objects")
        copied = str(record.get("copied_path", "")).strip()
        if not copied:
            continue
        path = _inside(output_dir, copied, "settings artifact")
        if not path.is_file():
            raise FileNotFoundError(f"settings artifact is missing: {path}")
        expected_size = int(record.get("size_bytes", -1))
        expected_hash = str(record.get("sha256", "")).lower()
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_hash:
            raise ValueError(f"settings artifact no longer matches its manifest: {path}")
        role = str(record.get("role", "")).strip()
        if role in roles:
            raise ValueError(f"settings manifest contains duplicate role: {role}")
        roles[role] = {"path": path, "sha256": expected_hash}

    profile_roles = [
        role for role in ("loaded_analysis_profile", "generated_analysis_profile") if role in roles
    ]
    if len(profile_roles) != 1:
        raise ValueError("settings manifest must contain exactly one analysis profile")
    for required in (
        "runtime_parameters",
        "unet_checkpoint",
        "resolved_calibration",
        "source_image_manifest",
    ):
        if required not in roles:
            raise ValueError(f"settings manifest is missing required role: {required}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "profile_sha256": roles[profile_roles[0]]["sha256"],
        "checkpoint_sha256": roles["unet_checkpoint"]["sha256"],
        "calibration_sha256": roles["resolved_calibration"]["sha256"],
        "runtime_parameters_path": roles["runtime_parameters"]["path"],
        "source_image_manifest_path": roles["source_image_manifest"]["path"],
    }


def _verify_event_provenance(
    output_dir: Path,
    events: tuple[CorrectionEvent, ...],
    identity: Mapping[str, Any],
    measurements: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not events:
        raise ValueError("at least one correction event is required")
    expected = {
        "base_run_manifest_sha256": identity["base_manifest_sha256"],
        "analysis_profile_sha256": identity["profile_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "calibration_provenance_sha256": identity["calibration_sha256"],
    }
    pipeline_version = str(identity["manifest"].get("pipeline_version", "")).strip()
    evidence_inventory = []
    required_measurement_columns = {
        "source_instance_key",
        "z_slice",
        "unet_mean_probability",
        "unet_max_probability",
    }
    missing_measurements = required_measurement_columns - set(measurements.columns)
    if missing_measurements:
        raise ValueError(
            "correction evidence validation is missing measurement columns: "
            + ", ".join(sorted(missing_measurements))
        )
    source_manifest = _load_json(
        Path(identity["source_image_manifest_path"]), "source image manifest"
    )
    source_hash_by_z = {
        int(record["z_index"]): str(record.get("sha256", "")).lower()
        for record in source_manifest.get("ordered_source_images", [])
        if record.get("z_index") is not None
    }
    for event in events:
        if event.action not in {"exclude_false_detection", "undo"}:
            raise ValueError(f"correction action {event.action!r} is not materializable")
        if event.software_version != pipeline_version:
            raise ValueError("correction software version does not match the completed run")
        for field, expected_hash in expected.items():
            if getattr(event, field) != expected_hash:
                raise ValueError(f"correction {field} does not match the completed run")
        matching_review_evidence = []
        for reference in event.evidence_references:
            evidence_path = _inside(output_dir, reference, "correction evidence reference")
            if not evidence_path.is_file():
                raise FileNotFoundError(f"correction evidence is missing: {evidence_path}")
            evidence_inventory.append(
                {
                    "relative_path": evidence_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256_file(evidence_path),
                    "size_bytes": evidence_path.stat().st_size,
                }
            )
            if evidence_path.suffix.lower() == ".json":
                payload = _load_json(evidence_path, "correction review evidence")
                if payload.get("schema_version") == "saturn_v571_exclusion_evidence/1.0":
                    matching_review_evidence.append(payload)
        if event.action == "undo":
            continue
        if len(matching_review_evidence) != 1:
            raise ValueError(
                "each exclusion requires exactly one source-bound correction evidence record"
            )
        evidence = matching_review_evidence[0]
        source_id = event.source_instance_ids[0]
        rows = measurements[
            measurements["source_instance_key"].astype(str).eq(source_id)
        ]
        if len(rows) != 1:
            raise ValueError("correction evidence source must match one original detection")
        row = rows.iloc[0]
        expected_values = {
            "source_instance_id": source_id,
            "z_index": int(event.z_index),
            "technical_reason": event.technical_reason,
            "label_state_before_sha256": event.before_hash,
        }
        for field, expected_value in expected_values.items():
            if evidence.get(field) != expected_value:
                raise ValueError(f"correction evidence {field} does not match the event")
        if int(row["z_slice"]) != int(event.z_index):
            raise ValueError("correction evidence Z plane does not match the measurement")
        for field, column in (
            ("observed_unet_mean_probability", "unet_mean_probability"),
            ("observed_unet_max_probability", "unet_max_probability"),
        ):
            if not np.isclose(
                float(evidence.get(field, np.nan)),
                float(row[column]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"correction evidence {field} does not match the measurement")
        source_hash = str(evidence.get("source_image_sha256", "")).lower()
        if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
            raise ValueError("correction evidence requires a source_image_sha256")
        if source_hash_by_z.get(int(event.z_index)) != source_hash:
            raise ValueError(
                "correction evidence source image hash does not match the source manifest"
            )
        if event.technical_reason == "weak_isolated_noise" and evidence.get(
            "reviewer_confirmed_non_nuclear_signal"
        ) is not True:
            raise ValueError(
                "weak isolated noise requires explicit reviewer confirmation of non-nuclear signal"
            )
    return evidence_inventory


def _load_verified_labels(evidence_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = evidence_dir / "evidence.json"
    metadata = _load_json(metadata_path, "authoritative evidence metadata")
    instance_path = evidence_dir / "instance_labels.tif"
    centerline_path = evidence_dir / "centerline_labels.tif"
    for path, hash_field in (
        (instance_path, "instance_labels_sha256"),
        (centerline_path, "centerline_labels_sha256"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"authoritative label evidence is missing: {path}")
        if _sha256_file(path) != str(metadata.get(hash_field, "")).lower():
            raise ValueError(f"authoritative label evidence hash changed: {path}")
    instances = tifffile.imread(instance_path)
    centerlines = tifffile.imread(centerline_path)
    return instances, centerlines


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _inventory_record(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return {
        "relative_path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def create_correction_base_manifest(
    completed_output_dir: os.PathLike[str] | str,
    *,
    specimen_id: str,
) -> tuple[Path, str]:
    """Freeze the completed-run artifacts required for correction replay."""
    output_dir = Path(completed_output_dir).resolve()
    specimen = str(specimen_id).strip()
    if not specimen:
        raise ValueError("specimen_id must not be blank")
    status = _load_json(output_dir / "run_status.json", "run status")
    if str(status.get("status", "")).lower() != "complete":
        raise ValueError("correction base manifests require a completed specimen run")
    identity = _settings_identity(output_dir)
    candidates = {
        output_dir / "run_status.json",
        identity["manifest_path"],
        output_dir / "spermatid_measurements_v5.7.1.csv",
    }
    for record in identity["manifest"].get("files", []):
        copied = str(record.get("copied_path", "")).strip()
        if copied:
            candidates.add(_inside(output_dir, copied, "settings artifact"))
    for pattern in (
        "slice_summary_v5.7.1.csv",
        "measurements_with_tracks_v5.7.1.csv",
        "track_summary_v5.7.1.csv",
    ):
        path = output_dir / pattern
        if path.is_file():
            candidates.add(path)
    candidates.update(
        path
        for path in (output_dir / "raw_evidence" / "instance_labels").rglob("*")
        if path.is_file()
    )
    pipeline_source = Path(__file__).resolve().parents[1] / "sperm_segmentation_saturnv5.7.1.py"
    if not pipeline_source.is_file():
        raise FileNotFoundError(f"v5.7.1 pipeline source is missing: {pipeline_source}")
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError("correction base artifact is missing: " + "; ".join(map(str, missing)))
    artifacts = [_inventory_record(output_dir, path) for path in sorted(candidates)]
    payload = {
        "schema_version": "1.0",
        "specimen_id": specimen,
        "pipeline_version": identity["manifest"].get("pipeline_version"),
        "pipeline_source_path": str(pipeline_source),
        "pipeline_source_sha256": _sha256_file(pipeline_source),
        "settings_manifest_sha256": identity["manifest_sha256"],
        "analysis_profile_sha256": identity["profile_sha256"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "calibration_provenance_sha256": identity["calibration_sha256"],
        "artifacts": artifacts,
    }
    destination = output_dir / "corrections" / "correction_base_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = _load_json(destination, "correction base manifest")
        if existing != payload:
            raise ValueError("existing correction base manifest does not match the completed run")
    else:
        _atomic_json(destination, payload)
    return destination, _sha256_file(destination)


def _verify_correction_base_manifest(output_dir: Path) -> tuple[dict[str, Any], Path, str]:
    path = output_dir / "corrections" / "correction_base_manifest.json"
    payload = _load_json(path, "correction base manifest")
    for record in payload.get("artifacts", []):
        artifact = _inside(output_dir, record.get("relative_path", ""), "correction base artifact")
        if not artifact.is_file():
            raise FileNotFoundError(f"correction base artifact is missing: {artifact}")
        if artifact.stat().st_size != int(record.get("size_bytes", -1)) or _sha256_file(artifact) != str(record.get("sha256", "")).lower():
            raise ValueError(f"correction base artifact changed after review began: {artifact}")
    source = Path(str(payload.get("pipeline_source_path", ""))).resolve()
    if not source.is_file() or _sha256_file(source) != str(payload.get("pipeline_source_sha256", "")).lower():
        raise ValueError("v5.7.1 pipeline source changed after correction review began")
    return payload, path, _sha256_file(path)


@contextmanager
def _revision_lock(corrections_root: Path):
    lock_path = corrections_root / ".correction.lock"
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise RuntimeError("another correction revision is currently being materialized") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def validate_correction_revision(
    revision_dir: os.PathLike[str] | str,
    *,
    expected_base_run_manifest_sha256: str | None = None,
    require_canonical_name: bool = True,
) -> dict[str, Any]:
    """Fail closed if any promoted correction artifact or identity has drifted."""
    revision = Path(revision_dir).resolve()
    completion_path = revision / "revision_complete.json"
    completion = _load_json(completion_path, "correction completion marker")
    manifest = _load_json(revision / "correction_manifest.json", "correction manifest")
    if str(completion.get("status", "")) != "complete":
        raise ValueError("correction revision is not complete")
    expected_revision = int(manifest.get("correction_revision", -1))
    if int(completion.get("correction_revision", -2)) != expected_revision:
        raise ValueError("correction manifest and completion marker revisions differ")
    if require_canonical_name and revision.name != f"revision_{expected_revision:04d}":
        raise ValueError("correction revision directory name does not match its manifest")
    base_hash = str(manifest.get("base_run_manifest_sha256", "")).lower()
    if expected_base_run_manifest_sha256 and base_hash != str(expected_base_run_manifest_sha256).lower():
        raise ValueError("correction revision belongs to a different base run")
    inventory = completion.get("artifacts", [])
    recorded = {str(item.get("relative_path", "")): item for item in inventory}
    actual = {
        path.relative_to(revision).as_posix(): path
        for path in revision.rglob("*")
        if path.is_file() and path != completion_path
    }
    if set(recorded) != set(actual):
        raise ValueError("correction revision artifact inventory is incomplete or has extras")
    for relative, path in actual.items():
        item = recorded[relative]
        if path.stat().st_size != int(item.get("size_bytes", -1)) or _sha256_file(path) != str(item.get("sha256", "")).lower():
            raise ValueError(f"correction revision artifact hash changed: {relative}")
    if int(completion.get("artifact_count", -1)) != len(recorded):
        raise ValueError("correction revision artifact count is inconsistent")
    if _sha256_file(revision / "correction_events.jsonl") != str(manifest.get("correction_events_sha256", "")).lower():
        raise ValueError("correction event log hash does not match the revision manifest")
    return {"completion": completion, "manifest": manifest}


def materialize_false_detection_revision(
    completed_output_dir: os.PathLike[str] | str,
    correction_log_path: os.PathLike[str] | str,
    *,
    retrack_callable: Callable[..., tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]],
    artifact_callback: ArtifactCallback,
    tracking_callable: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> Path:
    """Create one immutable corrected revision or leave no partial revision.

    ``artifact_callback`` must regenerate the user-facing summary/report set in
    the temporary revision and return its paths. The revision is promoted only
    after every returned artifact exists and the complete inventory is hashed.
    """
    output_dir = Path(completed_output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"completed specimen output is missing: {output_dir}")
    run_status = _load_json(output_dir / "run_status.json", "run status")
    if str(run_status.get("status", "")).lower() != "complete":
        raise ValueError("corrections require a completed specimen run")
    events = load_correction_events(correction_log_path)
    identity = _settings_identity(output_dir)
    base_manifest, base_manifest_path, base_manifest_sha256 = _verify_correction_base_manifest(output_dir)
    identity = dict(identity)
    identity["base_manifest_sha256"] = base_manifest_sha256
    if str(base_manifest.get("specimen_id", "")) != events[0].specimen_id:
        raise ValueError("correction log specimen does not match the correction base manifest")
    measurements_path = output_dir / "spermatid_measurements_v5.7.1.csv"
    if not measurements_path.is_file():
        raise FileNotFoundError(f"authoritative 2D measurement table is missing: {measurements_path}")
    measurements = pd.read_csv(measurements_path)
    correction_evidence = _verify_event_provenance(
        output_dir, events, identity, measurements
    )
    specimen_ids = {event.specimen_id for event in events}
    if len(specimen_ids) != 1:
        raise ValueError("one correction log must belong to exactly one specimen")
    specimen_id = next(iter(specimen_ids))
    revision = events[-1].revision

    corrections_root = output_dir / "corrections"
    final_dir = corrections_root / f"revision_{revision:04d}"
    corrections_root.mkdir(parents=True, exist_ok=True)
    with _revision_lock(corrections_root):
      if final_dir.exists():
        raise FileExistsError(f"correction revision already exists: {final_dir}")
      current_path = corrections_root / "CURRENT.json"
      parent_identity = None
      if current_path.exists():
        current = _load_json(current_path, "current correction pointer")
        current_revision = int(current.get("correction_revision", -1))
        if revision != current_revision + 1:
            raise ValueError("correction revision does not follow CURRENT.json")
        parent_dir = corrections_root / str(current.get("revision_path", ""))
        parent = validate_correction_revision(
            parent_dir,
            expected_base_run_manifest_sha256=base_manifest_sha256,
        )
        parent_completion_path = parent_dir / "revision_complete.json"
        if _sha256_file(parent_completion_path) != str(
            current.get("revision_complete_sha256", "")
        ).lower():
            raise ValueError("CURRENT.json completion hash does not match its revision")
        parent_identity = {
            "parent_revision": current_revision,
            "parent_manifest_sha256": _sha256_file(
                parent_dir / "correction_manifest.json"
            ),
            "parent_completion_sha256": _sha256_file(parent_completion_path),
        }
      elif revision != 1:
        raise ValueError("the first materialized correction revision must be 1")
      temp_dir = corrections_root / f".revision_{revision:04d}.tmp-{os.getpid()}-{time.time_ns()}"
      temp_dir.mkdir()
      promoted = False
      try:
        source_log_hash = _sha256_file(Path(correction_log_path))
        staged_log = temp_dir / "correction_events.jsonl"
        shutil.copyfile(correction_log_path, staged_log)
        if _sha256_file(staged_log) != source_log_hash or _sha256_file(Path(correction_log_path)) != source_log_hash:
            raise RuntimeError("correction event log changed while its snapshot was captured")
        label_output = temp_dir / "instance_labels"
        label_output.mkdir()
        evidence_root = output_dir / "raw_evidence" / "instance_labels"
        evidence_dirs = sorted(path for path in evidence_root.glob("z[0-9][0-9][0-9][0-9]") if path.is_dir())
        if not evidence_dirs:
            raise ValueError("authoritative per-slice instance evidence is unavailable")
        for evidence_dir in evidence_dirs:
            z_index = int(evidence_dir.name[1:])
            instances, centerlines = _load_verified_labels(evidence_dir)
            labels = [int(value) for value in np.unique(instances) if int(value) > 0]
            source_map = {f"z{z_index:04d}:instance:{label}": label for label in labels}
            corrected_instances, corrected_centerlines = replay_false_detection_exclusions(
                instances,
                centerlines,
                events,
                specimen_id=specimen_id,
                z_index=z_index,
                source_label_map=source_map,
                allow_other_planes=True,
            )
            plane_dir = label_output / evidence_dir.name
            plane_dir.mkdir()
            tifffile.imwrite(plane_dir / "instance_labels.tif", corrected_instances)
            tifffile.imwrite(plane_dir / "centerline_labels.tif", corrected_centerlines)

        runtime_cfg = _load_json(Path(identity["runtime_parameters_path"]), "runtime parameters")
        tracked, tracks, retrack_manifest = retrack_callable(
            measurements,
            events,
            runtime_cfg,
            specimen_id=specimen_id,
            tracking_callable=tracking_callable,
        )
        active_events = active_correction_events(events)
        excluded = {
            source for event in active_events for source in event.source_instance_ids
        }
        corrected_2d = measurements[
            ~measurements["source_instance_key"].astype(str).isin(excluded)
        ].copy()
        corrected_2d.to_csv(temp_dir / "spermatid_measurements_v5.7.1.csv", index=False)
        tracked.to_csv(temp_dir / "measurements_with_tracks_v5.7.1.csv", index=False)
        tracks.to_csv(temp_dir / "track_summary_v5.7.1.csv", index=False)
        revision_manifest = {
            "schema_version": "1.0",
            "pipeline_version": identity["manifest"].get("pipeline_version"),
            "specimen_id": specimen_id,
            "correction_revision": revision,
            "base_output_dir": str(output_dir),
            "base_run_manifest_path": str(base_manifest_path),
            "base_run_manifest_sha256": base_manifest_sha256,
            "analysis_profile_sha256": identity["profile_sha256"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "calibration_provenance_sha256": identity["calibration_sha256"],
            "event_count": len(events),
            "correction_evidence": correction_evidence,
            "correction_events_sha256": _sha256_file(
                temp_dir / "correction_events.jsonl"
            ),
            "retracking": dict(retrack_manifest),
            "original_outputs_overwritten": False,
        }
        if parent_identity is None:
            revision_manifest.update(
                {
                    "parent_revision": 0,
                    "parent_manifest_sha256": base_manifest_sha256,
                    "parent_completion_sha256": None,
                }
            )
        else:
            revision_manifest.update(parent_identity)
        _atomic_json(temp_dir / "correction_manifest.json", revision_manifest)
        generated = artifact_callback(temp_dir, corrected_2d, tracked, tracks, revision_manifest)
        generated_paths = [] if generated is None else list(generated)
        if not generated_paths:
            raise ValueError("artifact callback did not regenerate any report artifacts")
        for value in generated_paths:
            path = _inside(temp_dir, value, "generated correction artifact")
            if not path.is_file():
                raise FileNotFoundError(f"generated correction artifact is missing: {path}")

        inventory = []
        for path in sorted(candidate for candidate in temp_dir.rglob("*") if candidate.is_file()):
            inventory.append(
                {
                    "relative_path": path.relative_to(temp_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        completion = {
            "schema_version": "1.0",
            "status": "complete",
            "specimen_id": specimen_id,
            "correction_revision": revision,
            "artifact_count": len(inventory),
            "artifacts": inventory,
        }
        _atomic_json(temp_dir / "revision_complete.json", completion)
        validate_correction_revision(
            temp_dir,
            expected_base_run_manifest_sha256=base_manifest_sha256,
            require_canonical_name=False,
        )
        temp_dir.rename(final_dir)
        promoted = True
        completion_path = final_dir / "revision_complete.json"
        _atomic_json(
            current_path,
            {
                "schema_version": "1.0",
                "correction_revision": revision,
                "revision_path": final_dir.name,
                "revision_complete_sha256": _sha256_file(completion_path),
            },
        )
      except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if promoted and final_dir.exists():
            published_revision = -1
            if current_path.exists():
                try:
                    published_revision = int(
                        _load_json(current_path, "current correction pointer").get(
                            "correction_revision", -1
                        )
                    )
                except Exception:
                    published_revision = -1
            if published_revision != revision:
                shutil.rmtree(final_dir)
        raise
    return final_dir
