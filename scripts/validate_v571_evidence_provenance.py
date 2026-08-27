"""Bind and validate portable v5.7.1 acceptance-evidence manifests."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha256(path, commit):
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    content = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    return hashlib.sha256(content).hexdigest()


def bind_manifest(path, commit, generator, profile=None):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["git_commit_at_generation"] = commit
    payload["generator_git_blob_sha256"] = git_blob_sha256(generator, commit)
    if profile is not None:
        payload["pipeline_git_blob_sha256"] = git_blob_sha256(PIPELINE, commit)
        payload["profile_git_blob_sha256"] = git_blob_sha256(profile, commit)
        for record in payload.get("records", []):
            if isinstance(record, dict) and "profile_git_blob_sha256" in record:
                record["profile_git_blob_sha256"] = payload["profile_git_blob_sha256"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_artifact(repository_path, expected_hash):
    artifact = ROOT / repository_path
    if not artifact.is_file():
        raise FileNotFoundError(f"Manifest artifact is missing: {artifact}")
    actual = sha256(artifact)
    if actual != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {artifact}: {actual} != {expected_hash}")


def validate(stage_manifest, tracking_manifest, end_to_end_manifest, profile):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    expected_pipeline = git_blob_sha256(PIPELINE, commit)
    expected_profile = git_blob_sha256(profile, commit)
    for path in (stage_manifest, tracking_manifest):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("git_commit_at_generation") != commit:
            raise ValueError(f"Commit mismatch in {path}")
        if payload.get("pipeline_git_blob_sha256") != expected_pipeline:
            raise ValueError(f"Pipeline Git-blob hash mismatch in {path}")
        if payload.get("profile_git_blob_sha256") != expected_profile:
            raise ValueError(f"Profile Git-blob hash mismatch in {path}")
        for record in payload.get("records", []):
            if "artifact_repository_path" in record:
                validate_artifact(
                    record["artifact_repository_path"], record["artifact_sha256"]
                )
            if "calibration_metadata_path" in record:
                calibration = Path(record["calibration_metadata_path"])
                if sha256(calibration) != record["calibration_metadata_sha256"]:
                    raise ValueError(f"Calibration XML hash mismatch: {calibration}")
    end_payload = json.loads(end_to_end_manifest.read_text(encoding="utf-8"))
    if end_payload.get("git_commit_at_generation") != commit:
        raise ValueError(f"Commit mismatch in {end_to_end_manifest}")
    validate_artifact(
        end_payload["pdf_repository_path"], end_payload["pdf_sha256"]
    )
    for record in end_payload.get("records", []):
        validate_artifact(
            record["source_artifact_repository_path"],
            record["source_artifact_sha256"],
        )
    return commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--tracking-manifest", required=True, type=Path)
    parser.add_argument("--end-to-end-manifest", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--bind-current-commit", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if args.bind_current_commit:
        bind_manifest(
            args.stage_manifest.resolve(),
            commit,
            ROOT / "scripts" / "generate_v571_stage_evidence.py",
            args.profile.resolve(),
        )
        bind_manifest(
            args.tracking_manifest.resolve(),
            commit,
            ROOT / "scripts" / "generate_v571_stratified_tracking_evidence.py",
            args.profile.resolve(),
        )
        bind_manifest(
            args.end_to_end_manifest.resolve(),
            commit,
            ROOT / "scripts" / "generate_v571_end_to_end_evidence.py",
        )
    commit = validate(
        args.stage_manifest.resolve(),
        args.tracking_manifest.resolve(),
        args.end_to_end_manifest.resolve(),
        args.profile.resolve(),
    )
    print(f"PASS: evidence provenance matches commit {commit}")


if __name__ == "__main__":
    main()
