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


def require_ancestor(ancestor, descendant):
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Evidence generation commit {ancestor} is not an ancestor of "
            f"reviewed commit {descendant}"
        )


def validate_blob_continuity(path, recorded_hash, generation_commit, reviewed_commit):
    generated_hash = git_blob_sha256(path, generation_commit)
    reviewed_hash = git_blob_sha256(path, reviewed_commit)
    if recorded_hash != generated_hash:
        raise ValueError(
            f"Recorded Git-blob hash mismatch for {path}: "
            f"{recorded_hash} != {generated_hash}"
        )
    if reviewed_hash != generated_hash:
        raise ValueError(
            f"Git-blob changed after evidence generation for {path}: "
            f"{generated_hash} != {reviewed_hash}"
        )


def validate_artifact(repository_path, expected_hash):
    artifact = ROOT / repository_path
    if not artifact.is_file():
        raise FileNotFoundError(f"Manifest artifact is missing: {artifact}")
    actual = sha256(artifact)
    if actual != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {artifact}: {actual} != {expected_hash}")


def validate(stage_manifest, tracking_manifest, end_to_end_manifest, profile):
    reviewed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    generators = {
        stage_manifest: ROOT / "scripts" / "generate_v571_stage_evidence.py",
        tracking_manifest: ROOT / "scripts" / "evaluate_v571_tracking_replay.py",
    }
    generation_commits = set()
    for path, generator in generators.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        generation_commit = payload.get("git_commit_at_generation")
        if not generation_commit:
            raise ValueError(f"Missing generation commit in {path}")
        require_ancestor(generation_commit, reviewed_commit)
        generation_commits.add(generation_commit)
        validate_blob_continuity(
            PIPELINE,
            payload.get("pipeline_git_blob_sha256"),
            generation_commit,
            reviewed_commit,
        )
        validate_blob_continuity(
            profile,
            payload.get("profile_git_blob_sha256"),
            generation_commit,
            reviewed_commit,
        )
        validate_blob_continuity(
            generator,
            payload.get("generator_git_blob_sha256"),
            generation_commit,
            reviewed_commit,
        )
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
    end_generation_commit = end_payload.get("git_commit_at_generation")
    if not end_generation_commit:
        raise ValueError(f"Missing generation commit in {end_to_end_manifest}")
    require_ancestor(end_generation_commit, reviewed_commit)
    generation_commits.add(end_generation_commit)
    validate_blob_continuity(
        ROOT / "scripts" / "generate_v571_end_to_end_evidence.py",
        end_payload.get("generator_git_blob_sha256"),
        end_generation_commit,
        reviewed_commit,
    )
    validate_artifact(
        end_payload["pdf_repository_path"], end_payload["pdf_sha256"]
    )
    for record in end_payload.get("records", []):
        validate_artifact(
            record["source_artifact_repository_path"],
            record["source_artifact_sha256"],
        )
    return reviewed_commit, sorted(generation_commits)


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
        print(
            "NOTE: --bind-current-commit is retained for CLI compatibility but "
            "does not rewrite generation provenance. Content continuity is "
            "validated instead."
        )
    reviewed_commit, generation_commits = validate(
        args.stage_manifest.resolve(),
        args.tracking_manifest.resolve(),
        args.end_to_end_manifest.resolve(),
        args.profile.resolve(),
    )
    print(
        "PASS: evidence artifacts remain content-identical from generation "
        f"commit(s) {', '.join(generation_commits)} through reviewed commit "
        f"{reviewed_commit}"
    )


if __name__ == "__main__":
    main()
