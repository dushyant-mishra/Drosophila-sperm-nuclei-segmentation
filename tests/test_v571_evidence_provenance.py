import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_channel_resolution_is_explicit_and_preserves_input():
    package = load_script("package_v571_acceptance_provenance.py")
    source = {
        "ordered_source_images": [
            {"position": 0, "name": "Project001_Series002_z00_ch00.tif", "channel": None}
        ]
    }
    resolved = package.resolve_source_channel_manifest(source)
    assert source["ordered_source_images"][0]["channel"] is None
    record = resolved["ordered_source_images"][0]
    assert record["channel"] == 0
    assert record["channel_resolution_source"] == "filename:explicit_channel"
    assert resolved["channel_selection_rule"] == (
        "all accepted source images resolve to ch00"
    )


def test_source_channel_resolution_rejects_nonzero_channel():
    package = load_script("package_v571_acceptance_provenance.py")
    payload = {
        "ordered_source_images": [
            {"name": "Project001_Series002_z00_ch01.tif", "channel": None}
        ]
    }
    with pytest.raises(ValueError, match="channel 0"):
        package.resolve_source_channel_manifest(payload)


def test_evidence_binding_uses_ancestor_and_blob_continuity():
    validator = load_script("validate_v571_evidence_provenance.py")
    generation_commit = "35396b0e42f9d1afb9a2991c41bf31330a4c89aa"
    reviewed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    recorded = validator.git_blob_sha256(profile, generation_commit)

    validator.require_ancestor(generation_commit, reviewed_commit)
    validator.validate_blob_continuity(
        profile, recorded, generation_commit, reviewed_commit
    )


def test_evidence_binding_rejects_wrong_recorded_blob_hash():
    validator = load_script("validate_v571_evidence_provenance.py")
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    with pytest.raises(ValueError, match="Recorded Git-blob hash mismatch"):
        validator.validate_blob_continuity(profile, "0" * 64, commit, commit)
