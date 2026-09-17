import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "v5_7_illustrated_workflow" / "figures_v5"
MANIFEST_PATH = FIGURE_DIR / "figure_manifest.json"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_workflow_figure_manifest_matches_rendered_artifacts():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["figures"]

    assert entries
    assert manifest["figures_without_provenance"] == []
    assert len({entry["figure"] for entry in entries}) == len(entries)

    for entry in entries:
        figure = FIGURE_DIR / entry["figure"]
        assert figure.is_file(), entry["figure"]
        assert _sha256(figure) == entry["sha256"], entry["figure"]


def test_workflow_figure_evidence_is_tracked_by_git():
    artifacts = [MANIFEST_PATH]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifacts.extend(FIGURE_DIR / entry["figure"] for entry in manifest["figures"])

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *map(str, artifacts)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_intensity_width_claim_notes_do_not_assert_equivalence():
    registry = json.loads(
        (ROOT / "audits" / "claims_registry.json").read_text(encoding="utf-8")
    )
    claim = next(
        item
        for item in registry["claims"]
        if item["claim_id"] == "MEAS-INTENSITY-WIDTH-001"
    )

    assert "do not establish equivalence" in claim["notes"]
    assert "shared and equal bias" not in claim["notes"]
