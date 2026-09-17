import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "v5_7_illustrated_workflow" / "figures_v5"
MANIFEST_PATH = FIGURE_DIR / "figure_manifest.json"
DOCUMENT_PATH = ROOT / "Saturn_V5.7.1_Illustrated_Technical_Workflow_v5.docx"
DOCUMENT_BUILDER = ROOT / "scripts" / "build_v571_workflow_v5_document.py"


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


def test_document_embeds_the_tracked_figures():
    """The shipped document must contain the figures this manifest vouches for.

    Manifest-to-file agreement says nothing about the artifact a reader opens.
    Regenerating a figure without rebuilding the document ships a stale picture
    under a caption written for the new one, which is the same class of defect
    that left a stale digest in the manifest.
    """
    builder = DOCUMENT_BUILDER.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r'"(v5_fig[\w.]+\.png)"', builder)))
    assert referenced, "no figures referenced by the document builder"

    with zipfile.ZipFile(DOCUMENT_PATH) as archive:
        embedded = {
            hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("word/media/")
        }

    stale = [
        name
        for name in referenced
        if _sha256(FIGURE_DIR / name) not in embedded
    ]
    assert not stale, (
        "document does not embed the current figure for: "
        + ", ".join(stale)
        + ". Rebuild with scripts/build_v571_workflow_v5_document.py."
    )
