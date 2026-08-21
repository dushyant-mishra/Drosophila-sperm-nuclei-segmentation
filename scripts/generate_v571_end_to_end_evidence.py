"""Bundle current v5.7.1 segmentation stages and tracking evidence."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_page(path, title, note):
    image = plt.imread(path)
    figure = plt.figure(figsize=(16, 10))
    axis = figure.add_axes([0.02, 0.07, 0.96, 0.86])
    axis.imshow(image)
    axis.axis("off")
    figure.suptitle(title, fontsize=17, fontweight="bold", y=0.98)
    figure.text(0.5, 0.025, note, ha="center", fontsize=10)
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--tracking-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir).resolve()
    tracking_dir = Path(args.tracking_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "v571_end_to_end_visual_evidence.pdf"
    records = []
    with PdfPages(pdf_path) as pdf:
        for specimen in ("KJ-01", "WT-01"):
            stages = next(stage_dir.glob(f"{specimen}_z*_stages.png"))
            tracking = tracking_dir / f"{specimen}_stratified_tracking.png"
            for role, path, title, note in (
                (
                    "segmentation_stages",
                    stages,
                    f"{specimen}: raw image to measured 2D instances",
                    "All stage panels use the same ROI crop and pixel framing.",
                ),
                (
                    "cross_slice_tracking",
                    tracking,
                    f"{specimen}: cross-slice identity evidence",
                    "Color is fixed within each track; display strokes do not alter measurements.",
                ),
            ):
                figure = image_page(path, title, note)
                pdf.savefig(figure, bbox_inches="tight")
                plt.close(figure)
                records.append(
                    {
                        "specimen": specimen,
                        "role": role,
                        "source_artifact": str(path),
                        "source_artifact_sha256": sha256(path),
                    }
                )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_generation": commit,
        "generator_sha256": sha256(Path(__file__)),
        "pdf": str(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "records": records,
    }
    (output_dir / "end_to_end_visual_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pdf_path)


if __name__ == "__main__":
    main()
